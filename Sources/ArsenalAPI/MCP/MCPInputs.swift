// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import os
import ArsenalCore
import MCP

/// One input file that cannot be cloned (another volume): copied inside the job, with progress.
struct MCPCopy: Sendable, Equatable {
    let source: String
    let destination: URL
    let bytes: Int64
}

/// File inputs over MCP. A `{path: …}` in a run's input, or a path-type setting, may name any regular file on this Mac:
/// Arsenal clones it into a fresh inbox folder (mode 0600, pruned after 24 h) and the model reads the clone; across
/// volumes the job copies it first. Arsenal may hold macOS privacy grants (Desktop, Documents, Downloads) the calling
/// process lacks, so the rules are strict: the file is opened without following a final symlink, judged by `fstat` and
/// its real path (`F_GETPATH`), devices and Arsenal's own support folder are refused, hard links into the support folder
/// are refused, and paths inside the inbox or a job are rewritten to their real form. All of it runs off the router
/// actor, against a deadline, so a privacy prompt or a slow disk never stalls other requests.
extension APIRouter {
    nonisolated var inboxRoot: URL { support.appendingPathComponent("inbox") }
    nonisolated var mcpJobsRoot: URL { support.appendingPathComponent("jobs", isDirectory: true) }

    /// Checks every path in `value` and, when `importing`, clones external files into the inbox and rewrites the paths.
    /// Returns the rewritten value and the copies a job must still make.
    nonisolated func mcpPrepareInputs(_ value: JSONValue, environment: MCPEnvironment, importing: Bool, field: String = "input") async throws -> (JSONValue, [MCPCopy]) {
        let support = support
        return try await mcpWithDeadline(environment) {
            var copies: [MCPCopy] = []
            let rewritten = try APIRouter.mcpWalk(value, field: field, support: support, environment: environment, importing: importing, copies: &copies)
            return (rewritten, copies)
        }
    }
    nonisolated static func mcpWalk(
        _ value: JSONValue, field: String, support: URL, environment: MCPEnvironment, importing: Bool, copies: inout [MCPCopy]
    ) throws -> JSONValue {
        switch value {
        case .object(var object):
            for key in object.keys.sorted() {
                if key == "path", let path = object[key]?.string {
                    object[key] = .string(try mcpInput(path, field: field + ".path", support: support, environment: environment, importing: importing, copies: &copies))
                } else if let child = object[key] {
                    object[key] = try mcpWalk(child, field: field + "." + key, support: support, environment: environment, importing: importing, copies: &copies)
                }
            }
            return .object(object)
        case .array(let values):
            return .array(
                try values.enumerated().map {
                    try mcpWalk($0.element, field: field + "[\($0.offset)]", support: support, environment: environment, importing: importing, copies: &copies)
                })
        default: return value
        }
    }
    /// The real path (`/private/var/…`), also for a path that does not exist yet: its nearest existing folder's real
    /// path plus the rest.
    nonisolated static func mcpRealPath(_ url: URL) -> String {
        var existing = url.standardizedFileURL, rest: [String] = []
        var buffer = [CChar](repeating: 0, count: Int(PATH_MAX))
        while realpath(existing.path, &buffer) == nil {
            guard existing.path != "/" else { return url.standardizedFileURL.path }
            rest.insert(existing.lastPathComponent, at: 0); existing.deleteLastPathComponent()
        }
        return rest.reduce(String(cString: buffer)) { $0 + "/" + $1 }
    }
    /// One path: checked, and when importing, cloned (or queued for a copy). Returns the path the model will read.
    nonisolated static func mcpInput(
        _ path: String, field: String, support: URL, environment: MCPEnvironment, importing: Bool, copies: inout [MCPCopy]
    ) throws -> String {
        guard path.hasPrefix("/") else { throw MCPFailure(.invalidInput, "Not run: \(field) must be an absolute path to a file on this Mac (\"\(path)\" is not).") }
        func isDevice(_ text: String) -> Bool { text == "/dev" || text.hasPrefix("/dev/") || text == "/private/dev" || text.hasPrefix("/private/dev/") }
        guard !isDevice(URL(fileURLWithPath: path).standardizedFileURL.path) else {
            throw MCPFailure(.invalidInput, "Not run: \(field) points at a device (/dev), which is not an input file.")
        }
        let resolved = mcpRealPath(URL(fileURLWithPath: path))
        let descriptor = open(resolved, O_RDONLY | O_NOFOLLOW | O_NONBLOCK | O_CLOEXEC)
        guard descriptor >= 0 else {
            switch errno {
            case ENOENT, ENOTDIR: throw MCPFailure(.invalidInput, "Not run: \(path) does not exist or Arsenal cannot see it.")
            case ELOOP: throw MCPFailure(.invalidInput, "Not run: \(path) changed into a symlink while it was checked; pass the file itself.")
            default:
                throw MCPFailure(
                    .invalidInput,
                    "Not run: Arsenal cannot read \(path) (\(String(cString: strerror(errno)))). macOS privacy settings may keep Arsenal out of that folder; move the file elsewhere or send it with \(MCPNames.prefix)upload."
                )
            }
        }
        defer { close(descriptor) }
        var info = stat()
        guard fstat(descriptor, &info) == 0 else { throw MCPFailure(.invalidInput, "Not run: \(path) cannot be read.") }
        guard info.st_mode & S_IFMT == S_IFREG else {
            throw MCPFailure(.invalidInput, "Not run: \(path) is not a regular file\((info.st_mode & S_IFMT) == S_IFDIR ? " (it is a folder)" : ""). Pass a file.")
        }
        var buffer = [CChar](repeating: 0, count: Int(MAXPATHLEN))
        guard fcntl(descriptor, F_GETPATH, &buffer) == 0 else { throw MCPFailure(.invalidInput, "Not run: \(path) cannot be read.") }
        let real = String(cString: buffer)
        guard !isDevice(real) else { throw MCPFailure(.invalidInput, "Not run: \(field) points at a device (/dev), which is not an input file.") }
        let supportReal = mcpRealPath(support)
        let roots = [mcpRealPath(support.appendingPathComponent("inbox")), mcpRealPath(support.appendingPathComponent("jobs"))].map { $0 + "/" }
        let inside = roots.contains { real.hasPrefix($0) }
        let refusal = MCPFailure(.invalidInput, "Not run: \(field) points into Arsenal's own support folder, which is not an input location.")
        if inside {
            // A hard link in the inbox could name the token; inbox and job files are used in place, by their real path.
            guard info.st_nlink == 1 else { throw MCPFailure(.invalidInput, "Not run: \(path) is a hard link; pass the file itself.") }
            return real
        }
        guard real != supportReal, !real.hasPrefix(supportReal + "/") else { throw refusal }
        if info.st_nlink > 1 {
            for secret in ["mcp-token", "config.json"] {
                var other = stat()
                if stat(support.appendingPathComponent(secret).path, &other) == 0, other.st_dev == info.st_dev, other.st_ino == info.st_ino { throw refusal }
            }
        }
        guard importing else { return path }
        let inbox = URL(fileURLWithPath: mcpRealPath(support)).appendingPathComponent("inbox")
        guard (try? inbox.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw MCPFailure(.failed, "Not run: the Arsenal inbox is a symlink; it must be a folder.")
        }
        let folder = inbox.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let destination = folder.appendingPathComponent(URL(fileURLWithPath: real).lastPathComponent)
        let reason = environment.cloneFile(descriptor, destination.path)
        if reason == 0 {
            _ = chmod(destination.path, 0o600)
        } else if reason == EXDEV || reason == ENOTSUP || reason == EEXIST {
            copies.append(MCPCopy(source: real, destination: destination, bytes: Int64(info.st_size)))
        } else {
            try? FileManager.default.removeItem(at: folder)
            throw MCPFailure(
                .invalidInput,
                "Not run: Arsenal cannot read \(path) (\(String(cString: strerror(reason)))). macOS privacy settings may keep Arsenal out of that folder; move the file elsewhere or send it with \(MCPNames.prefix)upload."
            )
        }
        return destination.path
    }
    /// Refuses copies the startup disk cannot hold, removing their empty inbox folders.
    nonisolated func mcpRequireCopyDisk(_ copies: [MCPCopy], environment: MCPEnvironment) throws {
        let bytes = copies.reduce(0) { $0 + $1.bytes }
        guard bytes > 0, let free = environment.freeDiskBytes(support), free < bytes else { return }
        for copy in copies { try? FileManager.default.removeItem(at: copy.destination.deletingLastPathComponent()) }
        throw MCPFailure(.disk, "Not run: copying the input files needs \(MCPFormat.size(bytes)), \(MCPFormat.size(free)) free on the startup disk.")
    }
    /// Copies files that could not be cloned, in 8 MiB chunks, reporting progress to `job` (the "copying" phase).
    nonisolated static func mcpCopyInputs(_ copies: [MCPCopy], job: MCPJob?) async throws {
        let total = copies.reduce(0) { $0 + $1.bytes }
        var done: Int64 = 0
        for copy in copies {
            let source = open(copy.source, O_RDONLY | O_NOFOLLOW | O_CLOEXEC)
            guard source >= 0 else { throw MCPFailure(.invalidInput, "Not run: \(copy.source) could not be opened for copying.") }
            defer { close(source) }
            var info = stat()
            guard fstat(source, &info) == 0, info.st_mode & S_IFMT == S_IFREG else { throw MCPFailure(.invalidInput, "Not run: \(copy.source) is no longer a regular file.") }
            let target = open(copy.destination.path, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
            guard target >= 0 else { throw MCPFailure(.failed, "Not run: Arsenal cannot write its copy of \(copy.source).") }
            var ok = false
            defer {
                close(target)
                if !ok { try? FileManager.default.removeItem(at: copy.destination) }
            }
            var chunk = [UInt8](repeating: 0, count: 8 * 1024 * 1024)
            while true {
                try Task.checkCancellation()
                let count = read(source, &chunk, chunk.count)
                guard count >= 0 else { throw MCPFailure(.invalidInput, "Not run: reading \(copy.source) failed (\(String(cString: strerror(errno)))).") }
                if count == 0 { break }
                var offset = 0
                while offset < count {
                    let written = chunk.withUnsafeBytes { write(target, $0.baseAddress! + offset, count - offset) }
                    guard written > 0 else { throw MCPFailure(.disk, "Not run: copying \(copy.source) failed (\(String(cString: strerror(errno)))).") }
                    offset += written
                }
                done += Int64(count)
                await job?.copying(done, of: total)
            }
            ok = true
        }
    }
    /// Path-type settings take the same checks, clones and copies as run inputs (a copy happens before the call answers).
    func mcpImportSettingPaths(id: String, values: JSONObject) async throws -> JSONObject {
        var values = values
        let environment = mcpEnvironment
        for setting in try await service.settings(id: id) {
            let mapping = setting.declaration["maps_to"]?.string ?? ""
            guard setting.declaration["type"]?.string == "path" || mapping.hasSuffix(".path") || setting.id.hasSuffix("_path"),
                let path = values[setting.id]?.string, !path.isEmpty
            else { continue }
            let (prepared, copies) = try await mcpPrepareInputs(["path": .string(path)], environment: environment, importing: true, field: "set." + setting.id)
            try mcpRequireCopyDisk(copies, environment: environment)
            try await Self.mcpCopyInputs(copies, job: nil)
            values[setting.id] = prepared["path"] ?? .string(path)
        }
        return values
    }
    nonisolated func pruneMCPInbox(now: Date = Date()) {
        let root = inboxRoot
        guard (try? root.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else { return }
        for entry in (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: [.contentModificationDateKey, .isDirectoryKey, .isSymbolicLinkKey])) ?? []
        {
            guard let info = try? entry.resourceValues(forKeys: [.contentModificationDateKey, .isDirectoryKey, .isSymbolicLinkKey]),
                info.isDirectory == true, info.isSymbolicLink != true, let modified = info.contentModificationDate, modified < now.addingTimeInterval(-86400)
            else { continue }
            try? FileManager.default.removeItem(at: entry)
        }
    }
    /// `arsenal_upload`: base64 data from an agent whose file is not on this Mac.
    func mcpUpload(_ body: JSONObject) throws -> CallTool.Result {
        pruneMCPInbox()
        let name = body["filename"]?.string ?? ""
        guard !name.isEmpty, name == URL(fileURLWithPath: name).lastPathComponent, !name.hasPrefix("."), name.count <= 128 else {
            throw MCPFailure(.invalidInput, "Not stored: filename must be a plain file name (no folders, not hidden, up to 128 characters).")
        }
        guard let data = Data(base64Encoded: body["data"]?.string ?? ""), !data.isEmpty else { throw MCPFailure(.invalidInput, "Not stored: data must be nonempty base64.") }
        guard data.count <= 700 * 1024 else {
            throw MCPFailure(.limit, "Not stored: uploads are limited to 700 KiB. A file already on this Mac needs no upload: pass its absolute path to \(MCPNames.prefix)run.")
        }
        guard (try? inboxRoot.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw MCPFailure(.failed, "Not stored: the Arsenal inbox is a symlink; it must be a folder.")
        }
        let root = inboxRoot.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let path = root.appendingPathComponent(name)
        try data.write(to: path, options: .atomic); try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: path.path)
        return mcpText("Stored \(name) (\(MCPFormat.size(Int64(data.count)))) in the Arsenal inbox for 24 hours: \(path.path)\nPass it as {\"path\":\"\(path.path)\"}.")
    }
}

/// Runs blocking file work off the actor; refuses when it outlasts the deadline (a privacy prompt waiting for the user,
/// a sleeping disk). The work itself cannot be interrupted and finishes in the background.
func mcpWithDeadline<T: Sendable>(_ environment: MCPEnvironment, _ work: @escaping @Sendable () throws -> T) async throws -> T {
    let once = OSAllocatedUnfairLock(initialState: false)
    return try await withCheckedThrowingContinuation { (continuation: CheckedContinuation<T, any Error>) in
        func finish(_ result: Result<T, any Error>) {
            guard
                once.withLock({ done in
                    if done { return false }; done = true; return true
                })
            else { return }
            continuation.resume(with: result)
        }
        Task.detached { finish(Result { try work() }) }
        Task.detached {
            try? await environment.sleep(environment.pathDeadlineSeconds)
            finish(
                .failure(
                    MCPFailure(
                        .invalidInput,
                        "Not run: reading the input files took longer than \(Int(environment.pathDeadlineSeconds)) s. macOS may be asking the user to let Arsenal into that folder; ask them to answer it, or move the file."
                    )))
        }
    }
}
