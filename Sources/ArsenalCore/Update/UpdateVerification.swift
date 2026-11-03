// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import Security

public struct StagedArsenalUpdate: Codable, Sendable {
    public let directory: URL
    public let archive: URL
    public let app: URL
    public let version: String
    public let sha256: String
}

public enum UpdateVerification {
    public static func sha256(_ file: URL) throws -> String {
        let handle = try FileHandle(forReadingFrom: file)
        defer { try? handle.close() }
        var hasher = SHA256()
        while let block = try handle.read(upToCount: 1024 * 1024), !block.isEmpty { hasher.update(data: block) }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }
    public static func expectedHash(_ sums: String, name: String) throws -> String {
        let hashes = sums.split(whereSeparator: \.isNewline).compactMap { line -> String? in
            let fields = line.split(whereSeparator: \.isWhitespace)
            guard fields.count == 2, fields[1] == Substring(name) || fields[1] == Substring("*" + name) else { return nil }
            let hash = String(fields[0]).lowercased()
            return hash.count == 64 && hash.allSatisfy(\.isHexDigit) ? hash : nil
        }
        guard hashes.count == 1, let hash = hashes.first else { throw UpdateFailure("SHA256SUMS has no unique checksum for the app archive.") }
        return hash
    }

    /// Certificate identity is required; an ad-hoc signature is never sufficient to authorize an update.
    public static func identity(downloaded: URL, installed: URL) throws {
        func code(_ app: URL) throws -> SecStaticCode {
            var value: SecStaticCode?
            guard SecStaticCodeCreateWithPath(app as CFURL, [], &value) == errSecSuccess, let value else { throw UpdateFailure("Cannot read the app's code signature.") }
            return value
        }
        let current = try code(installed), replacement = try code(downloaded)
        let flags = SecCSFlags(rawValue: UInt32(kSecCSCheckAllArchitectures | kSecCSStrictValidate | kSecCSCheckNestedCode))
        guard SecStaticCodeCheckValidity(current, flags, nil) == errSecSuccess else { throw UpdateFailure("The installed app's signature is invalid.") }
        var info: CFDictionary?
        guard SecCodeCopySigningInformation(current, SecCSFlags(rawValue: kSecCSSigningInformation), &info) == errSecSuccess, let details = info as? [String: Any],
            let certificates = details[kSecCodeInfoCertificates as String] as? [SecCertificate], !certificates.isEmpty
        else { throw UpdateFailure("The installed app has no certificate identity to pin updates to.") }
        var requirement: SecRequirement?
        guard SecCodeCopyDesignatedRequirement(current, [], &requirement) == errSecSuccess, let pinned = requirement else {
            throw UpdateFailure("The installed app has no designated signing requirement.")
        }
        guard SecStaticCodeCheckValidity(replacement, flags, pinned) == errSecSuccess else {
            throw UpdateFailure("The update's code signature does not match the installed Arsenal identity.")
        }
    }

    public static func bundle(_ app: URL, version: String) throws {
        let infoURL = app.appendingPathComponent("Contents/Info.plist")
        guard let info = try PropertyListSerialization.propertyList(from: Data(contentsOf: infoURL), format: nil) as? [String: Any],
            info["CFBundleIdentifier"] as? String == "dev.arsenal.app", info["CFBundleExecutable"] as? String == "Arsenal",
            info["CFBundleShortVersionString"] as? String == version
        else { throw UpdateFailure("The app's identity or version does not match the release.") }
        for relative in ["Contents/MacOS/Arsenal", "Contents/Helpers/ArsenalUpdateHelper"] {
            guard FileManager.default.isExecutableFile(atPath: app.appendingPathComponent(relative).path) else { throw UpdateFailure("The update is missing \(relative).") }
        }
        let root = app.standardizedFileURL.path + "/"
        if let entries = FileManager.default.enumerator(at: app, includingPropertiesForKeys: [.isSymbolicLinkKey]) {
            for case let entry as URL in entries where try entry.resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink == true {
                guard entry.resolvingSymlinksInPath().path.hasPrefix(root) else { throw UpdateFailure("The archive contains a link outside Arsenal.app.") }
            }
        }
    }

    public static func prepare(_ release: UpdateRelease, network: UpdateNetwork, installed: URL, progress: @escaping @Sendable (Double) -> Void) async throws -> StagedArsenalUpdate
    {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false, attributes: [.posixPermissions: 0o700])
        do {
            let sums = directory.appendingPathComponent("SHA256SUMS"), archive = directory.appendingPathComponent(release.zipName)
            try await network.download(release.checksums, to: sums, maximum: 64 * 1024)
            let expected = try expectedHash(String(contentsOf: sums, encoding: .utf8), name: release.zipName)
            try await network.download(release.archive, to: archive, maximum: UpdateNetwork.maximumArchiveBytes, progress: progress)
            guard try sha256(archive) == expected else { throw UpdateFailure("The downloaded app failed its SHA256 check.") }
            try archiveEntries(archive)
            let extracted = directory.appendingPathComponent("unpacked")
            try FileManager.default.createDirectory(at: extracted, withIntermediateDirectories: false)
            _ = try tool("/usr/bin/ditto", ["-x", "-k", archive.path, extracted.path])
            let app = extracted.appendingPathComponent("Arsenal.app")
            try bundle(app, version: release.version)
            try identity(downloaded: app, installed: installed)
            return StagedArsenalUpdate(directory: directory, archive: archive, app: app, version: release.version, sha256: expected)
        } catch { try? FileManager.default.removeItem(at: directory); throw error }
    }

    /// Validate before extraction, including the relative worker and deduplicated licence links made by assembly.
    static func archiveEntries(_ archive: URL) throws {
        let names = try tool("/usr/bin/zipinfo", ["-1", archive.path]).split(whereSeparator: \.isNewline).map(String.init)
        guard !names.isEmpty, names.count <= 50000, Set(names).count == names.count,
            names.allSatisfy({ name in
                let parts = name.split(separator: "/", omittingEmptySubsequences: false)
                return name.hasPrefix("Arsenal.app/") && !name.contains("\\") && !name.contains(":") && !name.contains("\r")
                    && !parts.contains("..") && !parts.contains(".") && !parts.dropLast().contains("")
            })
        else { throw UpdateFailure("The release archive contains unsafe paths.") }
        let summary = try tool("/usr/bin/zipinfo", ["-t", archive.path])
        let fields = summary.split(whereSeparator: \.isWhitespace)
        guard fields.count >= 4, let expandedBytes = Int64(fields[2]), expandedBytes <= 512 * 1024 * 1024 else {
            throw UpdateFailure("The unpacked release archive is too large.")
        }
        let listing = try tool("/usr/bin/zipinfo", ["-l", archive.path])
        var links: [String: String] = [:]
        for line in listing.split(whereSeparator: \.isNewline) where line.hasPrefix("l") {
            let matching = names.filter { line.hasSuffix(" " + $0) }
            guard matching.count == 1, let name = matching.first else {
                throw UpdateFailure("The release archive contains an unexpected symbolic link.")
            }
            links[name] = try tool("/usr/bin/unzip", ["-p", archive.path, name])
        }
        let entries = Set(names)
        let helpers = "Arsenal.app/Contents/Helpers/", licences = "Arsenal.app/Contents/Resources/Licenses/"
        for (name, target) in links {
            guard let resolved = relativeLinkTarget(target, from: name), entries.contains(resolved), links[resolved] == nil,
                !names.contains(where: { $0.hasPrefix(name + "/") })
            else { throw UpdateFailure("The release archive contains an unsafe symbolic link.") }
            let worker =
                name.hasPrefix(helpers + "arsenal-worker-") && !name.dropFirst(helpers.count).contains("/")
                && target == "arsenal-worker-native" && resolved == helpers + "arsenal-worker-native"
            let licence = name.hasPrefix(licences) && resolved.hasPrefix(licences)
            guard worker || licence else { throw UpdateFailure("The release archive contains an unexpected symbolic link.") }
        }
    }

    /// Resolve lexically: no archive contents are extracted or followed during this check.
    private static func relativeLinkTarget(_ target: String, from name: String) -> String? {
        guard !target.isEmpty, !target.hasPrefix("/"), !target.contains("\\"), !target.contains(":"),
            !target.unicodeScalars.contains(where: { CharacterSet.controlCharacters.contains($0) })
        else { return nil }
        var parts = name.split(separator: "/").dropLast().map(String.init)
        for part in target.split(separator: "/", omittingEmptySubsequences: false) {
            switch part {
            case "", ".": continue
            case "..":
                guard parts.count > 1 else { return nil }
                parts.removeLast()
            default: parts.append(String(part))
            }
        }
        guard parts.first == "Arsenal.app", parts.count > 1 else { return nil }
        return parts.joined(separator: "/")
    }

    /// File-backed output avoids pipe deadlocks; all local tools have a bounded deadline.
    @discardableResult
    public static func tool(_ executable: String, _ arguments: [String], timeout: TimeInterval = 60) throws -> String {
        let output = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-update-tool-\(UUID().uuidString)")
        _ = FileManager.default.createFile(atPath: output.path, contents: nil, attributes: [.posixPermissions: 0o600])
        let handle = try FileHandle(forWritingTo: output)
        defer { try? handle.close(); try? FileManager.default.removeItem(at: output) }
        let process = Process(); process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        process.standardInput = FileHandle.nullDevice; process.standardOutput = handle; process.standardError = handle
        let exited = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in exited.signal() }
        try process.run()
        if exited.wait(timeout: .now() + timeout) == .timedOut {
            process.terminate(); kill(process.processIdentifier, SIGKILL)
            _ = exited.wait(timeout: .now() + 5)
            throw UpdateFailure("\(URL(fileURLWithPath: executable).lastPathComponent) timed out.")
        }
        guard process.terminationStatus == 0 else { throw UpdateFailure("\(URL(fileURLWithPath: executable).lastPathComponent) failed (\(process.terminationStatus)).") }
        let data = try Data(contentsOf: output)
        guard data.count <= 16 * 1024 * 1024 else { throw UpdateFailure("The archive listing is too large.") }
        return String(decoding: data, as: UTF8.self)
    }
}
