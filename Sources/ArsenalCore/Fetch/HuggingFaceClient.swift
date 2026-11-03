// SPDX-License-Identifier: MIT
import Foundation
import Security
import CryptoKit

public struct DownloadProgress: Codable, Sendable, Equatable {
    public var file: String; public var completedBytes: Int64; public var totalBytes: Int64; public var bytesPerSecond: Double
    public var contentID: String?
    public var phase: String?
    /// Bytes received in this HTTP attempt, excluding an existing .part prefix. Nil for local verification/cache hits.
    public var networkBytes: Int64?
    /// A rejected local donor was inspected, not admitted; reset only that content counter.
    public var resetsProgress: Bool?
    public init(
        file: String, completedBytes: Int64, totalBytes: Int64, bytesPerSecond: Double, contentID: String? = nil, phase: String? = nil, networkBytes: Int64? = nil,
        resetsProgress: Bool? = nil
    ) {
        self.file = file; self.completedBytes = completedBytes; self.totalBytes = totalBytes; self.bytesPerSecond = bytesPerSecond; self.contentID = contentID; self.phase = phase
        self.networkBytes = networkBytes; self.resetsProgress = resetsProgress
    }
}
public struct DownloadJournal: Codable, Sendable {
    public var repo: String; public var revision: String; public var path: String; public var sha256: String; public var bytes: Int64
    public var received: Int64; public var etag: String?
}
public enum HFTokenStore {
    /// The read token stored as `dev.arsenal.huggingface`.
    public static func read() -> String? { read(service: "dev.arsenal.huggingface") }
    static func read(service: String) -> String? {
        let query: [String: Any] = [
            kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: service, kSecAttrAccount as String: "read-token",
            kSecReturnData as String: true, kSecMatchLimit as String: kSecMatchLimitOne
        ]
        var result: CFTypeRef?
        guard silently({ SecItemCopyMatching(query as CFDictionary, &result) }) == errSecSuccess, let data = result as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }
    /// Runs a Keychain call that never shows the Keychain's password dialog. An item another tool saved (the `security`
    /// command, whose partition list Arsenal is not on) answers errSecAuthFailed instead, so the token falls back to
    /// HF_TOKEN or the CLI's token file rather than putting a password prompt on the screen. Only the background read
    /// uses it: a save or remove the user starts in Settings may still ask, so such an item can be replaced there.
    static func silently(_ call: () -> OSStatus) -> OSStatus {
        interactionLock.lock(); defer { interactionLock.unlock() }
        var allowed: DarwinBoolean = true
        SecKeychainGetUserInteractionAllowed(&allowed)
        SecKeychainSetUserInteractionAllowed(false)
        defer { SecKeychainSetUserInteractionAllowed(allowed.boolValue) }
        return call()
    }
    private static let interactionLock = NSLock()
}
public struct HuggingFaceClient: Sendable {
    public let baseURL: URL
    public var maximumAttempts = 4
    public var stallTimeout: TimeInterval = 60
    /// Compatibility alias: now an inactivity deadline, never a total file budget.
    public var deadline: TimeInterval {
        get { stallTimeout }
        set { stallTimeout = newValue }
    }
    /// The read token: Arsenal's Keychain item, else HF_TOKEN, else the Hugging Face CLI's token file (`HFTokenSource`).
    public var token: @Sendable () -> String? = { HFTokenSource.resolve() }
    /// Tests: send the token to a loopback stand-in too (it goes only to huggingface.co otherwise).
    var sendsTokenToLoopback = false
    /// Tests: a write to the partial fails with the returned error (a full disk, simulated without filling one). Given the bytes
    /// on disk after that write. Nil in the app.
    public var writeFault: (@Sendable (Int64) -> Error?)?
    public init(baseURL: URL = URL(string: "https://huggingface.co")!) { self.baseURL = baseURL }
    public func resolve(repo: String, revision: String, path: String, repoType: String = "model") throws -> URL {
        guard repo.split(separator: "/").count == 2, SafePath.isRelative(repo), ModelManifest.isHex(revision, count: 40), SafePath.isRelative(path),
            baseURL.scheme == "https" || ["127.0.0.1", "localhost"].contains(baseURL.host ?? "")
        else { throw ManifestError.invalid("invalid pinned Hugging Face URL") }
        var url = baseURL
        if repoType == "dataset" { url.appendPathComponent("datasets") }
        for segment in (repo + "/resolve/" + revision + "/" + path).split(separator: "/") { url.appendPathComponent(String(segment)) }; return url
    }
    public func download(
        repo: String, revision: String, file: DerivedFile, directory: URL, repoType: String = "model", progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> URL {
        try await downloadVerified(repo: repo, revision: revision, file: file, directory: directory, repoType: repoType, progress: progress).url
    }
    public func downloadVerified(
        repo: String, revision: String, file: DerivedFile, directory: URL, repoType: String = "model", progress: @escaping @Sendable (DownloadProgress) -> Void = { _ in }
    ) async throws -> VerifiedDownload {
        let url = try resolve(repo: repo, revision: revision, path: file.path, repoType: repoType)
        guard ModelManifest.isHex(file.sha256, count: 64), file.bytes >= 0, maximumAttempts > 0, stallTimeout > 0 else {
            throw ManifestError.invalid("invalid download pin or retry/stall policy")
        }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let part = try SafePath.resolve(file.sha256 + ".part", inside: directory); let journalURL = try SafePath.resolve(file.sha256 + ".json", inside: directory)
        var journal = DownloadJournal(repo: repo, revision: revision, path: file.path, sha256: file.sha256, bytes: file.bytes, received: 0, etag: nil)
        if FileManager.default.fileExists(atPath: journalURL.path) {
            let old = try JSONDecoder().decode(DownloadJournal.self, from: Data(contentsOf: journalURL))
            if old.repo == repo && old.revision == revision && old.path == file.path && old.sha256 == file.sha256 && old.bytes == file.bytes {
                journal = old
            } else if FileManager.default.fileExists(atPath: part.path) {
                try FileManager.default.moveItem(at: part, to: part.appendingPathExtension("old-" + UUID().uuidString))
            }
        } else if FileManager.default.fileExists(atPath: part.path) {
            try FileManager.default.moveItem(at: part, to: part.appendingPathExtension("old-" + UUID().uuidString))
        }
        let offset = (try? FileDigest.bytes(part)) ?? 0
        guard offset <= file.bytes else {
            try quarantine(part: part, journal: journalURL)
            throw ManifestError.invalid("partial download exceeded pinned size and was moved aside; retry Get")
        }
        // A new invocation hashes its existing prefix once. Retries retain the incremental state, rather than re-reading it.
        var state = HFHashState()
        if offset > 0 {
            let handle = try FileHandle(forReadingFrom: part); defer { try? handle.close() }
            let reporter = LocalVerificationProgress(file: file, sink: progress)
            while try autoreleasepool(invoking: { () throws -> Bool in
                try Task<Never, Never>.checkCancellation()
                guard let data = try handle.read(upToCount: 4 * 1024 * 1024), !data.isEmpty else { return false }
                state.add(data); reporter.report(state.bytes); return true
            }) {}
        }
        var failures = 0
        // A token Hugging Face rejects (expired, revoked: a bare 401) is dropped once, so a public file still downloads.
        var withoutToken = false
        while true {
            try Task.checkCancellation()
            let current = (try? FileDigest.bytes(part)) ?? 0
            guard state.bytes == current else { throw ManifestError.invalid("partial download changed during Get") }
            if current == file.bytes, FileManager.default.fileExists(atPath: part.path) {
                guard state.digest == file.sha256.lowercased() else {
                    try quarantine(part: part, journal: journalURL)
                    throw ManifestError.invalid("download checksum mismatch: \(file.path); corrupt part moved aside, retry Get")
                }
                try? FileManager.default.removeItem(at: journalURL)
                return try VerifiedDownload(url: part, sha256: file.sha256, bytes: file.bytes)
            }
            journal.received = current
            try JSONEncoder().encode(journal).write(to: journalURL, options: .atomic)
            var request = URLRequest(url: url); request.timeoutInterval = stallTimeout; request.setValue("identity", forHTTPHeaderField: "Accept-Encoding")
            if current > 0 {
                request.setValue("bytes=\(current)-", forHTTPHeaderField: "Range"); if let etag = journal.etag { request.setValue(etag, forHTTPHeaderField: "If-Range") }
            }
            let token = withoutToken ? nil : requestToken()
            if let token { request.setValue("Bearer \(token)", forHTTPHeaderField: "Authorization") }
            let transfer = HFTransfer(
                part: part, journalURL: journalURL, journal: journal, progress: progress, stallTimeout: stallTimeout, hashState: state, writeFault: writeFault,
                sentToken: token != nil)
            do {
                try await transfer.run(request); state = transfer.hashState
                guard state.bytes == file.bytes, state.digest == file.sha256.lowercased() else {
                    try quarantine(part: part, journal: journalURL)
                    throw ManifestError.invalid("download checksum mismatch: \(file.path); corrupt part moved aside, retry Get")
                }
                try? FileManager.default.removeItem(at: journalURL)
                return try VerifiedDownload(url: part, sha256: file.sha256, bytes: file.bytes)
            } catch {
                state = transfer.hashState
                if error is CancellationError || Task.isCancelled { throw CancellationError() }
                // A gated repository: the partial stays for the Get after the owner's terms are accepted.
                if let gate = error as? HFGateRefusal {
                    throw await gateError(repo: repo, repoType: repoType, sentToken: gate.sentToken, tokenRejected: withoutToken)
                }
                // A full disk leaves nothing to resume: the partial and its journal go (after the transfer's last journal write),
                // so the next Get starts clean and no blob is published from this download.
                if BlobStore.isDiskFull(error) {
                    try? FileManager.default.removeItem(at: part); try? FileManager.default.removeItem(at: journalURL)
                    throw BlobStore.diskFull(error)
                }
                if let e = error as? HFDownloadError, e.status == 401, token != nil, !withoutToken { withoutToken = true; continue }
                // The pinned revision is immutable: a 404 for its file never turns into bytes on a later Get.
                if let e = error as? HFDownloadError, e.status == 404 { throw HFMissingFileError(repo: repo, revision: revision, path: file.path) }
                if let e = error as? HFDownloadError, !e.retryable { throw e }
                if error is ManifestError { throw error }
                failures = state.bytes > current ? 0 : failures + 1
                if failures >= maximumAttempts { throw error }
                try await Task.sleep(for: .milliseconds(250 * (1 << min(4, failures))))
                if let updated = try? JSONDecoder().decode(DownloadJournal.self, from: Data(contentsOf: journalURL)) { journal = updated }
            }
        }
    }
    /// Preserve corrupt bytes and their journal for diagnosis, but remove both from the active resume paths.
    private func quarantine(part: URL, journal: URL) throws {
        let suffix = "bad-" + UUID().uuidString
        // Reset the journal even if the later data move fails; an unjournaled part is isolated on the next Get.
        if FileManager.default.fileExists(atPath: journal.path) { try FileManager.default.moveItem(at: journal, to: journal.appendingPathExtension(suffix)) }
        if FileManager.default.fileExists(atPath: part.path) { try FileManager.default.moveItem(at: part, to: part.appendingPathExtension(suffix)) }
    }
    /// Metadata hashes for LFS blobs avoid reading GB of data just to pin a catalog; small Git blobs are hashed locally.
    public func pins(_ weights: WeightSpec) async throws -> [DerivedFile] {
        guard ModelManifest.isHex(weights.revision, count: 40), SafePath.isRelative(weights.repo) else { throw ManifestError.invalid("pin requires immutable revision") }
        var endpoint = baseURL.appendingPathComponent(weights.repoType == "dataset" ? "api/datasets" : "api/models")
        for segment in weights.repo.split(separator: "/") { endpoint.appendPathComponent(String(segment)) }
        endpoint.appendPathComponent("revision"); endpoint.appendPathComponent(weights.revision)
        var components = URLComponents(url: endpoint, resolvingAgainstBaseURL: false)!; components.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        var request = URLRequest(url: components.url!); request.timeoutInterval = 30
        let (data, response, sent, rejected) = try await self.data(request)
        let repoType = weights.repoType ?? "model"
        try await throwIfGated(response, repo: weights.repo, repoType: repoType, sentToken: sent, tokenRejected: rejected)
        guard (response as? HTTPURLResponse)?.statusCode == 200, let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            object["sha"] as? String == weights.revision, let siblings = object["siblings"] as? [[String: Any]]
        else { throw ManifestError.invalid("Hugging Face did not return pinned metadata") }
        return try await weights.files.asyncMap { path in
            guard let meta = siblings.first(where: { $0["rfilename"] as? String == path }), let size = (meta["size"] as? NSNumber)?.int64Value, size >= 0 else {
                throw ManifestError.invalid("missing metadata: \(path)")
            }
            if let lfs = meta["lfs"] as? [String: Any], let hash = lfs["sha256"] as? String, ModelManifest.isHex(hash, count: 64) {
                return DerivedFile(path: path, bytes: size, sha256: hash)
            }
            guard size <= 16 * 1024 * 1024 else { throw ManifestError.invalid("unhashed large Git file: \(path)") }
            var smallRequest = URLRequest(url: try resolve(repo: weights.repo, revision: weights.revision, path: path, repoType: weights.repoType ?? "model"))
            smallRequest.timeoutInterval = 60
            let (blob, reply, sentSmall, rejectedSmall) = try await self.data(smallRequest)
            try await throwIfGated(reply, repo: weights.repo, repoType: repoType, sentToken: sentSmall, tokenRejected: rejectedSmall)
            guard (reply as? HTTPURLResponse)?.statusCode == 200, blob.count == size else { throw ManifestError.invalid("pin download failed: \(path)") }
            return DerivedFile(path: path, bytes: size, sha256: FileDigest.sha256(blob))
        }
    }
}
private extension Array where Element: Sendable {
    func asyncMap<T: Sendable>(_ transform: (Element) async throws -> T) async rethrows -> [T] {
        var result: [T] = []; for x in self { result.append(try await transform(x)) }; return result
    }
}
private struct HFDownloadError: Error, LocalizedError {
    let message: String; let retryable: Bool
    var status: Int?
    var errorDescription: String? { message }
}
/// A transfer's gate refusal, turned into `HFGateError` (with the repository's gate mode) by the client.
private struct HFGateRefusal: Error { let sentToken: Bool }

/// Streaming chunked URLSession delegate. Inspired by Vella Catalog/Downloads/NativeModelDownload.swift
/// reimplemented for SHA256 catalog pins rather than Hub ETags. Vella source is read-only and not changed.
private struct HFHashState: @unchecked Sendable {
    var hash = SHA256(); var bytes: Int64 = 0
    mutating func add(_ data: Data) { hash.update(data: data); bytes += Int64(data.count) }
    var digest: String { hash.finalize().map { String(format: "%02x", $0) }.joined() }
}
private final class HFTransfer: NSObject, URLSessionDataDelegate, @unchecked Sendable {
    private let part: URL; private let journalURL: URL; private var journal: DownloadJournal
    private let progress: @Sendable (DownloadProgress) -> Void; private let stallTimeout: TimeInterval
    private let writeFault: (@Sendable (Int64) -> Error?)?
    private let sentToken: Bool
    private(set) var hashState: HFHashState
    private var stallTimer: DispatchSourceTimer?
    private let lock = NSLock(); private var task: URLSessionDataTask?; private var cancelled = false
    private var session: URLSession?; private var continuation: CheckedContinuation<Void, Error>?
    private var handle: FileHandle?; private var failure: Error?; private var accepted = false
    private var started = Date(); private var startBytes: Int64 = 0; private var lastReport = Date.distantPast
    init(
        part: URL, journalURL: URL, journal: DownloadJournal, progress: @escaping @Sendable (DownloadProgress) -> Void, stallTimeout: TimeInterval,
        hashState: HFHashState, writeFault: (@Sendable (Int64) -> Error?)? = nil, sentToken: Bool = false
    ) {
        self.sentToken = sentToken
        self.part = part; self.journalURL = journalURL; self.journal = journal; self.progress = progress; self.stallTimeout = stallTimeout; self.hashState = hashState
        self.writeFault = writeFault
    }
    func run(_ request: URLRequest) async throws {
        try await withTaskCancellationHandler {
            try await withCheckedThrowingContinuation { c in
                continuation = c
                if !FileManager.default.fileExists(atPath: part.path) { FileManager.default.createFile(atPath: part.path, contents: nil) }
                do { handle = try FileHandle(forWritingTo: part); try handle?.seekToEnd() } catch { c.resume(throwing: error); continuation = nil; return }
                started = Date(); startBytes = journal.received
                let config = URLSessionConfiguration.ephemeral; config.timeoutIntervalForResource = .greatestFiniteMagnitude
                config.timeoutIntervalForRequest = request.timeoutInterval
                let queue = DispatchQueue(label: "dev.arsenal.hf-transfer.\(UUID().uuidString)")
                let q = OperationQueue(); q.maxConcurrentOperationCount = 1; q.underlyingQueue = queue
                let timer = DispatchSource.makeTimerSource(queue: queue)
                timer.setEventHandler { [weak self] in
                    guard let self else { return }
                    self.failure = HFDownloadError(message: "download stalled; retry Get to resume", retryable: true); self.task?.cancel()
                }
                stallTimer = timer; timer.schedule(deadline: .now() + stallTimeout); timer.resume()
                let s = URLSession(configuration: config, delegate: self, delegateQueue: q); session = s
                let t = s.dataTask(with: request)
                let wasCancelled = lock.withLock {
                    task = t; return cancelled
                }
                if wasCancelled { t.cancel() }; t.resume()
            }
        } onCancel: {
            self.lock.withLock {
                self.cancelled = true; self.task?.cancel()
            }
        }
    }
    func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        var clean = request
        if request.url?.host != "huggingface.co" { clean.setValue(nil, forHTTPHeaderField: "Authorization") }
        completionHandler(clean)
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive response: URLResponse, completionHandler: @escaping (URLSession.ResponseDisposition) -> Void) {
        do {
            guard let http = response as? HTTPURLResponse else { throw HFDownloadError(message: "invalid download response", retryable: false) }
            switch http.statusCode {
            case 200:
                try handle?.truncate(atOffset: 0); try handle?.seek(toOffset: 0); journal.received = 0; startBytes = 0; hashState = HFHashState()
                if http.expectedContentLength >= 0 && http.expectedContentLength != journal.bytes { throw HFDownloadError(message: "download length mismatch", retryable: false) }
            case 206:
                let value = http.value(forHTTPHeaderField: "Content-Range") ?? ""
                let pattern = "^bytes ([0-9]+)-([0-9]+)/([0-9]+)$"
                guard let regex = try? NSRegularExpression(pattern: pattern), let m = regex.firstMatch(in: value, range: NSRange(value.startIndex..., in: value)),
                    m.numberOfRanges == 4
                else { throw HFDownloadError(message: "invalid Content-Range", retryable: false) }
                let numbers = (1...3).compactMap { Range(m.range(at: $0), in: value).flatMap { Int64(value[$0]) } }
                guard numbers.count == 3, numbers[0] == journal.received, numbers[1] == journal.bytes - 1, numbers[2] == journal.bytes,
                    http.expectedContentLength < 0 || http.expectedContentLength == journal.bytes - journal.received
                else { throw HFDownloadError(message: "download Range reply rejected", retryable: false) }
            case 401, 403:
                if HFGateError.isGate(status: http.statusCode, errorCode: http.value(forHTTPHeaderField: "X-Error-Code"), sentToken: sentToken) {
                    throw HFGateRefusal(sentToken: sentToken)
                }
                throw HFDownloadError(message: "Hugging Face refused the download (HTTP \(http.statusCode))", retryable: false, status: http.statusCode)
            case 429, 500...599: throw HFDownloadError(message: "Hugging Face temporarily unavailable", retryable: true)
            default: throw HFDownloadError(message: "download HTTP \(http.statusCode)", retryable: false, status: http.statusCode)
            }
            if http.statusCode == 206, let old = journal.etag, let new = http.value(forHTTPHeaderField: "ETag"), old != new {
                throw HFDownloadError(message: "download validator changed", retryable: false)
            }
            journal.etag = http.value(forHTTPHeaderField: "ETag"); accepted = true
            stallTimer?.schedule(deadline: .now() + stallTimeout)
            progress(
                DownloadProgress(
                    file: journal.path, completedBytes: journal.received, totalBytes: journal.bytes, bytesPerSecond: 0, contentID: journal.sha256.lowercased(),
                    phase: "downloading", networkBytes: 0))
            try JSONEncoder().encode(journal).write(to: journalURL, options: .atomic); completionHandler(.allow)
        } catch { failure = error; completionHandler(.cancel) }
    }
    func urlSession(_ session: URLSession, dataTask: URLSessionDataTask, didReceive data: Data) {
        guard accepted, failure == nil else { return }
        do {
            guard journal.received + Int64(data.count) <= journal.bytes else { throw HFDownloadError(message: "download exceeds pinned size", retryable: false) }
            // The seam sees the bytes on disk after this write, so its outcome does not depend on how the network chunks the body.
            if let fault = writeFault?(journal.received + Int64(data.count)) { throw fault }
            try handle?.write(contentsOf: data); journal.received += Int64(data.count); hashState.add(data)
            stallTimer?.schedule(deadline: .now() + stallTimeout)
            if Date().timeIntervalSince(lastReport) >= 0.25 {
                lastReport = Date(); try JSONEncoder().encode(journal).write(to: journalURL, options: .atomic)
                progress(
                    DownloadProgress(
                        file: journal.path, completedBytes: journal.received, totalBytes: journal.bytes,
                        bytesPerSecond: Double(journal.received - startBytes) / max(0.001, Date().timeIntervalSince(started)), contentID: journal.sha256.lowercased(),
                        phase: "downloading", networkBytes: journal.received - startBytes))
            }
        } catch {
            failure = BlobStore.diskFull(error)
            dataTask.cancel()
        }
    }
    func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: Error?) {
        stallTimer?.setEventHandler(handler: nil); stallTimer?.cancel(); stallTimer = nil
        try? handle?.synchronize(); try? handle?.close(); handle = nil
        try? JSONEncoder().encode(journal).write(to: journalURL, options: .atomic)
        let c = continuation; continuation = nil
        if let failure {
            c?.resume(throwing: failure)
        } else if lock.withLock({ cancelled }) {
            c?.resume(throwing: CancellationError())
        } else if let error {
            c?.resume(throwing: error)
        } else if journal.received != journal.bytes || !accepted {
            c?.resume(throwing: HFDownloadError(message: "interrupted download", retryable: true))
        } else {
            progress(
                DownloadProgress(
                    file: journal.path, completedBytes: journal.received, totalBytes: journal.bytes,
                    bytesPerSecond: Double(journal.received - startBytes) / max(0.001, Date().timeIntervalSince(started)), contentID: journal.sha256.lowercased(),
                    phase: "downloading", networkBytes: journal.received - startBytes))
            c?.resume()
        }
        session.finishTasksAndInvalidate(); self.session = nil
    }
}
