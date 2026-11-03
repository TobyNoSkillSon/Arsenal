// SPDX-License-Identifier: MIT
import Foundation

public struct UpdateRelease: Codable, Sendable, Equatable {
    public let version: String
    public let notes: String
    public let size: Int64
    public let archive: URL
    public let checksums: URL
    public var zipName: String { "Arsenal-\(version)-arm64.zip" }

    public static func parse(_ data: Data) throws -> Self? {
        struct Asset: Decodable { let name: String; let size: Int64; let browser_download_url: URL }
        struct Release: Decodable {
            let tag_name: String; let body: String?; let draft: Bool; let prerelease: Bool; let assets: [Asset]
        }
        let release = try JSONDecoder().decode(Release.self, from: data)
        if release.draft || release.prerelease { return nil }
        guard let version = UpdateVersion(release.tag_name) else { throw UpdateFailure("The release has an invalid version.") }
        if version.isPrerelease { return nil }
        let zipName = "Arsenal-\(version)-arm64.zip"
        let archives = release.assets.filter { $0.name == zipName }, sums = release.assets.filter { $0.name == "SHA256SUMS" }
        guard archives.count == 1, sums.count == 1, let archive = archives.first, let checksums = sums.first,
            archive.size > 0, archive.size <= UpdateNetwork.maximumArchiveBytes
        else { throw UpdateFailure("The release is missing its app archive or SHA256SUMS.") }
        return Self(version: version.description, notes: release.body ?? "", size: archive.size, archive: archive.browser_download_url, checksums: checksums.browser_download_url)
    }
}

/// HTTPS-only in production. Explicit localhost allowance is a constructor seam for isolated tests, never an environment override.
public struct UpdateSource: Sendable {
    public let latest: URL
    public let permitsLocalhost: Bool
    public init(repository: String) throws {
        let parts = repository.split(separator: "/", omittingEmptySubsequences: false)
        guard parts.count == 2, parts.allSatisfy({ !$0.isEmpty && $0.allSatisfy { $0.isASCII && ($0.isLetter || $0.isNumber || $0 == "-" || $0 == "_" || $0 == ".") } }),
            let url = URL(string: "https://api.github.com/repos/\(repository)/releases/latest")
        else { throw UpdateFailure("ArsenalUpdateRepo must be an owner/repository name.") }
        latest = url; permitsLocalhost = false
    }
    public init(testServer: URL) throws {
        latest = testServer; permitsLocalhost = true
        guard testServer.scheme == "http", ["127.0.0.1", "localhost", "::1"].contains(testServer.host ?? ""), allows(testServer) else {
            throw UpdateFailure("The fake release server must be on localhost.")
        }
    }
    public func allows(_ url: URL) -> Bool {
        guard url.user == nil, url.password == nil else { return false }
        if permitsLocalhost { return url.scheme == "http" && ["127.0.0.1", "localhost", "::1"].contains(url.host ?? "") }
        return url.scheme == "https"
            && ["api.github.com", "github.com", "release-assets.githubusercontent.com", "objects.githubusercontent.com"].contains(url.host?.lowercased() ?? "")
    }
}

/// No cached credentials or cookies. Downloads stream to disk and report URLSession's byte progress.
public final class UpdateNetwork: NSObject, URLSessionDownloadDelegate, @unchecked Sendable {
    public static let maximumArchiveBytes: Int64 = 256 * 1024 * 1024
    private let source: UpdateSource
    private var session: URLSession!
    private let lock = NSLock()
    private struct Download {
        let destination: URL
        let maximum: Int64
        let progress: @Sendable (Double) -> Void
        let continuation: CheckedContinuation<Void, any Error>
        var percent: Int = -1
    }
    private var downloads: [Int: Download] = [:]

    public init(source: UpdateSource) {
        self.source = source
        super.init()
        let config = URLSessionConfiguration.ephemeral
        config.urlCache = nil; config.httpCookieStorage = nil; config.urlCredentialStorage = nil; config.httpShouldSetCookies = false
        config.timeoutIntervalForRequest = 60; config.timeoutIntervalForResource = 1800
        config.httpAdditionalHeaders = ["User-Agent": "Arsenal-Updater"]
        session = URLSession(configuration: config, delegate: self, delegateQueue: nil)
    }
    public func close() { session.invalidateAndCancel() }

    public func latest() async throws -> UpdateRelease? {
        guard source.allows(source.latest) else { throw UpdateFailure("Release URLs must use HTTPS.") }
        var request = URLRequest(url: source.latest)
        request.setValue("application/vnd.github+json", forHTTPHeaderField: "Accept")
        request.setValue("2022-11-28", forHTTPHeaderField: "X-GitHub-Api-Version")
        let (bytes, response) = try await session.bytes(for: request)
        defer { bytes.task.cancel() }
        guard let response = response as? HTTPURLResponse else { throw UpdateFailure("The release server did not answer.") }
        if response.statusCode == 404 { return nil }
        guard response.statusCode == 200 else { throw UpdateFailure("The release check answered HTTP \(response.statusCode).") }
        var data = Data()
        for try await byte in bytes {
            guard data.count < 1024 * 1024 else { throw UpdateFailure("The release response is too large.") }; data.append(byte)
        }
        return try UpdateRelease.parse(data)
    }

    public func download(_ url: URL, to destination: URL, maximum: Int64, progress: @escaping @Sendable (Double) -> Void = { _ in }) async throws {
        guard source.allows(url) else { throw UpdateFailure("Release downloads must use HTTPS.") }
        try await withCheckedThrowingContinuation { continuation in
            let task = session.downloadTask(with: url)
            lock.withLock { downloads[task.taskIdentifier] = Download(destination: destination, maximum: maximum, progress: progress, continuation: continuation) }
            task.resume()
        }
    }

    public func urlSession(
        _ session: URLSession, downloadTask: URLSessionDownloadTask, didWriteData bytesWritten: Int64, totalBytesWritten: Int64, totalBytesExpectedToWrite: Int64
    ) {
        guard let item = lock.withLock({ downloads[downloadTask.taskIdentifier] }) else { return }
        if totalBytesWritten > item.maximum || totalBytesExpectedToWrite > item.maximum {
            let removed = lock.withLock { downloads.removeValue(forKey: downloadTask.taskIdentifier) }
            downloadTask.cancel(); removed?.continuation.resume(throwing: UpdateFailure("The downloaded asset is too large.")); return
        }
        if totalBytesExpectedToWrite > 0 {
            let progress = min(1, Double(totalBytesWritten) / Double(totalBytesExpectedToWrite))
            let percent = Int(progress * 100)
            let changed = lock.withLock { () -> Bool in
                guard var current = downloads[downloadTask.taskIdentifier], percent > current.percent else { return false }
                current.percent = percent; downloads[downloadTask.taskIdentifier] = current; return true
            }
            if changed { item.progress(progress) }
        }
    }
    public func urlSession(_ session: URLSession, downloadTask: URLSessionDownloadTask, didFinishDownloadingTo location: URL) {
        guard let item = lock.withLock({ downloads.removeValue(forKey: downloadTask.taskIdentifier) }) else { return }
        do {
            guard (downloadTask.response as? HTTPURLResponse)?.statusCode == 200 else { throw UpdateFailure("The download server did not return a release asset.") }
            let size = try location.resourceValues(forKeys: [.fileSizeKey]).fileSize ?? 0
            guard size <= item.maximum else { throw UpdateFailure("The downloaded asset is too large.") }
            try FileManager.default.moveItem(at: location, to: item.destination)
            item.progress(1); item.continuation.resume()
        } catch { item.continuation.resume(throwing: error) }
    }
    public func urlSession(_ session: URLSession, task: URLSessionTask, didCompleteWithError error: (any Error)?) {
        guard let error, let item = lock.withLock({ downloads.removeValue(forKey: task.taskIdentifier) }) else { return }
        item.continuation.resume(throwing: error)
    }
    public func urlSession(
        _ session: URLSession, task: URLSessionTask, willPerformHTTPRedirection response: HTTPURLResponse, newRequest request: URLRequest,
        completionHandler: @escaping (URLRequest?) -> Void
    ) {
        completionHandler(request.url.map(source.allows) == true ? request : nil)
    }
}
