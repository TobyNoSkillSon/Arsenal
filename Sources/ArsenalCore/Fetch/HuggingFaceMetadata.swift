// SPDX-License-Identifier: MIT
import Foundation

public struct HFMetadataFile: Codable, Sendable, Equatable {
    public var path: String; public var bytes: Int64?; public var sha256: String?
}
public struct HFModelMetadata: Codable, Sendable, Equatable {
    public var repo: String; public var revision: String; public var pipelineTag: String?
    public var licence: String?; public var languages: [String]; public var parameterCount: Int64?
    public var downloads: Int64; public var likes: Int64; public var files: [HFMetadataFile]
    public static func decode(_ data: Data, repo: String) throws -> Self {
        guard repo.split(separator: "/").count == 2, SafePath.isRelative(repo), let object = try JSONSerialization.jsonObject(with: data) as? [String: Any],
            let revision = object["sha"] as? String, ModelManifest.isHex(revision, count: 40), let siblings = object["siblings"] as? [[String: Any]]
        else { throw ManifestError.invalid("HF metadata needs immutable revision and file inventory") }
        let card = object["cardData"] as? [String: Any] ?? [:]
        let languages = (card["language"] as? [String]) ?? (card["language"] as? String).map { [$0] } ?? []
        var seen = Set<String>()
        let files = try siblings.map { row -> HFMetadataFile in
            guard let path = row["rfilename"] as? String, SafePath.isRelative(path), seen.insert(path).inserted else {
                throw ManifestError.invalid("unsafe/duplicate HF metadata path")
            }
            let lfs = row["lfs"] as? [String: Any] ?? [:]
            let hash = lfs["sha256"] as? String
            let bytes = (row["size"] as? NSNumber)?.int64Value ?? (lfs["size"] as? NSNumber)?.int64Value
            guard hash.map({ ModelManifest.isHex($0, count: 64) }) ?? true, bytes.map({ $0 >= 0 }) ?? true else { throw ManifestError.invalid("invalid HF metadata hash/size") }
            return HFMetadataFile(path: path, bytes: bytes, sha256: hash)
        }.sorted { $0.path < $1.path }
        let params = ((object["safetensors"] as? [String: Any])?["total"] as? NSNumber)?.int64Value
        return Self(
            repo: repo, revision: revision, pipelineTag: object["pipeline_tag"] as? String, licence: card["license"] as? String, languages: languages, parameterCount: params,
            downloads: max(0, (object["downloads"] as? NSNumber)?.int64Value ?? 0), likes: max(0, (object["likes"] as? NSNumber)?.int64Value ?? 0), files: files)
    }
    public var inferredKind: String? {
        switch pipelineTag {
        case "text-generation", "conversational": "chat"
        case "text-to-image": "images"
        case "text-to-speech": "voices"
        case "text-to-audio": "sound"
        case "audio-to-audio": "enhancement"
        case "text-classification", "zero-shot-classification": "decisions"
        default: nil
        }
    }
}
extension HuggingFaceClient {
    /// Metadata only. Unlike pins(), this never requests any resolve/ file or executes repo code.
    public func metadata(repo: String, revision: String? = nil) async throws -> HFModelMetadata {
        guard repo.range(of: "^[A-Za-z0-9][A-Za-z0-9_.-]*/[A-Za-z0-9][A-Za-z0-9_.-]*$", options: .regularExpression) != nil,
            baseURL.scheme == "https" || ["localhost", "127.0.0.1"].contains(baseURL.host ?? ""), revision.map({ ModelManifest.isHex($0, count: 40) }) ?? true
        else { throw ManifestError.invalid("invalid HF metadata repo/revision/base") }
        var url = baseURL.appendingPathComponent("api/models")
        for segment in repo.split(separator: "/") { url.appendPathComponent(String(segment)) }
        if let revision { url.appendPathComponent("revision"); url.appendPathComponent(revision) }
        var components = URLComponents(url: url, resolvingAgainstBaseURL: false)!; components.queryItems = [URLQueryItem(name: "blobs", value: "true")]
        var request = URLRequest(url: components.url!); request.timeoutInterval = 30
        let (data, reply, sent, rejected) = try await self.data(request)
        try await throwIfGated(reply, repo: repo, repoType: "model", sentToken: sent, tokenRejected: rejected)
        guard (reply as? HTTPURLResponse)?.statusCode == 200, data.count <= 16 * 1024 * 1024 else { throw ManifestError.invalid("HF metadata request failed") }
        let metadata = try HFModelMetadata.decode(data, repo: repo)
        if let revision, metadata.revision != revision { throw ManifestError.invalid("HF metadata revision mismatch") }
        return metadata
    }
}
