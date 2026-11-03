// SPDX-License-Identifier: MIT
import Foundation

public struct DerivedFile: Codable, Sendable, Equatable, Hashable {
    public var path: String; public var bytes: Int64; public var sha256: String
    public init(path: String, bytes: Int64, sha256: String) { self.path = path; self.bytes = bytes; self.sha256 = sha256 }
}
public protocol PrecisionConverter: Sendable {
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, destinationDirectory: URL, precision: String, recipe: String) async throws -> [DerivedFile]
    func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, assets: [String: URL], destinationDirectory: URL, precision: String, recipe: String) async throws
        -> [DerivedFile]
}
extension PrecisionConverter {
    public func convert(model: ModelManifest, modelDirectory: URL, sourceDirectory: URL, assets: [String: URL], destinationDirectory: URL, precision: String, recipe: String)
        async throws -> [DerivedFile]
    {
        try await convert(
            model: model, modelDirectory: modelDirectory, sourceDirectory: sourceDirectory, destinationDirectory: destinationDirectory, precision: precision, recipe: recipe)
    }
}
public struct DerivedManifest: Codable, Sendable, Equatable {
    public var schema = 1
    public var model: String
    public var precision: String
    public var sourceRevision: String
    public var sourceDtype: String
    public var sourceHashes: [String: String]
    /// Recovery identity, not a live reference: source blobs may be collected after a verified derive.
    public var sourceFiles: [DerivedFile]?
    public var sourceRepository: String?
    public var normalization: String?
    public var assetSources: [String: WeightSpec]?
    public var recipe: String
    public var groupSize = 64
    public var files: [DerivedFile]
}
