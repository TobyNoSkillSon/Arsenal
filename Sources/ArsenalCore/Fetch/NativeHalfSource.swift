// SPDX-License-Identifier: MIT
import Foundation

extension ModelFetcher {
    /// A model-owned layout transform receives a complete half source, never a retained BF16/FP32 source copy.
    /// Each original shard is verified, normalized and removed before fetching the next shard.
    func normalizedSource(
        model: ModelManifest, modelDirectory: URL, converter: any PrecisionConverter, destination: URL,
        progress: @escaping @Sendable (DownloadProgress) -> Void, deriving: @escaping @Sendable () -> Void
    ) async throws {
        for file in try pinnedFiles(model.weights) {
            try Task.checkCancellation()
            let root = support.appendingPathComponent("weights/.cast-" + UUID().uuidString)
            let input = root.appendingPathComponent("input"), output = root.appendingPathComponent("output")
            try ConversionStaging.preflight(in: support.appendingPathComponent("weights"), sourceBytes: file.bytes)
            let staging = try ConversionStaging(root)
            defer { withExtendedLifetime(staging) {} }
            await store.reserve(files: [file])
            do {
                try await ensureBlob(file, source: model.weights, progress: progress)
                try await store.linkVerifiedBlob(file, to: SafePath.resolve(file.path, inside: input))
                try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
                deriving()
                let converted = try await converter.convert(
                    model: model, modelDirectory: modelDirectory, sourceDirectory: input,
                    destinationDirectory: output, precision: model.basePrecision, recipe: NativeHalfPolicy.recipe)
                guard converted.count == 1, converted[0].path == file.path, Set(try SafePath.files(in: output)) == [file.path] else {
                    throw ManifestError.invalid("Half normalization must preserve the source file path")
                }
                let normalized = try SafePath.resolve(file.path, inside: output), target = try SafePath.resolve(file.path, inside: destination)
                try FileDigest.verify(normalized, sha256: converted[0].sha256, bytes: converted[0].bytes)
                try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
                try FileManager.default.moveItem(at: normalized, to: target)
                try FileManager.default.removeItem(at: root)
                await store.unreserve(files: [file]); try await store.collectUnreferenced(files: [file])
            } catch {
                try? FileManager.default.removeItem(at: root)
                await store.unreserve(files: [file]); try? await store.collectUnreferenced(files: [file]); throw error
            }
        }
    }
}
