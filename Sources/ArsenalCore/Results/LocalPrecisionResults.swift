// SPDX-License-Identifier: MIT
import Foundation

/// Local measurements are mutable data beside signed folder evidence. They never supply code,
/// kernel recipes, legacy badges or runner ranking. All other installed files stay immutable.
public enum LocalPrecisionResults {
    public static let prefix = "results/local/"
    public static func validate(_ data: Data, model: ModelManifest) throws -> PrecisionStatsResult {
        guard data.count <= 1_048_576 else { throw ManifestError.invalid("Local stat result exceeds 1 MiB") }
        let result = try PrecisionStatsResult.decode(data)
        guard result.model == model.id, result.stats.kind.rawValue == model.kind else { throw ManifestError.invalid("Local stat result model/kind differs") }
        return result
    }
    public static func read(folder: URL, model: ModelManifest) -> [PrecisionStatsResult] {
        guard let directory = try? SafePath.resolve("results/local", inside: folder),
            let paths = try? SafePath.files(in: directory)
        else { return [] }
        return paths.sorted().compactMap { path in
            guard path.hasSuffix(".md"), let file = try? SafePath.resolve(path, inside: directory),
                let bytes = try? FileDigest.bytes(file), bytes <= 1_048_576, let data = try? Data(contentsOf: file),
                let result = try? validate(data, model: model),
                PrecisionEvidence.current(result.stats, revision: model.weights.revision)
            else { return nil }
            return result
        }
    }
}
