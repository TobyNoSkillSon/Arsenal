// SPDX-License-Identifier: MIT
import Foundation

/// Every cell in a precision comparison must belong to the same revision and run.
/// Methods differ by metric, but chip/date/source/baseline/run identity may not.
public enum PrecisionEvidence {
    public static func matches(_ lhs: StatProvenance?, _ rhs: StatProvenance?) -> Bool {
        guard let lhs, let rhs, let revision = lhs.revision, !revision.isEmpty, let run = lhs.run, !run.isEmpty else { return false }
        return lhs.revision == rhs.revision && lhs.run == rhs.run && lhs.chip == rhs.chip && lhs.gpuCores == rhs.gpuCores && lhs.date == rhs.date
            && lhs.source == rhs.source && lhs.baseline == rhs.baseline && lhs.defaultWorkload == rhs.defaultWorkload
    }
    public static func current(_ stats: PrecisionStatSet, revision: String) -> Bool {
        let identity = stats.availability.provenance
        guard identity?.revision == revision, matches(identity, identity) else { return false }
        return stats.metrics.values.filter { [.measured, .quickCheck, .derived].contains($0.state) }.allSatisfy { matches($0.provenance, identity) }
    }
    /// Recipe + model runner declaration + its implementation closure, independent of unrelated worker families.
    public static func qualityIdentity(model: ModelManifest, dtype: String, runner: String, implementation: String?, loadIdentity: String? = "") -> String? {
        let encoder = JSONEncoder(); encoder.outputFormatting = .sortedKeys
        guard let implementation, !implementation.isEmpty, let loadIdentity,
            let spec = model.runners.first(where: { $0.name == runner }), let bytes = try? encoder.encode(spec)
        else { return nil }
        var data = bytes
        data.append(Data(("\n" + (model.derive?[dtype] ?? "vendor:" + dtype) + "\n" + implementation).utf8))
        if !loadIdentity.isEmpty { data.append(Data(("\nload:" + loadIdentity).utf8)) }
        return FileDigest.sha256(data)
    }
    public static func scopeQuality(_ result: PrecisionStatsResult, identity: String?) -> PrecisionStatsResult {
        var result = result
        // Translate known quality-only legacy states before clearing a stale quality conclusion.
        result.stats = result.stats.applyingInt4Policy
        if identity == nil || result.stats.qualityIdentity != identity {
            // Stale evidence says nothing either way: a pass and a failure bound to another recipe/implementation are both
            // unchecked. int8 stays offered; int4 shows the not-yet-checked warning. Runtime failures are kept. Unbound
            // legacy outcomes carry no identity; their currency is ImplementationEvidence's (checks/fingerprints) call.
            let bound = result.stats.qualityIdentity != nil
            result.qualityIdentityMismatch = true
            result.stats.qualityCheckPassed = nil
            result.qualityEvidence = nil
            if bound, result.stats.hasLegacyQualityFailure {
                let provenance = result.stats.availability.provenance
                result.stats.availability = StatValue(state: provenance == nil ? .notMeasured : .quickCheck, provenance: provenance)
                result.stats.qualityOnlyUnavailable = nil
            }
        }
        return result
    }

}
