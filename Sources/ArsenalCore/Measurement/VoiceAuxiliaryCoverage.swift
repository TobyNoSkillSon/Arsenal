// SPDX-License-Identifier: MIT
import Foundation

extension ModelIntakeCommand {
    /// Pinned auxiliary assets derived by the ordinary loader have the same eligibility audit.
    /// Merge once into the tier receipt before matrix selection; never count an asset per case.
    static func mergeVoiceAuxiliaryCoverage(capture: URL, evidence: URL, label: String) throws {
        let assetFile = capture.appendingPathComponent("auxiliary-coverage.json")
        let coverageFile = evidence.appendingPathComponent(label + "-coverage.json")
        guard FileManager.default.fileExists(atPath: assetFile.path), FileManager.default.fileExists(atPath: coverageFile.path) else { return }
        var report = try WireJSON.decode(Data(contentsOf: coverageFile))
        guard report["auxiliary_coverage"] == nil else { return }
        let extra = try WireJSON.decode(Data(contentsOf: assetFile))
        if report["bits"]?.number == 8 {
            guard extra["packing_audit"]?.object?.count == Int(extra["quantized_layers"]?.number ?? -1) else {
                throw ManifestError.invalid("Auxiliary affine packing audit is incomplete")
            }
        }
        for field in ["parameters", "quantized_parameters", "fallback_parameters", "quantized_layers", "linear_layers"] {
            report[field] = .number((report[field]?.number ?? 0) + (extra[field]?.number ?? 0))
        }
        for field in ["kept", "component_eligible_parameters", "matrix_parameters", "matrix_components", "matrix_scales_sha256", "packing_audit"] {
            report[field] = .object((report[field]?.object ?? [:]).merging(extra[field]?.object ?? [:]) { _, b in b })
        }
        for field in ["quantized", "fallback_matrices"] {
            report[field] = .array(Set((report[field]?.array?.compactMap(\.string) ?? []) + (extra[field]?.array?.compactMap(\.string) ?? [])).sorted().map(JSONValue.string))
        }
        let kept = report["kept"]?.object ?? [:]
        report["kept_layers"] = .number(Double(kept.count))
        report["kept_reasons"] = .object(Dictionary(grouping: kept.values.compactMap(\.string), by: { $0 }).mapValues { .number(Double($0.count)) })
        report["auxiliary_coverage"] = .object(extra)
        try WireJSON.encode(report).write(to: coverageFile, options: .atomic)
    }
}
