// SPDX-License-Identifier: MIT
import Foundation

public struct PrecisionRecommendation: Codable, Sendable, Equatable {
    public var dtype: String
    public var pending: Bool
    public var reason: String
    public init(dtype: String, pending: Bool, reason: String) { self.dtype = dtype; self.pending = pending; self.reason = reason }
    public static func choose(_ sets: [String: PrecisionStatSet], baseDtype: String, chip: String) -> Self {
        let pending = Self(dtype: baseDtype, pending: true, reason: "16-bit until comparable speed and own-16-bit quality measurements are available.")
        guard let base = sets[baseDtype] else { return pending }
        let speed: StatMetric
        let higher: Bool
        let quality: StatMetric
        switch base.kind {
        case .decisions: speed = .millisecondsPerDecision; higher = false; quality = .agreementPercent
        case .chat: speed = .tokensPerSecond; higher = true; quality = .nextTokenAgreementPercent
        case .images: speed = .secondsPerImage; higher = false; quality = .psnrDB
        case .voices, .sound, .enhancement: speed = .realTimeFactor; higher = false; quality = .spectralDistance
        // No own-16-bit quality metric: the 16-bit format stays.
        case .audio, .embeddings, .perception, .restoration, .geometry: return pending
        }
        guard let baseline = base.value(speed).provenance, base.value(speed).state == .measured, baseline.chip == chip else { return pending }
        let eligible = sets.values.filter { set in
            let timing = set.value(speed), fidelity = set.value(quality)
            return set.qualityCheckPassed != false && set.availability.state == .measured && timing.state == .measured && fidelity.state == .measured
                && fidelity.provenance?.baseline == "own " + baseDtype
                && set.runner == base.runner && set.kind == base.kind
                && PrecisionEvidence.matches(set.availability.provenance, fidelity.provenance)
                && PrecisionEvidence.matches(fidelity.provenance, timing.provenance)
                && PrecisionEvidence.matches(timing.provenance, baseline)
                && timing.value.map { $0.isFinite && $0 > 0 } == true && fidelity.value?.isFinite == true
                && timing.provenance?.chip == baseline.chip && timing.provenance?.method == baseline.method
                && timing.provenance?.baseline == baseline.baseline && timing.provenance?.source == baseline.source
        }
        guard eligible.contains(where: { $0.dtype == baseDtype }),
            let fastest = eligible.sorted(by: { a, b in
                let left = a.value(speed).value ?? 0, right = b.value(speed).value ?? 0
                if left == right { return a.dtype == baseDtype || (b.dtype != baseDtype && a.dtype < b.dtype) }
                return higher ? left > right : left < right
            }).first
        else { return pending }
        return Self(dtype: fastest.dtype, pending: false, reason: "Fastest measured precision that passes the recorded own-16-bit quality gate on " + chip + ".")
    }
}
