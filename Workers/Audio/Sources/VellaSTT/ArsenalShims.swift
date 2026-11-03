// SPDX-License-Identifier: MIT
// Arsenal-side stand-ins for the two Vella-app pieces the vendored model code names. Vella's child-process self-test
// (FastPathGate) and its wire vocabulary (VellaWire) are not part of Arsenal; see PROVENANCE.md.
import Foundation

/// VellaWire's recipe vocabulary, which KeptLevers reads. Arsenal selects levers explicitly (`KeptLevers(enabled:)`).
public enum Recipe: String, Codable, CaseIterable, Sendable {
    case standard, optimized_exact, optimized_fast
    public static let variable = "VELLA_RECIPE"
}

/// Vella's gate writes kernel diagnostics to a lab log; Arsenal has no such log, so the hook is silent.
enum FastPathGate {
    static func debug(_ line: String) {}
}

public extension KeptLevers {
    /// An explicit lever set (Arsenal picks the levers Vella ships for the model instead of reading Vella's environment).
    init(enabled: Set<String>) { self.enabled = enabled.intersection(Self.switches) }
}
