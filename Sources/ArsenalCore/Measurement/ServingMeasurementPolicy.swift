// SPDX-License-Identifier: MIT
import Foundation

public enum ServingMeasurementPolicy {
    public static func refusal(model: String, fast: Bool) -> String? {
        if model == "re-use" {
            return "RE-USE measurement refused: the ordinary serving loader widens weights to FP32; this is not a 16-bit reference."
        }
        if model.hasPrefix("stable-audio-3-") {
            return fast
                ? "Stable Audio measurement refused: ordinary Fast qualification runs an FP32 stock reference and may fall back to an FP32 residual stream/promoted DiT weights."
                : "Stable Audio Exact measurement refused: the serving path computes in FP32 and can retain promoted DiT weights; this is not a 16-bit reference."
        }
        return nil
    }
}
