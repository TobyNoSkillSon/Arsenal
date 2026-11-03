// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
import Foundation
public enum MoGeWeightLoader {
    public static func remapWeightKey(_ key: String) -> String {
        // Resampler: insert `.layers` between resampler index and child index
        if let range = key.range(of: #"\.resamplers\.\d+\."#, options: .regularExpression) {
            let prefix = String(key[..<range.upperBound]) // up to and incl. "resamplers.N."
            let rest = String(key[range.upperBound...]) // "J.param..."
            if let dot = rest.firstIndex(of: ".") {
                let childIdx = rest[..<dot]
                let param = rest[rest.index(after: dot)...]
                return "\(prefix)layers.\(childIdx).\(param)"
            }
        }

        // ScaleHead: `scale_head.{j}.param` → `scale_head.layers.{j}.param`
        if key.hasPrefix("scale_head.") {
            let tail = key.dropFirst("scale_head.".count)
            if let dot = tail.firstIndex(of: "."),
                Int(tail[..<dot]) != nil
            {
                return "scale_head.layers.\(tail)"
            }
        }

        return key
    }
}

// swiftlint:enable all
