// SPDX-License-Identifier: MIT
import Foundation

/// The Exact checker: two outputs are the same numbers only when every byte is equal. Not a tolerance: +0 vs −0 and
/// two different NaN payloads count as differences, because they are different outputs of the reference runner.
public struct ExactComparison: Sendable, Equatable, CustomStringConvertible {
    public enum Element: String, Sendable { case float16, bfloat16, float32, bytes }

    public let elements: Int
    public let mismatches: Int
    /// First differing element index (nil when equal).
    public let firstMismatch: Int?
    /// Largest |a − b| over differing finite elements (0 when equal; ∞ when a difference involves a non-finite value).
    public let maxAbsDiff: Double
    /// Shapes or element types differ: nothing was compared element-wise.
    public let layoutMismatch: String?

    public var identical: Bool { layoutMismatch == nil && mismatches == 0 }

    public var description: String {
        if let layoutMismatch { return "not comparable: \(layoutMismatch)" }
        if identical { return "bit-identical (\(elements) elements)" }
        return "\(mismatches) of \(elements) elements differ (first at \(firstMismatch ?? -1), max |diff| \(String(format: "%.3g", maxAbsDiff)))"
    }

    public static func layout(_ why: String) -> ExactComparison {
        ExactComparison(elements: 0, mismatches: 0, firstMismatch: nil, maxAbsDiff: .infinity, layoutMismatch: why)
    }

    /// Compares raw element buffers of the same element type.
    public static func compare(_ a: Data, _ b: Data, element: Element) -> ExactComparison {
        guard a.count == b.count else { return .layout("\(a.count) vs \(b.count) bytes") }
        let width: Int
        switch element {
        case .float16, .bfloat16: width = 2
        case .float32: width = 4
        case .bytes: width = 1
        }
        guard a.count % width == 0 else { return .layout("\(a.count) bytes is not a whole number of \(element.rawValue) elements") }
        let count = a.count / width
        var mismatches = 0, first: Int?, maxDiff = 0.0
        a.withUnsafeBytes { pa in
            b.withUnsafeBytes { pb in
                for i in 0..<count {
                    let x = pa.loadUnaligned(fromByteOffset: i * width, as: UInt32.self, width: width)
                    let y = pb.loadUnaligned(fromByteOffset: i * width, as: UInt32.self, width: width)
                    guard x != y else { continue }
                    mismatches += 1
                    if first == nil { first = i }
                    let d = abs(value(x, element) - value(y, element))
                    maxDiff = Swift.max(maxDiff, d.isFinite ? d : .infinity)
                }
            }
        }
        return ExactComparison(elements: count, mismatches: mismatches, firstMismatch: first, maxAbsDiff: maxDiff, layoutMismatch: nil)
    }

    static func value(_ bits: UInt32, _ element: Element) -> Double {
        switch element {
        case .float16: return Double(Float16(bitPattern: UInt16(truncatingIfNeeded: bits)))
        case .bfloat16: return Double(Float(bitPattern: bits << 16))
        case .float32: return Double(Float(bitPattern: bits))
        case .bytes: return Double(bits)
        }
    }
}

private extension UnsafeRawBufferPointer {
    func loadUnaligned(fromByteOffset offset: Int, as: UInt32.Type, width: Int) -> UInt32 {
        switch width {
        case 1: return UInt32(self[offset])
        case 2: return UInt32(loadUnaligned(fromByteOffset: offset, as: UInt16.self))
        default: return loadUnaligned(fromByteOffset: offset, as: UInt32.self)
        }
    }
}
