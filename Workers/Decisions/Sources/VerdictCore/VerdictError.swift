// From Projects/Verdict/Sources/VerdictCore/Core.swift @ f8177ed (Apache-2.0): the error Safetensors throws.
import Foundation

public enum VerdictError: LocalizedError {
    case message(String)
    public var errorDescription: String? { switch self { case .message(let m): return m } }
}
