import Foundation
import IntegrationKit
import MLX
final class SeparationMonitor {
    let cancellation: NativeCancellation
    init(_ cancellation: NativeCancellation) { self.cancellation = cancellation }
    func scoped(start: Float, end: Float) -> SeparationMonitor { self }
    func checkCancellation() throws { try cancellation.check() }
    func trace(_ value: MLXArray, stage: String) throws {
        if ProcessInfo.processInfo.environment["ARSENAL_AUDIO_TRACE"] == "1" {
            let count = MLX.sum((.!MLX.isFinite(value)).asType(.int32)).item(Int.self)
            FileHandle.standardError.write(Data((stage + ": nonfinite=\(count), dtype=\(value.dtype), shape=\(value.shape)\n").utf8))
        }
    }
    func reportProgress(_ progress: Float, stage: String) {}
}
