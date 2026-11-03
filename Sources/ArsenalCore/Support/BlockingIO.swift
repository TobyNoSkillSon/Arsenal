// SPDX-License-Identifier: MIT
import Foundation

/// Blocking file/Git work and bounded pipe writes stay off Swift's cooperative executor.
/// Long-lived worker pipe/exit waits use Dispatch sources instead.
func processIO<T: Sendable>(_ operation: @escaping @Sendable () throws -> T) async throws -> T {
    try await withCheckedThrowingContinuation { reply in
        DispatchQueue.global(qos: .utility).async {
            do { reply.resume(returning: try operation()) } catch { reply.resume(throwing: error) }
        }
    }
}
