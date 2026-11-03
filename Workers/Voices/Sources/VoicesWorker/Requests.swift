// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import VireoCore

final class Requests: @unchecked Sendable {
    private let lock = NSLock()
    private var active: (JSONValue, GenerationControl)?
    func admit(_ id: JSONValue) -> GenerationControl? {
        lock.withLock {
            guard active == nil else { return nil }; let c = GenerationControl(); active = (id, c); return c
        }
    }
    func cancel(_ id: JSONValue) -> Bool {
        lock.withLock {
            guard let a = active, a.0 == id else { return false }; a.1.cancel(); return true
        }
    }
    func release() { lock.withLock { active = nil } }
    func stop() { lock.withLock { active?.1.cancel() } }
}
