// SPDX-License-Identifier: MIT
import Foundation

public enum WorkerThread {
    /// A process-lifetime blocking stdin reader must not occupy GCD's shared pool or Swift's cooperative executor.
    /// The reader exits on EOF; it sleeps in read(2) at idle, with no timers or polling.
    public static func start(name: String, body: @escaping @Sendable () -> Void) {
        let thread = Thread(block: body)
        thread.name = name
        thread.start()
    }
}
