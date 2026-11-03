// SPDX-License-Identifier: MIT
import Darwin
import Foundation

/// Outside-engine phase sampling. Fresh Load/operation phases never reuse a process-lifetime peak.
final class ServedFootprintPeak: @unchecked Sendable {
    private let lock = NSLock()
    private let pids: @Sendable () -> Set<Int32>
    private var timer: DispatchSourceTimer?
    private var peak: UInt64 = 0
    init(pids: @escaping @Sendable () -> Set<Int32> = { WorkerPIDRegistry.snapshot() }) { self.pids = pids }
    var value: UInt64? { sample(); return lock.withLock { peak == 0 ? nil : peak } }
    func start() {
        sample()
        let timer = DispatchSource.makeTimerSource(queue: .global(qos: .utility))
        timer.schedule(deadline: .now(), repeating: .milliseconds(20))
        timer.setEventHandler { [weak self] in self?.sample() }; self.timer = timer; timer.resume()
    }
    func stop() { timer?.cancel(); timer = nil; sample() }
    deinit { timer?.cancel() }
    static func processTree(roots: Set<Int32>) -> Set<Int32> {
        var result = roots, pending = Array(roots)
        while let pid = pending.popLast() {
            let capacity = max(16, Int(proc_listchildpids(pid, nil, 0)) / MemoryLayout<pid_t>.stride)
            var children = [pid_t](repeating: 0, count: capacity)
            let count = children.withUnsafeMutableBytes { proc_listchildpids(pid, $0.baseAddress, Int32($0.count)) }
            for child in children.prefix(max(0, Int(count))) where child > 0 && result.insert(child).inserted { pending.append(child) }
        }
        return result
    }
    private func sample() {
        var total: UInt64 = 0
        for pid in Self.processTree(roots: pids()) {
            var info = rusage_info_v4()
            let status = withUnsafeMutablePointer(to: &info) { pointer in
                pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(pid, RUSAGE_INFO_V4, $0) }
            }
            if status == 0 { total += info.ri_phys_footprint }
        }
        lock.withLock { peak = max(peak, total) }
    }
}
