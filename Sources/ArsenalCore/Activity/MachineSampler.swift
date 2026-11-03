// SPDX-License-Identifier: MIT
import Foundation
import IOKit

/// IORegistry counters are undocumented. A missing/reset counter stays unknown; it never becomes a fabricated 0%.
/// One sample reads: the GPU's own utilisation (one property per accelerator), all cores' tick counters, the SMC's die
/// sensors, the VM page counts and this process's footprint. It no longer walks every GPU client.
struct MachineSampler {
    private var previousTicks: (busy: UInt64, total: UInt64)?
    private let temperatures = SMCTemperatures()
    mutating func reset() { previousTicks = nil }
    mutating func sample() -> MachineActivity {
        var iterator: io_iterator_t = 0
        var busy: Double?
        if IOServiceGetMatchingServices(kIOMainPortDefault, IOServiceMatching("IOAccelerator"), &iterator) == KERN_SUCCESS {
            defer { IOObjectRelease(iterator) }
            while case let accelerator = IOIteratorNext(iterator), accelerator != 0 {
                defer { IOObjectRelease(accelerator) }
                if let stats = Self.property(accelerator, "PerformanceStatistics") as? [String: Any],
                    let value = stats["Device Utilization %"] as? NSNumber, value.doubleValue.isFinite
                {
                    busy = max(busy ?? 0, min(100, max(0, value.doubleValue)))
                }
            }
        }
        let thermal: String
        switch ProcessInfo.processInfo.thermalState {
        case .nominal: thermal = "nominal"
        case .fair: thermal = "fair"
        case .serious: thermal = "serious"
        case .critical: thermal = "critical"
        @unknown default: thermal = "unknown"
        }
        var cpu: Double?
        if let ticks = Self.cpuTicks() {
            if let previous = previousTicks, ticks.total > previous.total, ticks.busy >= previous.busy {
                cpu = min(100, Double(ticks.busy - previous.busy) / Double(ticks.total - previous.total) * 100)
            }
            previousTicks = ticks
        }
        let temps = temperatures.read()
        let memory = Self.memory()
        return .init(
            gpuBusyPct: busy, thermal: thermal, cpuBusyPct: cpu, cpuTempC: temps.cpu, gpuTempC: temps.gpu, memoryUsedMB: memory?.used,
            memoryTotalMB: memory?.total, arsenalMB: Self.footprintMB())
    }

    /// Busy and total ticks summed over every core (user + system + nice are busy).
    static func cpuTicks() -> (busy: UInt64, total: UInt64)? {
        var count: natural_t = 0, info: processor_info_array_t?, infoCount: mach_msg_type_number_t = 0
        guard host_processor_info(mach_host_self(), PROCESSOR_CPU_LOAD_INFO, &count, &info, &infoCount) == KERN_SUCCESS, let info else { return nil }
        defer { vm_deallocate(mach_task_self_, vm_address_t(bitPattern: info), vm_size_t(Int(infoCount) * MemoryLayout<integer_t>.stride)) }
        var busy: UInt64 = 0, total: UInt64 = 0
        for cpu in 0..<Int(count) {
            let base = cpu * Int(CPU_STATE_MAX)
            let user = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_USER)])), system = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_SYSTEM)]))
            let nice = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_NICE)])), idle = UInt64(UInt32(bitPattern: info[base + Int(CPU_STATE_IDLE)]))
            busy += user + system + nice; total += user + system + nice + idle
        }
        return (busy, total)
    }

    /// Activity Monitor's "Memory Used": app memory (internal − purgeable) + wired + compressed, in MB.
    static func memory() -> (used: Double, total: Double)? {
        var stats = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let status = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard status == KERN_SUCCESS else { return nil }
        var pageSize: vm_size_t = 0
        host_page_size(mach_host_self(), &pageSize)
        let page = Double(pageSize)
        let app = Double(stats.internal_page_count) - Double(stats.purgeable_count)
        let used = (max(0, app) + Double(stats.wire_count) + Double(stats.compressor_page_count)) * page / 1e6
        return (used, Double(ProcessInfo.processInfo.physicalMemory) / 1e6)
    }

    /// This process's physical footprint (what Activity Monitor shows as its Memory), in MB.
    static func footprintMB() -> Double? {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.stride / MemoryLayout<natural_t>.stride)
        let status = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count) }
        }
        return status == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : nil
    }
    private static func property(_ entry: io_registry_entry_t, _ key: String) -> Any? {
        IORegistryEntryCreateCFProperty(entry, key as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue()
    }
}

/// Blocking IORegistry work has its own executor; it never occupies the request state or subscriber actor.
actor ActivityMachineSampler {
    private var sampler = MachineSampler()
    private let fixture: (@Sendable () -> MachineActivity)?
    init(fixture: (@Sendable () -> MachineActivity)? = nil) { self.fixture = fixture }
    func sample(reset: Bool) -> MachineActivity {
        guard !Task.isCancelled else { return .init() }
        if reset { sampler.reset() }
        return fixture?() ?? sampler.sample()
    }
}
