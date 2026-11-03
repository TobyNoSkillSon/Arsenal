// SPDX-License-Identifier: MIT
// Ported from Projects/Vella/Sources/VellaCore/Residency/Residency.swift; see docs/notices/core.md.
import Foundation
import Darwin

public struct MemoryProbe: Sendable {
    public static let headroomMB = 512.0
    public let totalMB: Double
    public var marginMB: Double { max(1000, totalMB * 0.10) }
    public let testFile: URL?
    public let vmStatsFile: URL?
    public init(environment: [String: String] = ProcessInfo.processInfo.environment, totalMB: Double? = nil) {
        self.totalMB = totalMB ?? Double(ProcessInfo.processInfo.physicalMemory) / 1e6
        testFile = environment["ARSENAL_TEST_MEMORY_FILE"].flatMap { $0.hasPrefix("/") ? URL(fileURLWithPath: $0) : nil }
        vmStatsFile = environment["ARSENAL_TEST_VM_STATS"].flatMap { $0.hasPrefix("/") ? URL(fileURLWithPath: $0) : nil }
    }
    /// Never negative.
    public func availableMB(loadedMB: Double) -> Double { max(0, rawAvailableMB(loadedMB: loadedMB)) }
    /// `loadedMB`: Σ reclaimable MB of the models loaded now (used by the test file probe only).
    public func rawAvailableMB(loadedMB: Double) -> Double {
        if let testFile {
            let base = (Self.json(testFile)?["available_mb"] as? NSNumber)?.doubleValue ?? 0
            return base - loadedMB
        }
        let counters = vmStatsFile.map { Self.json($0) ?? [:] }
        return min(pagesMB(counters), levelPercent(counters) / 100 * totalMB) - marginMB
    }
    /// Estimated capacity reclaimable without swapping. File-backed pages include reclaimable inactive cache;
    /// anonymous inactive pages are not counted, since reclaiming those can require compression or swap.
    /// Speculative pages overlap free and file-backed counts, so count them only once.
    /// XNU definitions: https://github.com/apple-oss-distributions/xnu/blob/main/osfmk/mach/vm_statistics.h
    public func noSwapAvailableMB(loadedMB: Double) -> Double {
        if let testFile {
            return max(0, min(totalMB, ((Self.json(testFile)?["available_mb"] as? NSNumber)?.doubleValue ?? 0) - loadedMB))
        }
        return max(0, min(totalMB, pagesMB(vmStatsFile.map { Self.json($0) ?? [:] })))
    }
    private static func json(_ url: URL) -> [String: Any]? {
        (try? Data(contentsOf: url)).flatMap { try? JSONSerialization.jsonObject(with: $0) as? [String: Any] }
    }
    public static func reclaimableMB(free: Double, speculative: Double, external: Double, purgeable: Double, pageSize: Double) -> Double {
        (max(0, free - speculative) + external + purgeable) * pageSize / 1e6
    }
    private func pagesMB(_ counters: [String: Any]?) -> Double {
        if let counters {
            func value(_ key: String) -> Double { (counters[key] as? NSNumber)?.doubleValue ?? 0 }
            return Self.reclaimableMB(
                free: value("free_count"), speculative: value("speculative_count"),
                external: value("external_page_count"), purgeable: value("purgeable_count"), pageSize: value("page_size"))
        }
        var info = vm_statistics64_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64_data_t>.stride / MemoryLayout<integer_t>.stride)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) { host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count) }
        }
        guard result == KERN_SUCCESS else { return 0 }
        return Self.reclaimableMB(
            free: Double(info.free_count), speculative: Double(info.speculative_count),
            external: Double(info.external_page_count), purgeable: Double(info.purgeable_count),
            pageSize: Double(Self.kernelPageSize))
    }
    /// The kernel's VM page size (`vm_kernel_page_size`), which the page counts above are in.
    static let kernelPageSize: UInt64 = {
        var size: vm_size_t = 0
        return host_page_size(mach_host_self(), &size) == KERN_SUCCESS ? UInt64(size) : UInt64(getpagesize())
    }()
    private func levelPercent(_ counters: [String: Any]?) -> Double {
        if let level = counters?["memorystatus_level"] as? NSNumber { return level.doubleValue }
        return Self.levelPercent()
    }
    public static func levelPercent() -> Double {
        var value: Int32 = 100; var size = MemoryLayout<Int32>.size
        return sysctlbyname("kern.memorystatus_level", &value, &size, nil, 0) == 0 ? Double(value) : 100
    }
}
