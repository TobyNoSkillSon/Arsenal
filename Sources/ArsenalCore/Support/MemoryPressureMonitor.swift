// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import IOKit

final class MemoryPressureMonitor: @unchecked Sendable {
    private let source: DispatchSourceMemoryPressure
    init(handler: @escaping @Sendable () -> Void) {
        source = DispatchSource.makeMemoryPressureSource(eventMask: [.critical], queue: .global(qos: .utility)); source.setEventHandler(handler: handler); source.resume()
    }
    deinit { source.cancel() }
}
public enum HostChip {
    public static func slug(_ brand: String) -> String {
        brand.lowercased().replacingOccurrences(of: "apple ", with: "").split(whereSeparator: { $0.isWhitespace }).joined(separator: "-")
    }
    public static let hardwareModel: String = {
        var size = 0; guard sysctlbyname("hw.model", nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: size); guard sysctlbyname("hw.model", &bytes, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }()
    public static let gpuCores: Int = {
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AGXAccelerator"))
        guard service != 0 else { return 0 }; defer { IOObjectRelease(service) }
        return (IORegistryEntryCreateCFProperty(service, "gpu-core-count" as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? NSNumber)?.intValue ?? 0
    }()
    public static var current: String { slug(brand) }
    public static let brand: String = {
        var size = 0; guard sysctlbyname("machdep.cpu.brand_string", nil, &size, nil, 0) == 0, size > 0 else { return "unknown" }
        var bytes = [CChar](repeating: 0, count: size); guard sysctlbyname("machdep.cpu.brand_string", &bytes, &size, nil, 0) == 0 else { return "unknown" }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }()
}
