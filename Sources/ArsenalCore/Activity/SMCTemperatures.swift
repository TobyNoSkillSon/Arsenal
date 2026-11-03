// SPDX-License-Identifier: MIT
import Foundation
import IOKit

/// CPU and GPU die temperatures from the System Management Controller's float sensors, read through the AppleSMC user
/// client (no privileges needed). The keys are undocumented: which family belongs to which part was established on an
/// M5 Max (macOS 27) by load — a 12 s all-core CPU load raised `Tp*` by 12 °C and `Tm*` by 14 °C (`Tg*` 6 °C); a 12 s
/// Metal compute load raised `Tg*` by 21 °C (`Tp*`/`Tm*` 10 °C). A Mac without these families reports nothing, never a
/// guess. Key discovery (all ~3,800 keys) runs once per process; each later reading touches only the chosen keys.
final class SMCTemperatures: @unchecked Sendable {
    static let cpuFamilies = ["Tp", "Tm"], gpuFamilies = ["Tg"]
    private var connection: io_connect_t = 0
    private var cpuKeys: [UInt32] = [], gpuKeys: [UInt32] = []
    private var discovered = false

    deinit { if connection != 0 { IOServiceClose(connection) } }

    /// Mean °C of each family's plausible sensors; nil where none reads.
    func read() -> (cpu: Double?, gpu: Double?) {
        guard open() else { return (nil, nil) }
        if !discovered { discover() }
        return (mean(cpuKeys), mean(gpuKeys))
    }

    private func open() -> Bool {
        if connection != 0 { return true }
        let service = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSMC"))
        guard service != 0 else { return false }
        defer { IOObjectRelease(service) }
        return IOServiceOpen(service, mach_task_self_, 0, &connection) == KERN_SUCCESS
    }

    private func discover() {
        discovered = true
        guard let count = value(Self.code("#KEY")).map({ $0.bytes.reduce(UInt32(0)) { $0 << 8 | UInt32($1) } }), count < 20_000 else { return }
        for index in 0..<count {
            guard let key = keyAt(index) else { continue }
            let family = String(Self.name(key).prefix(2))
            let cpu = Self.cpuFamilies.contains(family), gpu = Self.gpuFamilies.contains(family)
            guard cpu || gpu, temperature(key) != nil else { continue }
            if cpu { cpuKeys.append(key) } else { gpuKeys.append(key) }
        }
    }

    private func mean(_ keys: [UInt32]) -> Double? {
        let values = keys.compactMap(temperature)
        guard !values.isEmpty else { return nil }
        return values.reduce(0, +) / Double(values.count)
    }

    /// A float sensor's reading when it is plausible for a die (5–130 °C).
    private func temperature(_ key: UInt32) -> Double? {
        guard let v = value(key), v.type == Self.code("flt "), v.bytes.count == 4 else { return nil }
        let t = Double(v.bytes.withUnsafeBytes { $0.loadUnaligned(as: Float.self) })
        return t.isFinite && t > 5 && t < 130 ? t : nil
    }

    // MARK: AppleSMC user client (selector 2, the struct the kernel extension expects)

    fileprivate struct Version { var major: UInt8 = 0, minor: UInt8 = 0, build: UInt8 = 0, reserved: UInt8 = 0, release: UInt16 = 0 }
    fileprivate struct Limit { var version: UInt16 = 0, length: UInt16 = 0, cpu: UInt32 = 0, gpu: UInt32 = 0, memory: UInt32 = 0 }
    fileprivate struct KeyInfo { var size: UInt32 = 0, type: UInt32 = 0, attributes: UInt8 = 0 }
    fileprivate struct Parameter {
        var key: UInt32 = 0
        var version = Version()
        var limit = Limit()
        var info = KeyInfo()
        var padding: UInt16 = 0
        var result: UInt8 = 0
        var status: UInt8 = 0
        var command: UInt8 = 0
        var index: UInt32 = 0
        /// 32 bytes at offset 48 of the kernel's 80-byte struct (the 8-byte element keeps that offset and size).
        var bytes: (UInt64, UInt64, UInt64, UInt64) = (0, 0, 0, 0)
    }
    static var parameterLayout: (size: Int, bytesOffset: Int?) { (MemoryLayout<Parameter>.size, MemoryLayout<Parameter>.offset(of: \Parameter.bytes)) }
    private static let readBytes: UInt8 = 5, readIndex: UInt8 = 8, readInfo: UInt8 = 9

    private func call(_ input: Parameter) -> Parameter? {
        var input = input, output = Parameter(), size = MemoryLayout<Parameter>.stride
        let status = IOConnectCallStructMethod(connection, 2, &input, MemoryLayout<Parameter>.stride, &output, &size)
        return status == KERN_SUCCESS && output.result == 0 ? output : nil
    }
    private func keyAt(_ index: UInt32) -> UInt32? {
        var p = Parameter(); p.command = Self.readIndex; p.index = index
        return call(p)?.key
    }
    private var infos: [UInt32: KeyInfo] = [:]
    private func value(_ key: UInt32) -> (type: UInt32, bytes: [UInt8])? {
        let info: KeyInfo
        if let known = infos[key] {
            info = known
        } else {
            var p = Parameter(); p.key = key; p.command = Self.readInfo
            guard let found = call(p)?.info else { return nil }
            info = found; infos[key] = found
        }
        guard info.size <= 32 else { return nil }
        var p = Parameter(); p.key = key; p.info.size = info.size; p.command = Self.readBytes
        guard let out = call(p) else { return nil }
        return (info.type, withUnsafeBytes(of: out.bytes) { Array($0.prefix(Int(info.size))) })
    }
    static func code(_ s: String) -> UInt32 { s.utf8.reduce(0) { $0 << 8 | UInt32($1) } }
    static func name(_ v: UInt32) -> String {
        String(decoding: [UInt8(v >> 24), UInt8(v >> 16 & 255), UInt8(v >> 8 & 255), UInt8(v & 255)], as: UTF8.self)
    }
}
