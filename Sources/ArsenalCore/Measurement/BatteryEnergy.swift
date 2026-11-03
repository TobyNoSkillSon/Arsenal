// SPDX-License-Identifier: MIT
import Foundation
import IOKit
import Darwin

public struct CPUGPUEnergy: Sendable, Equatable {
    public var cpuJoules: Double
    public var gpuJoules: Double
    public var joules: Double { cpuJoules + gpuJoules }
    public init(cpuJoules: Double, gpuJoules: Double) { self.cpuJoules = cpuJoules; self.gpuJoules = gpuJoules }
}

public enum BatteryEnergy {
    /// mAh × mV × 0.0036 = joules. Full-charge capacity accounts for battery wear.
    public static func capacityJoules(milliampHours: Double, millivolts: Double) throws -> Double {
        guard milliampHours.isFinite, millivolts.isFinite, milliampHours > 0, millivolts > 0 else { throw ManifestError.invalid("Battery capacity unavailable") }
        return milliampHours * millivolts * 0.0036
    }
    public static func percent(joules: Double, capacityJoules: Double, completedUnits: Double, statedUnits: Double) throws -> Double {
        guard [joules, capacityJoules, completedUnits, statedUnits].allSatisfy(\.isFinite), joules >= 0,
            capacityJoules > 0, completedUnits > 0, statedUnits > 0
        else { throw ManifestError.invalid("Invalid battery energy denominator") }
        let percent = joules / capacityJoules * 100 * statedUnits / completedUnits
        // Two significant digits reflect the approximate counters; do not imply microscopic precision.
        if percent == 0 { return 0 }
        let scale = pow(10, 1 - floor(log10(percent)))
        return (percent * scale).rounded() / scale
    }
    public static func liveCapacityJoules() throws -> Double {
        let battery = IOServiceGetMatchingService(kIOMainPortDefault, IOServiceMatching("AppleSmartBattery"))
        guard battery != 0 else { throw ManifestError.invalid("This Mac has no battery capacity counter") }
        defer { IOObjectRelease(battery) }
        func number(_ key: String) -> Double? { (IORegistryEntryCreateCFProperty(battery, key as CFString, kCFAllocatorDefault, 0)?.takeRetainedValue() as? NSNumber)?.doubleValue }
        guard let capacity = number("AppleRawMaxCapacity") ?? number("DesignCapacity"),
            let voltage = number("DesignVoltage") ?? number("Voltage")
        else { throw ManifestError.invalid("Battery capacity/voltage unavailable") }
        return try capacityJoules(milliampHours: capacity, millivolts: voltage)
    }
}

/// Rootless IOReport Energy Model subscription. Nothing is sampled until explicitly requested.
/// Whole-machine CPU+GPU energy is approximate on a busy Mac; no ANE/DRAM/display channels are added.
public final class IOReportEnergyMeter {
    private typealias CopyChannels = @convention(c) (CFString?, CFString?, UInt64, UInt64, UInt64) -> Unmanaged<CFDictionary>?
    private typealias CreateSubscription =
        @convention(c) (UnsafeMutableRawPointer?, CFDictionary?, UnsafeMutablePointer<Unmanaged<CFDictionary>?>?, UInt64, CFTypeRef?) -> UnsafeMutableRawPointer?
    private typealias CreateSamples = @convention(c) (UnsafeMutableRawPointer?, CFDictionary?, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias CreateDelta = @convention(c) (CFDictionary?, CFDictionary?, CFTypeRef?) -> Unmanaged<CFDictionary>?
    private typealias ChannelString = @convention(c) (CFDictionary?) -> Unmanaged<CFString>?
    private typealias ChannelValue = @convention(c) (CFDictionary?, Int32) -> Int64
    private let library: UnsafeMutableRawPointer
    private let subscription: UnsafeMutableRawPointer
    private let channels: CFDictionary
    private let sampleFunction: CreateSamples
    private let deltaFunction: CreateDelta
    private let nameFunction: ChannelString
    private let unitFunction: ChannelString
    private let valueFunction: ChannelValue
    private var start: CFDictionary?

    public init() throws {
        guard let handle = dlopen("/usr/lib/libIOReport.dylib", RTLD_NOW) else { throw ManifestError.invalid("IOReport is unavailable") }
        var keep = false
        defer { if !keep { dlclose(handle) } }
        func symbol<T>(_ name: String, _ type: T.Type) throws -> T {
            guard let pointer = dlsym(handle, name) else { throw ManifestError.invalid("IOReport symbol unavailable: \(name)") }
            return unsafeBitCast(pointer, to: type)
        }
        let copy = try symbol("IOReportCopyChannelsInGroup", CopyChannels.self)
        let create = try symbol("IOReportCreateSubscription", CreateSubscription.self)
        sampleFunction = try symbol("IOReportCreateSamples", CreateSamples.self)
        deltaFunction = try symbol("IOReportCreateSamplesDelta", CreateDelta.self)
        nameFunction = try symbol("IOReportChannelGetChannelName", ChannelString.self)
        unitFunction = try symbol("IOReportChannelGetUnitLabel", ChannelString.self)
        valueFunction = try symbol("IOReportSimpleGetIntegerValue", ChannelValue.self)
        guard let group = copy("Energy Model" as CFString, nil, 0, 0, 0)?.takeRetainedValue() else { throw ManifestError.invalid("Energy Model counters unavailable") }
        var subscribed: Unmanaged<CFDictionary>?
        guard let sub = create(nil, group, &subscribed, 0, nil), let subscribed else { throw ManifestError.invalid("IOReport subscription refused without root") }
        library = handle; subscription = sub; channels = subscribed.takeRetainedValue(); keep = true
    }
    deinit {
        // IOReport subscriptions are CF objects, returned through an opaque ABI pointer.
        _ = Unmanaged<AnyObject>.fromOpaque(subscription).takeRetainedValue()
        dlclose(library)
    }
    public func begin() throws {
        guard let sample = sampleFunction(subscription, channels, nil)?.takeRetainedValue() else { throw ManifestError.invalid("Energy sample unavailable") }
        start = sample
    }
    public func end() throws -> CPUGPUEnergy {
        guard let start, let end = sampleFunction(subscription, channels, nil)?.takeRetainedValue(),
            let delta = deltaFunction(start, end, nil)?.takeRetainedValue(),
            let rows = (delta as NSDictionary)["IOReportChannels"] as? [NSDictionary]
        else { throw ManifestError.invalid("Energy delta unavailable") }
        self.start = nil
        var counters: [String: Double] = [:]
        for row in rows {
            let channel = row as CFDictionary
            guard let name = nameFunction(channel)?.takeUnretainedValue() as String? else { continue }
            guard name.hasSuffix("CPU Energy") || name == "GPU Energy" else { continue }
            guard let unit = unitFunction(channel)?.takeUnretainedValue() as String?,
                let scale = Self.jouleScale(unit.trimmingCharacters(in: .whitespaces)), counters[name] == nil
            else {
                throw ManifestError.invalid("CPU/GPU energy channel unit unavailable or duplicated: " + name)
            }
            let value = Double(valueFunction(channel, 0)) * scale
            guard value.isFinite, value >= 0 else { throw ManifestError.invalid("CPU/GPU energy channel reset or invalid: " + name) }
            counters[name] = value
        }
        return try Self.cpuGPU(counters)
    }
    public static func jouleScale(_ unit: String) -> Double? { ["J": 1, "mJ": 1e-3, "uJ": 1e-6, "nJ": 1e-9][unit] }
    public static func cpuGPU(_ counters: [String: Double]) throws -> CPUGPUEnergy {
        let clusters = counters.filter { $0.key != "CPU Energy" && $0.key.hasSuffix("CPU Energy") }
        let recognized: [Set<String>] = [["ECPU Energy", "PCPU Energy"], ["ECPU Energy", "P0CPU Energy", "P1CPU Energy"]]
        guard clusters.values.allSatisfy({ $0.isFinite && $0 >= 0 }),
            let gpu = counters["GPU Energy"], gpu.isFinite, gpu >= 0
        else {
            throw ManifestError.invalid("CPU/GPU counters missing, reset or invalid")
        }
        guard clusters.isEmpty || recognized.contains(Set(clusters.keys)) else { throw ManifestError.invalid("Complete recognized CPU cluster energy set unavailable") }
        let cpu: Double
        if let aggregate = counters["CPU Energy"] {
            cpu = aggregate
        } else {
            guard recognized.contains(Set(clusters.keys)) else { throw ManifestError.invalid("Complete recognized CPU cluster energy set unavailable") }
            cpu = clusters.values.reduce(0, +)
        }
        guard cpu.isFinite, cpu >= 0 else { throw ManifestError.invalid("CPU/GPU counters missing, reset or invalid") }
        return CPUGPUEnergy(cpuJoules: cpu, gpuJoules: gpu)
    }
}
