// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// Short telemetry processes use files instead of pipes: ioreg output can exceed a pipe buffer.
public enum BenchCommand {
    public static func output(_ executable: String, _ arguments: [String], timeout: TimeInterval = 8) throws -> String {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-bench-telemetry")
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let file = directory.appendingPathComponent(UUID().uuidString)
        _ = FileManager.default.createFile(atPath: file.path, contents: nil)
        let handle = try FileHandle(forWritingTo: file)
        defer { try? handle.close(); try? FileManager.default.removeItem(at: file) }
        let process = Process(); process.executableURL = URL(fileURLWithPath: executable); process.arguments = arguments
        process.standardOutput = handle; process.standardError = FileHandle.nullDevice
        try process.run()
        let deadline = Date().addingTimeInterval(timeout)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.02) }
        if process.isRunning {
            process.terminate(); Thread.sleep(forTimeInterval: 0.1)
            if process.isRunning { kill(process.processIdentifier, SIGKILL) }
            throw BenchmarkError.invalid("Benchmark telemetry command timed out.")
        }
        guard process.terminationStatus == 0 else { throw BenchmarkError.invalid("Benchmark telemetry command failed.") }
        let data = try Data(contentsOf: file)
        guard data.count <= 8 * 1024 * 1024 else { throw BenchmarkError.invalid("Benchmark telemetry exceeded its output limit.") }
        return String(decoding: data, as: UTF8.self)
    }
}

public enum BenchmarkGate {
    public static func check(
        repository: URL, environment: [String: String] = ProcessInfo.processInfo.environment,
        requireLock: Bool = true, requireMeasurement: Bool = true
    ) throws {
        let main = URL(fileURLWithPath: repository.path.components(separatedBy: "/.worktrees/")[0])
        if FileManager.default.fileExists(atPath: main.appendingPathComponent(".gpu-hold").path) {
            throw BenchmarkError.invalid("GPU gate closed: coordinator hold; benchmark not started.")
        }
        let vella = main.deletingLastPathComponent().appendingPathComponent("Vella")
        if !FileManager.default.fileExists(atPath: vella.appendingPathComponent("lab/bench/full-measure.STOP").path),
            let pids = try? BenchCommand.output("/usr/bin/pgrep", ["-f", "full-measure.py"]), !pids.isEmpty
        {
            throw BenchmarkError.invalid("GPU gate closed: Vella measurement window is open.")
        }
        if requireLock {
            let mode = requireMeasurement ? "measure" : "gpulock"
            guard environment["VELLA_MEASURE_WINDOW"] == (requireMeasurement ? "1" : "0"), environment["VELLA_LOCK_HELD"] == "1",
                let holder = try? String(contentsOf: vella.appendingPathComponent("lab/locks/gpu.lock"), encoding: .utf8),
                holder.split(separator: " ").dropFirst().first == Substring(mode),
                let pidText = holder.split(separator: " ").first, let pid = Int32(pidText),
                ancestry().contains(pid)
            else {
                throw BenchmarkError.invalid("Real benchmarks require scripts/mlock \(mode) <label> -- arsenal-bench run …")
            }
        }
    }
    static func ancestry() -> Set<Int32> {
        var ids: Set<Int32> = [getpid()]; var pid = getpid()
        for _ in 0..<32 {
            guard let output = try? BenchCommand.output("/bin/ps", ["-o", "ppid=", "-p", String(pid)]),
                let parent = Int32(output.trimmingCharacters(in: .whitespacesAndNewlines)), parent > 1,
                ids.insert(parent).inserted
            else { break }
            pid = parent
        }
        return ids
    }
}

public struct ConditionProbe: Sendable {
    public typealias Sample = @Sendable (String) async throws -> ConditionSample
    public let sample: Sample
    public init(sample: @escaping Sample) { self.sample = sample }
    public static var synthetic: Self {
        Self { phase in
            ConditionSample(
                at: ISO8601DateFormatter().string(from: Date()), phase: phase,
                cpu: 0, gpu: 0, compositing: 0, note: "Synthetic stub telemetry; never a measurement.")
        }
    }
    public static var live: Self {
        Self { phase in try await Task.detached { try sampleMachine(phase: phase) }.value }
    }
    public func requireIdle(refuseContended: Bool = true) async throws -> [ConditionSample] {
        var samples: [ConditionSample] = []
        for _ in 0..<3 {
            let value = try await sample("idle"); samples.append(value)
            if !value.valid && refuseContended { throw BenchmarkError.conditions(samples) }
        }
        return samples
    }

    private struct GPUClient { var name: String; var time: Double }
    private static func gpuClients(_ text: String) -> [Int32: GPUClient] {
        var result: [Int32: GPUClient] = [:]
        for block in text.components(separatedBy: "+-o ") {
            guard let creator = capture(block, #""IOUserClientCreator"\s*=\s*"pid (\d+), ([^"]+)""#),
                creator.count == 2, let pid = Int32(creator[0]),
                let time = capture(block, #""accumulatedGPUTime"\s*=\s*(\d+)"#)?.first.flatMap(Double.init)
            else { continue }
            let previous = result[pid]?.time ?? 0
            result[pid] = GPUClient(name: creator[1], time: previous + time)
        }
        return result
    }
    private static func capture(_ text: String, _ expression: String) -> [String]? {
        guard let regex = try? NSRegularExpression(pattern: expression),
            let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
        else { return nil }
        return (1..<match.numberOfRanges).compactMap { Range(match.range(at: $0), in: text).map { String(text[$0]) } }
    }
    private static func sampleMachine(phase: String) throws -> ConditionSample {
        let processList = try BenchCommand.output("/bin/ps", ["-axo", "pid=,ppid="])
        var owned: Set<Int32> = [getpid()]
        let pairs = processList.split(separator: "\n").compactMap { line -> (Int32, Int32)? in
            let parts = line.split(whereSeparator: \.isWhitespace)
            guard parts.count == 2, let pid = Int32(parts[0]), let parent = Int32(parts[1]) else { return nil }
            return (pid, parent)
        }
        for _ in 0..<32 { let before = owned.count; for (pid, parent) in pairs where owned.contains(parent) { owned.insert(pid) }; if owned.count == before { break } }
        let a = gpuClients(try BenchCommand.output("/usr/sbin/ioreg", ["-r", "-c", "AGXDeviceUserClient", "-l"]))
        let start = Date()
        let top = try BenchCommand.output("/usr/bin/top", ["-l", "2", "-s", "3", "-stats", "pid,cpu,command", "-n", "10000"], timeout: 8)
        let b = gpuClients(try BenchCommand.output("/usr/sbin/ioreg", ["-r", "-c", "AGXDeviceUserClient", "-l"]))
        let interval = Date().timeIntervalSince(start)
        let device = try BenchCommand.output("/usr/sbin/ioreg", ["-r", "-d", "1", "-c", "IOAccelerator"])
        let utilization = capture(device, #""Device Utilization %"\s*=\s*(\d+)"#)?.first.flatMap(Double.init)
        var comp = 0.0, other = 0.0
        for (pid, entry) in b {
            let share = max(0, entry.time - (a[pid]?.time ?? entry.time)) / (interval * 1e9) * 100
            if ["WindowServer", "ghostty"].contains(where: { entry.name.hasPrefix($0) }) { comp += share } else if !owned.contains(pid) { other += share }
        }
        var cpu = 0.0, count = 0
        for line in (top.components(separatedBy: "PID").last ?? "").split(separator: "\n") {
            let parts = line.split(maxSplits: 2, whereSeparator: \.isWhitespace)
            guard parts.count == 3, let pid = Int32(parts[0]), let usage = Double(parts[1]) else { continue }
            count += 1
            if !owned.contains(pid) && !parts[2].hasPrefix("top") { cpu += usage }
        }
        // Do not subtract interval worker time from an instantaneous device reading: the two time bases differ.
        let gpu = utilization.map { phase == "run" ? other : max(0, $0 - comp) }
        return ConditionSample(
            at: ISO8601DateFormatter().string(from: Date()), phase: phase,
            cpu: count > 0 ? cpu : nil, gpu: gpu, compositing: b.isEmpty ? nil : comp,
            note: "top interval CPU; idle=device GPU minus compositor; run=non-owned client GPU-time delta; 3 s sample.")
    }
}
