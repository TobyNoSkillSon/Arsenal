import CryptoKit
import Darwin
import Foundation
import Metal
import VireoCore

/// Fast-path gate for custom kernels (Vella FastPathGate pattern, family rule): an optimized path serves
/// only after a self-test on this machine agreed with stock MLX.
///
/// - The self-test runs in a child process (`vireo-helper --kernel-self-test <engine> <weights>`) with a
///   timeout, so a faulting or hanging kernel cannot take the helper down.
/// - The verdict is persisted in `<support>/kernel-gate.json`, keyed by engine + kernel version + weights
///   fingerprint (SHA-256) + GPU + macOS build + app build + the model-path configuration (`FastPath.identity`: every
///   set switch that changes what a model computes). Any change retests; a failure is sticky for that key (delete
///   the file to retest). No switch serves an optimized path without a verdict.
/// - Engines add a runtime fallback to stock on non-finite output and report `engineLabel`/`engineReason`.
public enum KernelGate {
    public struct Verdict: Codable, Sendable {
        public var passed: Bool
        public var detail: String
        public var date: String
        public var seconds: Double
        public var key: [String: String]
    }

    static let childFlag = "--kernel-self-test"
    /// Engine key → self-test run inside the child. Returns a one-line detail; throws on failure.
    static let selfTests: [String: (URL) throws -> String] = [
        MLXSpeech.qwenRopeKernelKey: MLXSpeech.qwenRopeChildSelfTest,
        MLXSpeech.qwenSamplerKernelKey: MLXSpeech.qwenSamplerChildSelfTest,
        MLXSpeech.breezeFusedKernelKey: MLXSpeech.breezeFusedChildSelfTest,
        MLXSpeech.smallMKernelKey: MLXSpeech.smallMChildSelfTest,
    ]

    /// First thing in vireo-helper's main: runs a requested self-test and returns the exit code, else nil.
    public static func runChildIfRequested(_ arguments: [String] = CommandLine.arguments) -> Int32? {
        guard arguments.count >= 4, arguments[1] == childFlag else { return nil }
        RuntimeCaches.useBundledMetallib()
        guard let test = selfTests[arguments[2]] else { print("no self-test for engine \(arguments[2])"); return 2 }
        do { print(try test(URL(fileURLWithPath: arguments[3]))); return 0 } catch { print("\(error.localizedDescription)"); return 1 }
    }

    /// The persisted verdict for this machine and build, or the result of a new child self-test.
    public static func verdict(engine: String, kernelVersion: String, weights: URL, configuration: FastPath = .current,
                               timeout: TimeInterval = 90) -> Verdict {
        let key = keyComponents(engine: engine, kernelVersion: kernelVersion, weights: weights, configuration: configuration)
        let id = digest(key.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: "\n"))
        if let saved = load()[id] { return saved }
        let start = Date()
        let (passed, detail) = runChild(engine: engine, weights: weights, timeout: timeout, configuration: configuration)
        let verdict = Verdict(passed: passed, detail: detail, date: ISO8601DateFormatter().string(from: Date()),
                              seconds: (Date().timeIntervalSince(start) * 100).rounded() / 100, key: key)
        store(id, verdict)
        return verdict
    }

    /// Graph-level fast paths built from stock MLX ops (no custom Metal kernels, so no fault can take the helper
    /// down): the self-test runs in this process against the already-loaded model; persistence, key and stickiness
    /// are the same as `verdict`. `test` returns a one-line detail or throws.
    public static func inProcessVerdict(engine: String, kernelVersion: String, weights: URL, configuration: FastPath = .current,
                                        test: () throws -> String) -> Verdict {
        let key = keyComponents(engine: engine, kernelVersion: kernelVersion, weights: weights, configuration: configuration)
        let id = digest(key.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: "\n"))
        if let saved = load()[id] { return saved }
        let start = Date()
        let (passed, detail): (Bool, String)
        do { (passed, detail) = (true, try test()) } catch { (passed, detail) = (false, "\(error.localizedDescription)") }
        let verdict = Verdict(passed: passed, detail: detail, date: ISO8601DateFormatter().string(from: Date()),
                              seconds: (Date().timeIntervalSince(start) * 100).rounded() / 100, key: key)
        store(id, verdict)
        return verdict
    }

    /// Runtime fallback: an optimized path produced bad output after passing; make stock sticky for this key.
    public static func recordRuntimeFailure(engine: String, kernelVersion: String, weights: URL, configuration: FastPath = .current,
                                            detail: String) {
        let key = keyComponents(engine: engine, kernelVersion: kernelVersion, weights: weights, configuration: configuration)
        let id = digest(key.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: "\n"))
        store(id, Verdict(passed: false, detail: "runtime: " + detail, date: ISO8601DateFormatter().string(from: Date()), seconds: 0, key: key))
    }

    // MARK: Key

    static func keyComponents(engine: String, kernelVersion: String, weights: URL, configuration: FastPath = .current) -> [String: String] {
        let device = MTLCreateSystemDefaultDevice()
        var gpu = device?.name ?? "none"
        if let device, #available(macOS 14.0, *) { gpu += " / " + device.architecture.name }
        let info = Bundle.main.infoDictionary ?? [:]
        var build = [info["CFBundleShortVersionString"], info["CFBundleVersion"]].compactMap { $0 as? String }.joined(separator: "+")
        // A bare helper binary (lab builds) has no Info.plist: its size and mtime stand in for the build.
        if let exe = Bundle.main.executableURL, let a = try? FileManager.default.attributesOfItem(atPath: exe.path) {
            build += " exe \((a[.size] as? NSNumber)?.intValue ?? 0)@\(Int((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0))"
        }
        return ["engine": engine, "kernel": kernelVersion, "weights": fingerprint(weights), "gpu": gpu,
                "macos": sysctl("kern.osversion"), "build": build, "env": configuration.identity]
    }

    static func sysctl(_ name: String) -> String {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return "?" }
        var buffer = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &buffer, &size, nil, 0) == 0 else { return "?" }
        return String(cString: buffer)
    }


    /// A file's SHA-256, or for a model directory the content addresses of its files: Hugging Face snapshot files
    /// are symlinks to `blobs/<sha256>` (LFS) or `blobs/<git sha1>`, so the link targets name the content without
    /// reading gigabytes; a regular file inside contributes its name, size and mtime.
    static func fingerprint(_ url: URL) -> String {
        var isDirectory: ObjCBool = false
        guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDirectory), isDirectory.boolValue else { return fileDigest(url) }
        guard let walker = FileManager.default.enumerator(atPath: url.path) else { return "unreadable" }
        var entries: [String] = []
        for case let relative as String in walker {
            let path = url.appendingPathComponent(relative).path
            if let target = try? FileManager.default.destinationOfSymbolicLink(atPath: path) {
                entries.append("\(relative) -> \((target as NSString).lastPathComponent)")
            } else if let a = try? FileManager.default.attributesOfItem(atPath: path), (a[.type] as? FileAttributeType) == .typeRegular {
                entries.append("\(relative) \((a[.size] as? NSNumber)?.intValue ?? 0)@\(Int((a[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0))")
            }
        }
        return "dir:" + digest(entries.sorted().joined(separator: "\n"))
    }

    static func digest(_ text: String) -> String { SHA256.hash(data: Data(text.utf8)).map { String(format: "%02x", $0) }.joined() }

    /// SHA-256 of the weights in 8 MB reads; each chunk in its own autorelease pool (skill trap: autoreleased
    /// `Data` on a task thread otherwise lives until the task ends).
    static func fileDigest(_ url: URL) -> String {
        guard let handle = try? FileHandle(forReadingFrom: url) else { return "unreadable" }
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            let more: Bool = autoreleasepool {
                guard let chunk = try? handle.read(upToCount: 8 << 20), !chunk.isEmpty else { return false }
                hasher.update(data: chunk); return true
            }
            if !more { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    // MARK: Child

    static func runChild(engine: String, weights: URL, timeout: TimeInterval, configuration: FastPath = .current) -> (Bool, String) {
        guard let exe = Bundle.main.executableURL else { return (false, "no helper executable for the self-test child") }
        let process = Process()
        process.executableURL = exe
        process.arguments = [childFlag, engine, weights.path]
        // The child runs under this load's configuration (the cell's switches over the environment), like the load.
        var environment = ProcessInfo.processInfo.environment.filter { !DiagnosticSwitches.affectsModelPath($0.key) }
        environment.merge(configuration.switches) { _, new in new }
        process.environment = environment
        let pipe = Pipe()
        process.standardOutput = pipe
        process.standardError = FileHandle.nullDevice
        process.standardInput = FileHandle.nullDevice
        let done = DispatchSemaphore(value: 0)
        process.terminationHandler = { _ in done.signal() }
        do { try process.run() } catch { return (false, "self-test child did not start: \(error.localizedDescription)") }
        // Drain stdout concurrently so a chatty child cannot block on a full pipe.
        var output = Data()
        let reader = DispatchQueue(label: "vireo.kernel-gate.reader")
        let readDone = DispatchSemaphore(value: 0)
        reader.async { output = pipe.fileHandleForReading.readDataToEndOfFile(); readDone.signal() }
        if done.wait(timeout: .now() + timeout) == .timedOut {
            process.terminate()
            if done.wait(timeout: .now() + 2) == .timedOut { kill(process.processIdentifier, SIGKILL); _ = done.wait(timeout: .now() + 2) }
            _ = readDone.wait(timeout: .now() + 2)
            return (false, "self-test timed out after \(Int(timeout)) s")
        }
        _ = readDone.wait(timeout: .now() + 5)
        let last = String(decoding: output, as: UTF8.self).split(separator: "\n").last.map(String.init) ?? ""
        if process.terminationReason == .uncaughtSignal { return (false, "self-test crashed (signal \(process.terminationStatus))") }
        return (process.terminationStatus == 0, last.isEmpty ? "exit \(process.terminationStatus)" : last)
    }

    // MARK: Store

    private static let lock = NSLock()
    static var file: URL { VireoFiles.support.appendingPathComponent("kernel-gate.json") }

    static func load() -> [String: Verdict] {
        lock.withLock { (try? Data(contentsOf: file)).flatMap { try? JSONDecoder().decode([String: Verdict].self, from: $0) } ?? [:] }
    }

    static func store(_ id: String, _ verdict: Verdict) {
        lock.withLock {
            var all = (try? Data(contentsOf: file)).flatMap { try? JSONDecoder().decode([String: Verdict].self, from: $0) } ?? [:]
            all[id] = verdict
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            try? FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
            if let data = try? encoder.encode(all) { try? data.write(to: file, options: .atomic) }
        }
    }
}
