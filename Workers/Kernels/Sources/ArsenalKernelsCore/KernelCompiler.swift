// SPDX-License-Identifier: MIT
import Foundation
import Metal

/// This Mac's identity for kernel caches and verdicts: a compiled library or a verdict is reused only on the same
/// OS build and GPU.
public enum KernelHost {
    /// `kern.osversion`, e.g. "25G5034".
    public static let osBuild: String = sysctl("kern.osversion") ?? "?"
    /// "Apple M5 Max" → "m5" (the manifest's chip family); "unknown" when not an Apple M chip.
    public static let chipFamily: String = family(brand: sysctl("machdep.cpu.brand_string") ?? "")
    public static func family(brand: String) -> String {
        let words = brand.lowercased().split(whereSeparator: { $0.isWhitespace })
        return words.first(where: { $0.count >= 2 && $0.first == "m" && $0.dropFirst().allSatisfy(\.isNumber) }).map(String.init) ?? "unknown"
    }
    /// Metal device name + GPU architecture (e.g. "Apple M5 Max/applegpu_g17s").
    public static let gpu: String = {
        guard let device = MTLCreateSystemDefaultDevice() else { return "no-gpu" }
        return device.name + "/" + device.architecture.name
    }()
    /// The running executable (name:size:mtime): the MLX and runner code that produced a gate verdict. Every rebuild
    /// requalifies (the Verdict/Vella rule); bare development binaries have no other identity.
    public static let binary: String = {
        let exe = URL(fileURLWithPath: CommandLine.arguments.first ?? "").resolvingSymlinksInPath()
        let attrs = try? FileManager.default.attributesOfItem(atPath: exe.path)
        let size = (attrs?[.size] as? NSNumber)?.intValue ?? 0
        let mtime = Int((attrs?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)
        return "\(exe.lastPathComponent):\(size):\(mtime)"
    }()
    /// Part of every key: OS build + GPU.
    public static var key: String { osBuild + "|" + gpu }

    static func sysctl(_ name: String) -> String? {
        var size = 0
        guard sysctlbyname(name, nil, &size, nil, 0) == 0, size > 0 else { return nil }
        var bytes = [CChar](repeating: 0, count: size)
        guard sysctlbyname(name, &bytes, &size, nil, 0) == 0 else { return nil }
        return String(decoding: bytes.prefix { $0 != 0 }.map { UInt8(bitPattern: $0) }, as: UTF8.self)
    }
}

/// Runtime Metal compilation (no toolchain on the user's Mac): `MTLDevice.makeLibrary(source:)` with MLX's options,
/// libraries cached in process by (source digest, options); compile verdicts persisted by (source digest, OS build, GPU)
/// so a source that failed once is not recompiled at every load and one that passed skips the preflight.
public final class KernelCompiler: @unchecked Sendable {
    public struct Failure: Error, CustomStringConvertible, Sendable {
        public let message: String
        public var description: String { message }
    }

    public static let shared = KernelCompiler()

    public let device: MTLDevice?
    private let lock = NSLock()
    private var libraries: [String: MTLLibrary] = [:]
    private var failures: [String: String] = [:]
    /// Compiles performed by this instance (tests and status: cache hits do not count).
    public private(set) var compileCount = 0

    public init(device: MTLDevice? = MTLCreateSystemDefaultDevice()) { self.device = device }

    /// MLX's options: MathMode::Safe and the Metal language version MLX picks for this OS.
    public static func mlxOptions() -> MTLCompileOptions {
        let options = MTLCompileOptions()
        options.mathMode = .safe
        if #available(macOS 27, *) {
            options.languageVersion = MTLLanguageVersion(rawValue: (4 << 16) + 1) ?? .version4_0
        } else {
            options.languageVersion = .version4_0
        }
        return options
    }

    static func optionsKey(_ o: MTLCompileOptions) -> String { "lang\(o.languageVersion.rawValue).math\(o.mathMode.rawValue)" }

    /// The compiled library for `source`, from the in-process cache or a fresh compile. Throws the compiler's message.
    public func library(source: String, options: MTLCompileOptions = KernelCompiler.mlxOptions()) throws -> MTLLibrary {
        guard let device else { throw Failure(message: "No Metal device.") }
        let key = KernelDigest.sha256(source) + "|" + Self.optionsKey(options)
        lock.lock(); defer { lock.unlock() }
        if let library = libraries[key] { return library }
        if let message = failures[key] { throw Failure(message: message) }
        compileCount += 1
        do {
            let library = try device.makeLibrary(source: source, options: options)
            libraries[key] = library
            return library
        } catch {
            let message = Self.firstError((error as NSError).localizedDescription)
            failures[key] = message
            throw Failure(message: message)
        }
    }

    /// Compiles `source` and builds the pipeline for `function`; checks the threadgroup size MLX would dispatch
    /// (MLX throws, i.e. aborts the process, when it exceeds the pipeline's maximum). nil = OK, else one sentence.
    public func check(
        source: String, function: String, threadsPerThreadgroup: Int,
        options: MTLCompileOptions = KernelCompiler.mlxOptions()
    ) -> String? {
        do {
            let library = try library(source: source, options: options)
            guard let fn = library.makeFunction(name: function) else { return "The library has no function \(function)." }
            guard let device else { return "No Metal device." }
            let pipeline = try device.makeComputePipelineState(function: fn)
            if threadsPerThreadgroup > pipeline.maxTotalThreadsPerThreadgroup {
                return "Threadgroup of \(threadsPerThreadgroup) threads exceeds this pipeline's maximum of \(pipeline.maxTotalThreadsPerThreadgroup)."
            }
            return nil
        } catch let failure as Failure {
            return "Metal compile failed: " + failure.message
        } catch {
            return "Metal pipeline failed: " + Self.firstError((error as NSError).localizedDescription)
        }
    }

    /// The first `error:` line of a compiler log (whole logs are pages long), trimmed to one line.
    static func firstError(_ log: String) -> String {
        let lines = log.split(separator: "\n").map { $0.trimmingCharacters(in: .whitespaces) }
        let line = lines.first(where: { $0.contains("error:") }) ?? lines.first ?? "unknown error"
        return String(line.prefix(240))
    }
}
