// SPDX-License-Identifier: MIT
// Vella's FastPathGate (Projects/Vella/Worker/Sources/MLXAudioSTT/Gate/FastPathGate.swift), reduced to its one-time
// token check. Vella runs the check in a child process with a 45 s deadline and a retry budget; here it runs inside the
// worker's Load, on one short clip.
import CryptoKit
import Foundation
import Metal
import MLX
import os
import VellaSTT

/// Vella enables its optimized kernels only after they reproduce stock MLX on this Mac. The first Load per model files,
/// GPU (family and name), macOS build and gate revision transcribes a short bundled clip twice, stock then optimized
/// (`qualificationTokens`), and keeps the optimized path only when the token IDs are identical, non-empty and finite.
/// The verdict is saved beside Arsenal's support data (`<ARSENAL_SUPPORT_DIR>/FastPath/vella-asr/<key>.json`) and a failed
/// verdict is sticky for that key, as in Vella; a new GPU or macOS build tests again. Without a support directory (lab
/// runs, tests) the verdict lasts for the worker process.
struct VellaFastPathGate {
    /// Changes whenever the optimized components, their defaults or the clip change, so an old "fast" is not reused.
    /// 2: Qwen's decoder finiteness probe sums in f32 (its FP16 sum overflowed, failing every FP16 Load); saved
    /// verdicts of 1 are tested again.
    static let revision = "mobius-vella-gate-2" // legacy-compat: keys saved verdicts; renaming would re-run the gate
    static let clipName = "clip-c"
    struct Host: Equatable {
        var gpuFamily: String
        var gpuName: String
        var osBuild: String
        static var current: Host {
            let device = MTLCreateSystemDefaultDevice()
            let family = device?.supportsFamily(.apple9) == true ? "apple9" : device?.supportsFamily(.apple8) == true ? "apple8" : "unsupported"
            var length = 0
            sysctlbyname("kern.osversion", nil, &length, nil, 0)
            var build = [CChar](repeating: 0, count: max(length, 1))
            sysctlbyname("kern.osversion", &build, &length, nil, 0)
            return Host(gpuFamily: family, gpuName: device?.name ?? "unavailable", osBuild: String(cString: build))
        }
    }
    enum Verdict: Equatable {
        case fast
        case stock(String)
    }

    /// nil: verdicts last for this process only.
    let storage: URL?
    let host: Host
    /// 16 kHz mono samples of the self-test clip; nil when the clip is missing (the check cannot run: stock for this Load).
    let clip: () -> [Float]?

    init(environment: [String: String] = ProcessInfo.processInfo.environment, host: Host = .current, clip: @escaping () -> [Float]? = Self.bundledClip) {
        storage = environment["ARSENAL_SUPPORT_DIR"].flatMap { $0.hasPrefix("/") ? URL(fileURLWithPath: $0).appendingPathComponent("FastPath/vella-asr") : nil }
        self.host = host; self.clip = clip
    }
    init(storage: URL?, host: Host, clip: @escaping () -> [Float]?) { self.storage = storage; self.host = host; self.clip = clip }

    /// Everything a verdict depends on. The weights count by name, size and modification time: Arsenal's snapshots are
    /// immutable, and hashing gigabytes on every Load would cost more than the check.
    func key(model: String, precision: String, weights: URL, component: String, configuration: String) -> String {
        var digest = SHA256()
        func add(_ text: String) { digest.update(data: Data("\(text.utf8.count):\(text)|".utf8)) }
        for part in [Self.revision, model, precision, component, configuration, host.gpuFamily, host.gpuName, host.osBuild, weights.standardizedFileURL.path] { add(part) }
        let files = (try? FileManager.default.contentsOfDirectory(at: weights, includingPropertiesForKeys: [.fileSizeKey, .contentModificationDateKey])) ?? []
        for file in files.filter({ ["json", "safetensors"].contains($0.pathExtension) }).sorted(by: { $0.lastPathComponent < $1.lastPathComponent }) {
            let values = try? file.resourceValues(forKeys: [.fileSizeKey, .contentModificationDateKey])
            add("\(file.lastPathComponent):\(values?.fileSize ?? -1):\(values?.contentModificationDate?.timeIntervalSince1970 ?? 0)")
        }
        return digest.finalize().map { String(format: "%02x", $0) }.joined()
    }

    /// Verdicts of a worker without a support directory.
    private static let memory = OSAllocatedUnfairLock(initialState: [String: Verdict]())
    func saved(_ key: String) -> Verdict? {
        guard let storage else { return Self.memory.withLock { $0[key] } }
        guard let data = try? Data(contentsOf: storage.appendingPathComponent(key + ".json")) else { return nil }
        // Any status other than "fast", or an unreadable record, counts as a failed verdict, never as a pass.
        let record = (try? JSONSerialization.jsonObject(with: data)) as? [String: String]
        return record?["status"] == "fast" ? .fast : .stock(record?["reason"] ?? "The optimized path failed its self-test against stock MLX on this Mac.")
    }
    func save(_ verdict: Verdict, key: String, model: String) {
        guard let storage else { Self.memory.withLock { $0[key] = verdict }; return }
        var record = [
            "status": verdict == .fast ? "fast" : "stock", "model": model, "gpu_family": host.gpuFamily, "gpu_name": host.gpuName, "os_build": host.osBuild,
            "revision": Self.revision, "date": ISO8601DateFormatter().string(from: Date())
        ]
        if case .stock(let reason) = verdict { record["reason"] = reason }
        do {
            try FileManager.default.createDirectory(at: storage, withIntermediateDirectories: true)
            try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]).write(to: storage.appendingPathComponent(key + ".json"), options: .atomic)
        } catch {} // Unsaved: the check runs again next Load, never a fast path without a verdict.
    }

    /// The saved verdict, or the check now. Leaves the optimized path off; the caller enables it on `.fast`.
    func verdict(_ model: any FastPathCapable, component: String, key: String, name: String) -> Verdict {
        if let saved = saved(key) { return saved }
        guard let samples = clip(), !samples.isEmpty else {
            // Setup, not evidence against the kernels: stock for this Load, checked again next Load.
            return .stock("The optimized path's self-test clip is missing; using stock MLX.")
        }
        let verdict = Self.check(model, component: component, audio: MLXArray(samples))
        save(verdict, key: key, model: name)
        return verdict
    }

    static func check(_ model: any FastPathCapable, component: String, audio: MLXArray) -> Verdict {
        let failed = Verdict.stock("The optimized path failed its self-test against stock MLX on this Mac.")
        defer { _ = try? withError { model.configureFastPath(enabled: false, component: component) } }
        do {
            _ = try withError { model.configureFastPath(enabled: false, component: component) }
            let stock = try withError { model.qualificationTokens(audio: audio) }
            guard try withError({ model.configureFastPath(enabled: true, component: component) }) else {
                return .stock("The optimized path does not support this checkpoint.")
            }
            let fast = try withError { model.qualificationTokens(audio: audio) }
            return !stock.isEmpty && fast == stock && model.fastPathFinite ? .fast : failed
        } catch {
            return failed
        }
    }

    /// The bundled clip (`Resources/selftest/clip-c.wav`), found explicitly rather than through `Bundle.module`, whose
    /// accessor traps when the bundle is missing: beside the image this code is linked into (the shipped worker in
    /// Helpers, or a test bundle inside the build directory) or the main bundle.
    static func bundledClip() -> [Float]? {
        let name = "Audio_AudioWorker.bundle"
        var roots = [Bundle.main.bundleURL, Bundle.main.executableURL?.deletingLastPathComponent(), Bundle.main.resourceURL].compactMap { $0 }
        var info = Dl_info()
        if dladdr(#dsohandle, &info) != 0, let path = info.dli_fname {
            var directory = URL(fileURLWithPath: String(cString: path)).deletingLastPathComponent()
            for _ in 0..<4 { roots.append(directory); directory.deleteLastPathComponent() }
        }
        for root in roots {
            for folder in [root.appendingPathComponent(name), root.appendingPathComponent(name).appendingPathComponent("Contents/Resources")] {
                let url = folder.appendingPathComponent("selftest/\(clipName).wav")
                if let data = try? Data(contentsOf: url) { return pcm16(wav: data) }
            }
        }
        return nil
    }

    /// Samples of a 16 kHz mono 16-bit PCM WAV file; nil for any other layout.
    static func pcm16(wav data: Data) -> [Float]? {
        let bytes = [UInt8](data)
        guard bytes.count >= 12, bytes[0..<4] == [0x52, 0x49, 0x46, 0x46], bytes[8..<12] == [0x57, 0x41, 0x56, 0x45] else { return nil }
        func u16(_ i: Int) -> Int { Int(bytes[i]) | Int(bytes[i + 1]) << 8 }
        func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
        var offset = 12, format: (channels: Int, rate: Int, bits: Int, pcm: Bool)?
        while offset + 8 <= bytes.count {
            let id = String(decoding: bytes[offset..<offset + 4], as: UTF8.self), size = u32(offset + 4), body = offset + 8
            guard body + size <= bytes.count else { return nil }
            // Plain PCM, or WAVE_FORMAT_EXTENSIBLE whose sub-format is PCM (how the LibriSpeech conversions are written).
            if id == "fmt ", size >= 16 { format = (u16(body + 2), u32(body + 4), u16(body + 14), u16(body) == 1 || u16(body) == 0xFFFE && size >= 26 && u16(body + 24) == 1) }
            if id == "data" {
                guard let format, format.pcm, format.channels == 1, format.rate == 16000, format.bits == 16 else { return nil }
                return stride(from: body, to: body + size - 1, by: 2).map { Float(Int16(bitPattern: UInt16(u16($0)))) / 32768 }
            }
            offset = body + size + size % 2
        }
        return nil
    }
}
