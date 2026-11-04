// SPDX-License-Identifier: MIT
import Foundation
import CryptoKit
import Darwin
import MLX
import MLXLLM
import MLXLMCommon
import ChatContract
import ArsenalWire

public struct ChatOutput: Codable, Sendable {
    public let text: String
    public let reasoning: String?
    public let finish_reason: String
    public let usage: Usage
    public let ttft_s: Double
    public let tokens_per_s: Double
    public let timing: RequestTiming
    public let _arsenal_first_token_at: Double?
    public let cached_tokens: Int
    public let generated_tokens: Int
    public struct Usage: Codable, Sendable { public let prompt_tokens: Int; public let completion_tokens: Int }
}
public struct OutputFile: Codable, Sendable { public let path: String; public let bytes: Int; public let sha256: String }
public struct ChatMemory: Codable, Sendable {
    public let footprint_mb: Double
    public let mlx_active_mb: Double
    public let mlx_cache_mb: Double
    public let peak_mb: Double
    public let mlx_peak_mb: Double
    /// MLX active memory right after the load (the weights); absent while nothing is loaded.
    public let mlx_weights_mb: Double?
    /// Set when a load finishes, cleared at unload.
    static let weights = WeightsMark()
    final class WeightsMark: @unchecked Sendable {
        private let lock = NSLock()
        private var bytes: Int?
        var mb: Double? { lock.withLock { bytes.map { Double($0) / 1e6 } } }
        func set(_ value: Int?) { lock.withLock { bytes = value } }
    }
    public static func snapshot() -> ChatMemory {
        var info = task_vm_info_data_t()
        var count = mach_msg_type_number_t(MemoryLayout<task_vm_info_data_t>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) { pointer in
            pointer.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                task_info(mach_task_self_, task_flavor_t(TASK_VM_INFO), $0, &count)
            }
        }
        return .init(
            footprint_mb: result == KERN_SUCCESS ? Double(info.phys_footprint) / 1e6 : 0,
            mlx_active_mb: Double(Memory.activeMemory) / 1e6,
            mlx_cache_mb: Double(Memory.cacheMemory) / 1e6,
            peak_mb: result == KERN_SUCCESS ? Double(info.ledger_phys_footprint_peak) / 1e6 : 0,
            mlx_peak_mb: Double(Memory.peakMemory) / 1e6, mlx_weights_mb: weights.mb)
    }
}

public actor ChatEngine {
    private var context: ModelContext?
    private var contextLimit = 8192
    private var loadedDtype: String?
    public init() {}
    public func resetTimingPeak() -> Double { Memory.peakMemory = 0; return ProcessInfo.processInfo.systemUptime }
    public func timingPeakMB() -> Double { Double(Memory.peakMemory) / 1e6 }
    public func load(directory: String, dtype: String, contextLimit: Int = 8192) async throws -> Double {
        guard ["bf16", "int8", "int4"].contains(dtype) else { throw ChatFailure("unsupported", "Chat supports bf16 and locally derived int8/int4 weights.") }
        guard context == nil else { throw ChatFailure("already_loaded", "Unload the current chat model before loading another snapshot.") }
        let url = URL(fileURLWithPath: directory, isDirectory: true)
        let config = try Self.configuration(url)
        let approved = try QwenConfiguration(config)
        let quant = config["quantization"] as? [String: Any]
        if dtype == "bf16" {
            guard approved.bf16Source else { throw ChatFailure("dtype_mismatch", "The bf16 snapshot must be an unquantized bfloat16 checkpoint.") }
        } else {
            let bits = dtype == "int8" ? 8 : 4
            guard quant?["bits"] as? Int == bits, quant?["group_size"] as? Int == 64,
                (quant?["mode"] as? String ?? "affine") == "affine"
            else { throw ChatFailure("dtype_mismatch", "Load a locally derived MLX affine group-64 snapshot matching the requested precision.") }
        }
        guard contextLimit >= 256, contextLimit <= approved.maximumContext else {
            throw ChatFailure("context_limit must be between 256 and \(approved.maximumContext) and fit the model's context.")
        }
        Memory.cacheLimit = 256 * 1024 * 1024
        let start = ProcessInfo.processInfo.systemUptime
        Memory.peakMemory = 0
        context = try await LLMModelFactory.shared.load(from: url, using: LocalTokenizerLoader())
        self.contextLimit = contextLimit
        loadedDtype = dtype
        ChatMemory.weights.set(Memory.activeMemory)
        return ProcessInfo.processInfo.systemUptime - start
    }
    public func unload() {
        context = nil; loadedDtype = nil; ChatMemory.weights.set(nil); Memory.clearCache()
    }
    public func status() -> (loaded: Bool, dtype: String?, context: Int) { (context != nil, loadedDtype, contextLimit) }
    public func count(text: String?, messages: [ChatMessage]?, thinking: Bool = false) throws -> Int {
        guard let context else { throw ChatFailure("not_loaded", "Load a chat model before counting tokens.") }
        if let text { return context.tokenizer.encode(text: text, addSpecialTokens: false).count }
        return try template(messages ?? [], thinking: thinking, context: context).count
    }
    private func template(_ messages: [ChatMessage], thinking: Bool, context: ModelContext) throws -> [Int] {
        try context.tokenizer.applyChatTemplate(
            messages: messages.map { ["role": $0.role, "content": $0.content] as [String: any Sendable] }, tools: nil, additionalContext: ["enable_thinking": thinking])
    }
    public func chat(
        _ request: ChatRequest, cancellation: Cancellation, accepted: Double? = nil,
        progress: @Sendable (JSONObject) -> Void = { _ in }, emit: @Sendable (ChatDelta) -> Void
    ) throws -> ChatOutput {
        guard let context else { throw ChatFailure("not_loaded", "Load a chat model before generating text.") }
        let start = ProcessInfo.processInfo.systemUptime
        Memory.peakMemory = 0
        var telemetry = TokenTiming(accepted: accepted ?? start, scheduled: start)
        let ids = try template(request.messages, thinking: request.thinking, context: context)
        guard ids.count + request.maxTokens <= contextLimit else {
            throw ChatFailure("context_length_exceeded", "Prompt plus max_tokens exceeds the loaded \(contextLimit)-token context; shorten the prompt or output limit.")
        }
        if cancellation.isCancelled { throw ChatFailure("cancelled", "Chat was cancelled before generation began.") }
        let input = LMInput(tokens: MLXArray(ids))
        let parameters = GenerateParameters(
            maxTokens: request.maxTokens, maxKVSize: contextLimit,
            temperature: request.temperature, topP: request.topP, seed: request.seed)
        var filter = StopFilter(stops: request.stops)
        var splitter = ReasoningFilter(thinking: request.thinking)
        var detokenizer = NaiveStreamingDetokenizer(tokenizer: context.tokenizer)
        var text = ""
        var reasoning = ""
        nonisolated func receive(_ deltas: [ChatDelta]) {
            for delta in deltas {
                let safe: String
                if delta.event == .reasoning { safe = delta.text; reasoning += safe } else { safe = filter.append(delta.text); text += safe }
                if request.stream && !safe.isEmpty { emit(.init(event: delta.event, text: safe)) }
            }
        }
        var first: Double?
        var last: Double?
        var lastProgress: Double?
        telemetry.promptTokens = ids.count
        telemetry.prefillStarted = ProcessInfo.processInfo.systemUptime
        progress(telemetry.progress(id: .null, phase: "prefill", peakMB: Double(Memory.peakMemory) / 1e6))
        let result = try MLXLMCommon.generate(input: input, parameters: parameters, context: context) { tokens in
            let now = ProcessInfo.processInfo.systemUptime
            if first == nil { first = now }
            last = now
            telemetry.token(at: now)
            if lastProgress == nil || now - lastProgress! >= 0.1 {
                progress(telemetry.progress(id: .null, phase: "decode", peakMB: Double(Memory.peakMemory) / 1e6)); lastProgress = now
            }
            if cancellation.isCancelled { return .stop }
            detokenizer.append(token: tokens.last!)
            if let delta = detokenizer.next() {
                receive(splitter.append(delta))
            }
            return filter.stopped ? .stop : .more
        }
        receive(splitter.finish())
        let tail = filter.finish(); text += tail
        if request.stream && !tail.isEmpty { emit(.init(event: .token, text: tail)) }
        let reason = cancellation.isCancelled ? "cancelled" : (filter.stopped ? "stop" : (result.generationTokenCount >= request.maxTokens ? "length" : "stop"))
        let generated = result.generationTokenCount
        let elapsed = (last ?? start) - (first ?? start)
        return ChatOutput(
            text: text, reasoning: request.thinking ? reasoning : nil, finish_reason: reason,
            usage: .init(prompt_tokens: ids.count, completion_tokens: generated),
            ttft_s: max(0, (first ?? ProcessInfo.processInfo.systemUptime) - start),
            tokens_per_s: elapsed > 0 && generated > 1 ? Double(generated - 1) / elapsed : 0,
            timing: telemetry.timing(peakMB: Double(Memory.peakMemory) / 1e6), _arsenal_first_token_at: telemetry.first, cached_tokens: 0, generated_tokens: generated)
    }
    public func convert(source: String, destination: String, dtype: String, groupSize: Int, recipe: String) async throws -> [OutputFile] {
        guard context == nil else { throw ChatFailure("busy", "Convert in a worker without a loaded model.") }
        guard ["int8", "int4"].contains(dtype), groupSize == 64, recipe == "mlx-affine-g64" else {
            throw ChatFailure("unsupported", "Conversion requires int8 or int4 with MLX affine group 64 and recipe mlx-affine-g64.")
        }
        let src = URL(fileURLWithPath: source, isDirectory: true)
        let dst = URL(fileURLWithPath: destination, isDirectory: true)
        let config = try Self.configuration(src)
        guard try QwenConfiguration(config).bf16Source else { throw ChatFailure("invalid_source", "Derive chat precisions only from the original unquantized bf16 checkpoint.") }
        let base = try await LLMTypeRegistry.shared.createModel(
            configuration: Data(contentsOf: src.appendingPathComponent("config.json")), modelType: config["model_type"] as? String ?? "")
        let exists = FileManager.default.fileExists(atPath: dst.path)
        let entries = exists ? try FileManager.default.contentsOfDirectory(atPath: dst.path) : []
        if !entries.isEmpty {
            throw ChatFailure("invalid_destination", "Conversion destination must be empty; existing files are never replaced.")
        }
        _ = try MLXLMCommon.convert(
            modelDirectory: src, model: base, to: dst,
            options: .init(bits: dtype == "int8" ? 8 : 4, groupSize: 64, mode: .affine, overwriteExistingOutput: exists),
            progressHandler: { progress in
                // Blob-store snapshots are immutable; copied config permissions must allow the quantization block update.
                if progress.stage == .updatingConfiguration {
                    try? FileManager.default.setAttributes([.posixPermissions: 0o644], ofItemAtPath: dst.appendingPathComponent("config.json").path)
                }
            })
        // The store owns provenance, so conversion must not copy its old source snapshot manifests.
        for name in StoredNames.reserved {
            let path = dst.appendingPathComponent(name)
            if FileManager.default.fileExists(atPath: path.path) { try FileManager.default.removeItem(at: path) }
        }
        Memory.clearCache()
        return try FileManager.default.contentsOfDirectory(at: dst, includingPropertiesForKeys: [.isRegularFileKey]).sorted { $0.path < $1.path }.map { url in
            guard try url.resourceValues(forKeys: [.isRegularFileKey]).isRegularFile == true else {
                throw ChatFailure("conversion_failed", "Conversion output contains an unexpected directory.")
            }
            let file = try FileHandle(forReadingFrom: url); defer { try? file.close() }
            var hash = SHA256(); var bytes = 0
            while let data = try file.read(upToCount: 4 * 1024 * 1024), !data.isEmpty { hash.update(data: data); bytes += data.count }
            return OutputFile(path: url.lastPathComponent, bytes: bytes, sha256: hash.finalize().map { String(format: "%02x", $0) }.joined())
        }
    }
    private static func configuration(_ url: URL) throws -> [String: Any] {
        guard let value = try JSONSerialization.jsonObject(with: Data(contentsOf: url.appendingPathComponent("config.json"))) as? [String: Any] else {
            throw ChatFailure("invalid_model", "config.json must contain an object.")
        }
        return value
    }
}

// The sole added model operation: teacher-forced logits from the ordinary loaded model.
extension ChatEngine {
    public func measurementLogits(prefix: String, continuation: String, steps: Int) throws -> [[Double]] {
        guard let context, steps > 0, steps <= 32 else { throw ChatFailure("measurement", "Load chat and request 1...32 fixed continuation tokens.") }
        let prefixIDs = context.tokenizer.encode(text: prefix, addSpecialTokens: true)
        let continuationIDs = Array(context.tokenizer.encode(text: continuation, addSpecialTokens: false).prefix(steps))
        return try ChatMeasurementProbe.capture(model: context.model, prefix: prefixIDs, continuation: continuationIDs)
    }
}
