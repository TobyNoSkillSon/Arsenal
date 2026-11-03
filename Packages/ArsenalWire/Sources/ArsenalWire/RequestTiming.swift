// SPDX-License-Identifier: MIT
import Foundation

/// Seconds except max_itl_ms and peak_mb (decimal MB). Optional means unknown, including an empty decode.
public struct RequestTiming: Codable, Sendable, Equatable {
    public var queue_s: Double?
    public var load_s: Double?
    public var ttft_s: Double?
    public var prefill_tps: Double?
    public var decode_tps: Double?
    public var max_itl_ms: Double?
    public var peak_mb: Double?
    public init(
        queue_s: Double? = nil, load_s: Double? = nil, ttft_s: Double? = nil, prefill_tps: Double? = nil,
        decode_tps: Double? = nil, max_itl_ms: Double? = nil, peak_mb: Double? = nil
    ) {
        self.queue_s = queue_s; self.load_s = load_s; self.ttft_s = ttft_s; self.prefill_tps = prefill_tps
        self.decode_tps = decode_tps; self.max_itl_ms = max_itl_ms; self.peak_mb = peak_mb
    }
    public var json: JSONObject { (try? WireJSON.decode(JSONEncoder().encode(self))) ?? [:] }
}
/// Tracks generated token boundaries, even when detokenization buffers text or a token belongs to reasoning.
public struct TokenTiming: Sendable {
    public let accepted: Double
    public var scheduled: Double
    public var prefillStarted: Double?
    public var promptTokens: Int = 0
    public var cachedTokens: Int = 0
    public private(set) var generatedTokens = 0
    public private(set) var first: Double?
    public private(set) var last: Double?
    public private(set) var worstGap: Double?
    public init(accepted: Double = ProcessInfo.processInfo.systemUptime, scheduled: Double? = nil) {
        self.accepted = accepted; self.scheduled = scheduled ?? accepted
    }
    public mutating func token(at now: Double = ProcessInfo.processInfo.systemUptime) {
        if let last { worstGap = max(worstGap ?? 0, max(0, now - last)) }
        if first == nil { first = now }
        last = now; generatedTokens += 1
    }
    public func timing(peakMB: Double? = nil) -> RequestTiming {
        let prefill = first.map { $0 - (prefillStarted ?? scheduled) }
        let decode = first.flatMap { first in last.map { $0 - first } }
        return RequestTiming(
            queue_s: max(0, scheduled - accepted), ttft_s: first.map { max(0, $0 - accepted) },
            prefill_tps: prefill.flatMap { $0 > 0 ? Double(max(0, promptTokens - cachedTokens)) / $0 : nil },
            decode_tps: decode.flatMap { $0 > 0 && generatedTokens > 1 ? Double(generatedTokens - 1) / $0 : nil },
            max_itl_ms: worstGap.map { $0 * 1000 }, peak_mb: peakMB)
    }
    public func progress(id: JSONValue, phase: String, peakMB: Double? = nil) -> JSONObject {
        var result: JSONObject = [
            "id": id, "event": "progress", "phase": .string(phase), "prompt_tokens": .number(Double(promptTokens)),
            "cached_tokens": .number(Double(cachedTokens)), "generated_tokens": .number(Double(generatedTokens)), "timing": .object(timing(peakMB: peakMB).json)
        ]
        result["_arsenal_first_token_at"] = first.map(JSONValue.number)
        result["batch"] = 1
        return result
    }
}
