// SPDX-License-Identifier: MIT
import Foundation
import MLX

/// Streaming geometry in 80 ms frames (NeMo `sortformer_modules` attributes). Presets are the model card's table.
struct SortformerV2Geometry: Equatable {
    var chunk: Int
    var rightContext: Int
    var fifo: Int
    var updatePeriod: Int
    var cache: Int
    var leftContext = 1

    static let presets: [String: SortformerV2Geometry] = [
        "very_high": .init(chunk: 340, rightContext: 40, fifo: 40, updatePeriod: 300, cache: 188),
        "high": .init(chunk: 124, rightContext: 1, fifo: 124, updatePeriod: 124, cache: 188),
        "low": .init(chunk: 6, rightContext: 7, fifo: 188, updatePeriod: 144, cache: 188),
        "ultra_low": .init(chunk: 3, rightContext: 1, fifo: 188, updatePeriod: 144, cache: 188)
    ]
    static let defaultPreset = "very_high"
    static let silenceFramesPerSpeaker = 3

    /// NeMo `_check_streaming_parameters` (hard limits) plus Arsenal's bounds on one step's attention length.
    func validate(speakers: Int = 4) throws {
        guard chunk >= 1, rightContext >= 0, leftContext >= 0, fifo >= 0, updatePeriod >= 1,
            cache >= (1 + Self.silenceFramesPerSpeaker) * speakers
        else { throw audioFailure("Invalid streaming geometry: chunk ≥ 1, update period ≥ 1, context and FIFO ≥ 0, speaker cache ≥ 16 frames.") }
        guard chunk <= 1_000, rightContext <= 100, fifo <= 1_000, updatePeriod <= 1_000, cache <= 1_000 else {
            throw audioFailure("Streaming geometry is limited to 1000 frames per buffer and 100 frames of right context.")
        }
    }

    var updater: SortformerConfig {
        SortformerConfig(
            chunkLenSeconds: Float(chunk), leftContextSeconds: Float(leftContext), rightContextSeconds: Float(rightContext),
            spkcacheLen: cache, fifoLen: fifo, fcDModel: 512, maxSpeakers: 4, spkcacheUpdatePeriod: updatePeriod)
    }
}

/// One NeMo streaming diarization over mel frames (`forward_streaming`, synchronous): chunks of `chunk * 8` mel frames
/// with up to `leftContext`/`rightContext` encoder frames of context, each step running the network over
/// [speaker cache | FIFO | chunk] and updating the cache with NeMo's `streaming_update` (SortformerStateUpdater).
final class SortformerV2Stream {
    let network: SortformerV2Network
    let geometry: SortformerV2Geometry
    private(set) var state: SortformerStreamingState
    private let updater: SortformerStateUpdater
    /// Confirmed speaker probabilities, [frames × 4] row-major, 80 ms per frame.
    private(set) var probabilities: [Float] = []
    /// Next mel frame that starts a chunk.
    private(set) var nextFrame = 0
    var cancellation: NativeCancellationLike?

    init(network: SortformerV2Network, geometry: SortformerV2Geometry) {
        self.network = network
        self.geometry = geometry
        state = SortformerStreamingState(config: geometry.updater)
        updater = SortformerStateUpdater(config: geometry.updater)
    }

    /// Runs every chunk whose features (with full right context unless `final`) are available. `features(range)` returns
    /// mel frames [range.count × 128]; `totalFrames` is the number of mel frames known so far.
    func advance(totalFrames: Int, final: Bool, features: (Range<Int>) -> [Float]) throws {
        let sub = 8
        while nextFrame < totalFrames {
            try cancellation?.check()
            let start = nextFrame
            let left = min(geometry.leftContext * sub, start)
            let end = min(start + geometry.chunk * sub, totalFrames)
            let right = min(geometry.rightContext * sub, totalFrames - end)
            if !final, end - start < geometry.chunk * sub || right < geometry.rightContext * sub { break }
            let chunkPreds = try step(
                features((start - left)..<(end + right)), frames: end + right - start + left,
                leftFrames: (left + sub / 2) / sub, rightFrames: (right + sub - 1) / sub)
            probabilities.append(contentsOf: chunkPreds)
            nextFrame = end
        }
    }

    /// Starts from a given cache state (component parity on the maker's own state).
    func replaceState(_ state: SortformerStreamingState) { self.state = state }

    /// One `forward_streaming_step`; returns the chunk's core probabilities.
    func step(_ mel: [Float], frames: Int, leftFrames: Int, rightFrames: Int) throws -> [Float] {
        let width = 512, speakers = 4
        let embeddings = network.preEncode(MLXArray(mel, [1, frames, SortformerV2Features.mels]))
        let chunkFrames = embeddings.dim(1)
        var parts: [MLXArray] = []
        if state.spkcacheLength > 0 { parts.append(MLXArray(state.spkcache, [1, state.spkcacheLength, width]).asType(network.dtype)) }
        if state.fifoLength > 0 { parts.append(MLXArray(state.fifo, [1, state.fifoLength, width]).asType(network.dtype)) }
        parts.append(embeddings)
        let joined = parts.count == 1 ? embeddings : concatenated(parts, axis: 1)
        let preds = network.infer(network.frontEnd(joined))
        eval(preds, embeddings)
        // The state is kept at the network's half precision, as NeMo keeps it in the model dtype.
        let chunk = embeddings.asType(.float32).asArray(Float.self)
        // NeMo's predictions are tensors of the network dtype; its cache scoring and output use those rounded values.
        let all = preds.asType(network.dtype).asType(.float32).asArray(Float.self)
        guard all.count == (state.spkcacheLength + state.fifoLength + chunkFrames) * speakers, chunk.count == chunkFrames * width else {
            throw audioFailure("Sortformer step returned an unexpected shape.")
        }
        let update = updater.update(state: &state, chunkEmbs: chunk, preds: all, leftContext: leftFrames, rightContext: rightFrames)
        state.meanSilenceEmbedding = state.meanSilenceEmbedding.map { Float(Float16($0)) }
        return update.confirmed
    }
}

/// A cancellation check the stream can call without depending on IntegrationKit's concrete type in tests.
protocol NativeCancellationLike: AnyObject { func check() throws }

/// NeMo `ts_vad_post_processing` per speaker (frames repeated to 10 ms, `binarization` then `filtering`), rounded to
/// 2 decimals as `_diarize_output_processing` does, then `merge_float_intervals`.
struct SortformerV2PostProcessing: Equatable {
    var onset: Float = 0.5
    var offset: Float = 0.5
    var padOnset: Float = 0
    var padOffset: Float = 0
    var minDurationOn: Float = 0
    var minDurationOff: Float = 0

    struct Segment: Equatable { var start: Double; var end: Double; var speaker: Int }

    func segments(_ probabilities: [Float], speakers: Int = 4) -> [Segment] {
        let frames = probabilities.count / speakers
        var all: [Segment] = []
        for speaker in 0..<speakers {
            var spans: [(Double, Double)] = []
            var speech = false, start = 0.0
            let total = frames * 8
            for i in 0..<total {
                let value = probabilities[(i / 8) * speakers + speaker]
                if speech {
                    if value < offset {
                        let end = Double(i) * 0.01 + Double(padOffset)
                        if end > max(0, start - Double(padOnset)) { spans.append((max(0, start - Double(padOnset)), end)) }
                        start = Double(i) * 0.01
                        speech = false
                    }
                } else if value > onset {
                    start = Double(i) * 0.01
                    speech = true
                }
            }
            if speech, total > 0 { spans.append((max(0, start - Double(padOnset)), Double(total - 1) * 0.01 + Double(padOffset))) }
            spans = Self.mergeOverlaps(spans)
            // NeMo filters float32 tensors: durations are float32 differences compared with the threshold in float32.
            if minDurationOn > 0 { spans = spans.filter { Float($0.1) - Float($0.0) >= minDurationOn } }
            if minDurationOff > 0, spans.count > 1 {
                var gaps: [(Double, Double)] = []
                for (a, b) in zip(spans, spans.dropFirst()) where Float(b.0) - Float(a.1) < minDurationOff { gaps.append((a.1, b.0)) }
                spans = Self.mergeOverlaps(spans + gaps)
            }
            let rounded = spans.map { (($0.0 * 100).rounded() / 100, ($0.1 * 100).rounded() / 100) }
            all += Self.mergeFloatIntervals(rounded).map { Segment(start: $0.0, end: $0.1, speaker: speaker) }
        }
        return all.sorted { ($0.start, $0.speaker) < ($1.start, $1.speaker) }
    }

    /// NeMo `merge_overlap_segment`: sort and merge segments that overlap or touch.
    static func mergeOverlaps(_ spans: [(Double, Double)]) -> [(Double, Double)] {
        var merged: [(Double, Double)] = []
        for span in spans.sorted(by: { $0.0 < $1.0 }) {
            if let last = merged.last, span.0 <= last.1 { merged[merged.count - 1].1 = max(last.1, span.1) } else { merged.append(span) }
        }
        return merged
    }

    /// NeMo `merge_float_intervals(decimals=5, margin=2)`: integer ranges in 1e-5 units with the start shifted by two,
    /// merged when they overlap or are adjacent, then shifted back.
    static func mergeFloatIntervals(_ spans: [(Double, Double)]) -> [(Double, Double)] {
        var ints: [(Int, Int)] = spans.compactMap { span in
            let start = Int((span.0 * 100_000).rounded()) + 2, end = Int((span.1 * 100_000).rounded())
            return start < end ? (start, end) : nil
        }
        guard ints.count > 1 else { return ints.map { (Double($0.0 - 2) / 100_000, Double($0.1) / 100_000) } }
        ints.sort { $0 < $1 }
        var merged: [(Int, Int)] = [ints[0]]
        for (start, end) in ints.dropFirst() {
            if start <= merged[merged.count - 1].1 + 1 { merged[merged.count - 1].1 = max(merged[merged.count - 1].1, end) } else { merged.append((start, end)) }
        }
        return merged.map { (Double($0.0 - 2) / 100_000, Double($0.1) / 100_000) }
    }
}
