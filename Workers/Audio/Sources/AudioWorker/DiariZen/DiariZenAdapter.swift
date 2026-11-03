// SPDX-License-Identifier: MIT
// BUT-FIT/diarizen-wavlm-large-s80-md-v2 run the way DiariZenPipeline.__call__ runs it: 16 s sliding windows through the
// WavLM + Conformer segmentation model (hard powerset → multilabel), a median filter, speaker counting, WeSpeaker
// embeddings with overlap exclusion, VBx clustering, reconstruction and binarisation. Output follows the audio kind's
// diarization contract (pyannote Community-1's), with exclusive segments made the pyannote 4.0 way.
import AVFoundation
import Foundation
import IntegrationKit
import MLX
import ArsenalWire

final class DiariZenAdapter: AudioAdapter {
    override var architecture: String { "diarizen" }
    override var operations: Set<String> { ["diarize"] }
    static let maximumSeconds = 600
    /// Windows per segmentation batch: the maker uses 32; eight bound the attention bias memory and give the same values.
    static let segmentationBatch = 8
    private var windowSamples = DiariZenPipeline.windowSamples
    private var stepSamples = DiariZenPipeline.stepSamples
    var network: DiariZenNetwork?

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        var dims = DiariZenDimensions.s80
        let file = context.weights.appendingPathComponent("config.json")
        if !context.randomWeights {
            dims = try JSONDecoder().decode(DiariZenDimensions.self, from: Data(contentsOf: file))
        }
        windowSamples = DiariZenPipeline.windowSamples; stepSamples = DiariZenPipeline.stepSamples
        var blocks = [3, 4, 6, 3]
        if context.randomWeights {
            dims = .lab(layers: try audioInteger(context.config, key: "lab_layers", fallback: 24, range: 1...24))
            if context.config["lab_small_resnet"]?.bool == true { blocks = [1, 1, 1, 1] }
            windowSamples = try audioInteger(context.config, key: "lab_window_samples", fallback: windowSamples, range: 32_000...256_000)
            guard windowSamples % 320 == 0 else { throw audioFailure("lab_window_samples must be a multiple of 320.") }
            stepSamples = windowSamples / 10
        }
        let network = DiariZenNetwork(dims, blocks: blocks)
        try prepare(network, context, cancellation: cancellation, subdirectories: [DiariZenConversion.embeddingDirectory], eligible: { _, _ in false })
        self.network = network
    }

    override func unload() { network = nil; super.unload() }

    var dtype: DType { network?.segmentation.classifier.weight.dtype ?? .float16 }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "diarize" else { throw audioFailure("DiariZen supports diarize only.") }
        try audioKeys(input, allowed: ["audio"])
        guard let path = input["audio"]?["path"]?.string else { throw audioFailure("An audio file is required.") }
        return try diarize(path: path, clip: true, cancellation: cancellation)
    }

    /// DiariZen reads the first channel only ("force to use the SDM data"), so no channels are mixed: channel 0 alone
    /// is resampled to 16 kHz. NativeMedia streams it, checks every frame and refuses unreadable files with a 422.
    static func firstChannel(_ url: URL, maximumSeconds: Int = maximumSeconds) throws -> [Float] {
        try NativeMedia.audio(
            url, sampleRate: DiariZenPipeline.sampleRate, channels: 1, mix: .channel(0), maximumFrames: DiariZenPipeline.sampleRate * maximumSeconds
        ).samples
    }

    /// The maker's pipeline end to end. `clip` ends segments with the recording (the served default, an Arsenal
    /// deviation); parity compares the unclipped segments, which are the maker's exactly.
    func diarize(path: String, clip: Bool, cancellation: NativeCancellation) throws -> JSONObject {
        guard let network else { throw audioFailure("DiariZen is not loaded.") }
        let samples = try Self.firstChannel(URL(fileURLWithPath: path))
        guard !samples.isEmpty else { throw audioFailure("The audio file has no samples.") }
        try cancellation.check()
        let raw = try segment(samples, network: network, cancellation: cancellation)
        let segmentation = DiariZenPipeline.medianFilter(raw)
        let timeline = Community1Timeline(
            chunks: segmentation.count, chunkDuration: Double(windowSamples) / Double(DiariZenPipeline.sampleRate),
            chunkStep: Double(stepSamples) / Double(DiariZenPipeline.sampleRate), frameDuration: DiariZenPipeline.frameDuration,
            frameStep: DiariZenPipeline.frameStep, localSpeakers: DiariZenPipeline.localSpeakers)
        let count = timeline.speakerCount(segmentation).map { min($0, DiariZenPipeline.maximumSpeakers) }
        let embeddings = try embed(samples, segmentation: segmentation, network: network, cancellation: cancellation)
        let clusters = try DiariZenPipeline.cluster(
            embeddings: embeddings, segmentations: segmentation, plda: network.plda.value(), cancellation: cancellation)
        try cancellation.check()
        return Self.output(
            segmentation: segmentation, hard: clusters.hard, count: count, timeline: timeline, uri: Community1Adapter.uri(path),
            duration: clip ? Double(samples.count) / Double(DiariZenPipeline.sampleRate) : .infinity)
    }

    /// Inactive local speakers → -2, reconstruct (and its exclusive variant), binarise, name speakers.
    /// Segments are clipped to the recording: the last window is zero-padded and its frames can outlast the audio (the
    /// maker's DiariZenPipeline does not crop; pyannote's own pipeline does).
    static func output(
        segmentation: [[[Float]]], hard raw: [[Int]], count: [Int], timeline: Community1Timeline, uri: String, duration: Double = .infinity
    ) -> JSONObject {
        var hard = raw
        for chunk in hard.indices {
            for speaker in hard[chunk].indices where segmentation[chunk].allSatisfy({ $0[speaker] == 0 }) { hard[chunk][speaker] = -2 }
        }
        func clipped(_ list: [(start: Double, end: Double, cluster: Int)]) -> [(start: Double, end: Double, cluster: Int)] {
            list.compactMap { $0.start < duration ? ($0.start, min($0.end, duration), $0.cluster) : nil }
        }
        let regular = clipped(timeline.annotate(timeline.reconstruct(segmentation, hard: hard, count: count)))
        let exclusive = clipped(timeline.annotate(timeline.reconstruct(segmentation, hard: hard, count: count.map { min($0, 1) })))
        let labels = Community1Adapter.labelOrder(Set(regular.map { $0.cluster }))
        var names = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, String(format: "SPEAKER_%02d", $0.offset)) })
        for cluster in Community1Adapter.labelOrder(Set(exclusive.map { $0.cluster }).subtracting(labels)) { names[cluster] = String(format: "SPEAKER_%02d", names.count) }
        var result = Community1Adapter.result(
            regular.map { ($0.start, $0.end, names[$0.cluster]!) }, exclusive.map { ($0.start, $0.end, names[$0.cluster]!) },
            speakers: labels.map { names[$0]! }, embeddings: [], uri: uri, decimal: true)
        result["speaker_embeddings"] = nil
        return result
    }

    /// [chunk][frame][4] hard multilabel activity: powerset argmax mapped through the 16-class table.
    func segment(_ samples: [Float], network: DiariZenNetwork, cancellation: NativeCancellation) throws -> [[[Float]]] {
        let starts = DiariZenPipeline.chunkStarts(samples.count, window: windowSamples, step: stepSamples)
        var result: [[[Float]]] = []
        result.reserveCapacity(starts.count)
        for batchStart in stride(from: 0, to: starts.count, by: Self.segmentationBatch) {
            try cancellation.check()
            let batch = starts[batchStart..<min(batchStart + Self.segmentationBatch, starts.count)]
            let audio = batch.flatMap { DiariZenPipeline.window(samples, start: $0, length: windowSamples) }
            let scores = try network.segmentation(MLXArray(audio, [batch.count, windowSamples]), dtype: dtype, cancellation: cancellation)
            let classes = argMax(scores, axis: -1).asType(.int32)
            eval(classes)
            let frames = classes.dim(1), values = classes.asArray(Int32.self)
            for row in 0..<batch.count {
                result.append((0..<frames).map { DiariZenPipeline.powerset[Int(values[row * frames + $0])] })
            }
            Memory.clearCache()
        }
        return result
    }

    /// [chunk][4][256] WeSpeaker embeddings: ResNet frames once per window, pooled per speaker mask (nearest-interpolated).
    func embed(_ samples: [Float], segmentation: [[[Float]]], network: DiariZenNetwork, cancellation: NativeCancellation) throws -> [[[Float]]] {
        let starts = DiariZenPipeline.chunkStarts(samples.count, window: windowSamples, step: stepSamples)
        var result: [[[Float]]] = []
        result.reserveCapacity(starts.count)
        for (chunk, start) in starts.enumerated() {
            try cancellation.check()
            let binary = segmentation[chunk], frames = binary.count
            let masks = DiariZenPipeline.embeddingMasks(binary, samples: windowSamples)
            let features = try network.embedding.frames(
                try Community1Fbank.features(DiariZenPipeline.window(samples, start: start, length: windowSamples)), cancellation: cancellation)
            let pooled = features.dim(1)
            // F.interpolate(mode="nearest"): source = ⌊destination × (input / output)⌋ in Float32.
            let scale = Float(frames) / Float(pooled)
            let weights = masks.flatMap { mask in (0..<pooled).map { mask[min(frames - 1, Int(floor(Float($0) * scale)))] } }
            let vectors = network.embedding.embed(features, weights: MLXArray(weights, [DiariZenPipeline.localSpeakers, pooled]))
            eval(vectors)
            let values = vectors.asType(.float32).asArray(Float.self), dimension = values.count / DiariZenPipeline.localSpeakers
            result.append((0..<DiariZenPipeline.localSpeakers).map { Array(values[($0 * dimension)..<(($0 + 1) * dimension)]) })
        }
        return result
    }
}
