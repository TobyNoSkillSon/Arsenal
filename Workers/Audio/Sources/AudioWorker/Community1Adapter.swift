// SPDX-License-Identifier: MIT
// pyannote/speaker-diarization-community-1 with pyannote.audio 4.0.0's SpeakerDiarization.apply:
// sliding PyanNet segmentation (powerset → multilabel), speaker counting, masked WeSpeaker embeddings with
// overlap exclusion, VBx clustering, reconstruction and binarisation into regular and exclusive diarizations.
import Foundation
import IntegrationKit
import MLX
import ArsenalWire

final class Community1Adapter: AudioAdapter {
    override var architecture: String { "pyannote_community_1" }
    override var operations: Set<String> { ["diarize"] }
    /// Lab fixtures may shrink the 10 s window; the step stays one tenth of the window.
    private var chunkSamples = Community1Config.chunkSamples
    private var filters: MLXArray?

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        var layers = 4, blocks = [3, 4, 6, 3]
        chunkSamples = Community1Config.chunkSamples
        if context.randomWeights {
            layers = try audioInteger(context.config, key: "lab_layers", fallback: 4, range: 1...4)
            if context.config["lab_small_resnet"]?.bool == true { blocks = [1, 1, 1, 1] }
            chunkSamples = try audioInteger(context.config, key: "lab_chunk_samples", fallback: chunkSamples, range: 16_000...160_000)
            guard chunkSamples % 160 == 0 else { throw audioFailure("lab_chunk_samples must be a multiple of 160.") }
        }
        try prepare(C1Network(layers: layers, blocks: blocks), context, cancellation: cancellation, eligible: c1Eligible)
        filters = nil
    }

    override func unload() {
        filters = nil
        super.unload()
    }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "diarize", let network = graph as? C1Network else { throw audioFailure("Community-1 is not loaded or the operation is unsupported.") }
        try audioKeys(input, allowed: ["audio", "num_speakers", "min_speakers", "max_speakers"])
        func optional(_ key: String) throws -> Int? { input[key] == nil ? nil : try audioInteger(input, key: key, fallback: 1, range: 1...32) }
        let bounds = try Community1SpeakerBounds(num: try optional("num_speakers"), min: try optional("min_speakers"), max: try optional("max_speakers"))
        guard let path = input["audio"]?["path"]?.string else { throw audioFailure("An audio file is required.") }
        // pyannote's Audio(mono="downmix"): average the channels. Mono sources arrive as two identical channels.
        let stereo = try NativeMedia.audio(URL(fileURLWithPath: path), sampleRate: Community1Config.sampleRate, channels: 2, maximumFrames: Community1Config.sampleRate * 600)
        let samples = (0..<(stereo.samples.count / 2)).map { (stereo.samples[2 * $0] + stereo.samples[2 * $0 + 1]) / 2 }
        guard !samples.isEmpty else { throw audioFailure("The audio file has no samples.") }
        try cancellation.check()

        let segmentation = try segment(samples, network: network, cancellation: cancellation)
        let chunkSeconds = Double(chunkSamples) / Double(Community1Config.sampleRate), stepSeconds = chunkSeconds / 10
        let timeline = Community1Timeline(chunks: segmentation.count, chunkDuration: chunkSeconds, chunkStep: stepSeconds)
        var count = timeline.speakerCount(segmentation)
        if count.allSatisfy({ $0 == 0 }) { return Self.result([], [], speakers: [], embeddings: [], uri: Self.uri(path)) }

        let embeddings = try embed(samples, segmentation: segmentation, network: network, cancellation: cancellation)
        let clusters = try Community1Clustering.cluster(
            embeddings: embeddings, segmentations: segmentation, plda: network.plda.value(), config: .default, bounds: bounds, cancellation: cancellation)
        if let maximum = bounds.maximum { count = count.map { min($0, maximum) } }
        var hard = clusters.hardClusters
        for chunk in hard.indices {
            for speaker in 0..<Community1Config.localSpeakers where segmentation[chunk].allSatisfy({ $0[speaker] == 0 }) { hard[chunk][speaker] = -2 }
        }
        try cancellation.check()
        let regular = timeline.annotate(timeline.reconstruct(segmentation, hard: hard, count: count))
        let exclusive = timeline.annotate(timeline.reconstruct(segmentation, hard: hard, count: count.map { min($0, 1) }))
        // Annotation.labels() sorts cluster indices as strings ("10" before "2"); that order becomes SPEAKER_00, SPEAKER_01, …
        let labels = Self.labelOrder(Set(regular.map { $0.cluster }))
        var names = Dictionary(uniqueKeysWithValues: labels.enumerated().map { ($0.element, String(format: "SPEAKER_%02d", $0.offset)) })
        for cluster in Self.labelOrder(Set(exclusive.map { $0.cluster }).subtracting(labels)) { names[cluster] = String(format: "SPEAKER_%02d", names.count) }
        let vectors = labels.map { cluster -> [Double] in
            cluster < clusters.centroids.count ? clusters.centroids[cluster] : [Double](repeating: 0, count: Community1Config.embeddingDimension)
        }
        return Self.result(
            regular.map { ($0.start, $0.end, names[$0.cluster]!) }, exclusive.map { ($0.start, $0.end, names[$0.cluster]!) },
            speakers: labels.map { names[$0]! }, embeddings: vectors, uri: Self.uri(path))
    }

    /// Sliding 10 s windows every 1 s; a final partial window starts after the last complete one and is zero-padded.
    func chunkStarts(_ count: Int) -> [Int] {
        let step = chunkSamples / 10
        let complete = count >= chunkSamples ? (count - chunkSamples) / step + 1 : 0
        let last = count < chunkSamples || (count - chunkSamples) % step > 0
        return (0..<(complete + (last ? 1 : 0))).map { $0 * step }
    }

    func window(_ samples: [Float], start: Int) -> [Float] {
        let end = min(start + chunkSamples, samples.count)
        var audio = Array(samples[start..<end])
        if audio.count < chunkSamples { audio += [Float](repeating: 0, count: chunkSamples - audio.count) }
        return audio
    }

    /// [chunk][frame][3] binary local-speaker activity: hard powerset argmax converted to multilabel.
    func segment(_ samples: [Float], network: C1Network, cancellation: NativeCancellation) throws -> [[[Float]]] {
        if filters == nil { filters = network.segmentation.sinc.filters(dtype: .float16) }
        guard let filters else { throw audioFailure("Community-1 filters are unavailable.") }
        let starts = chunkStarts(samples.count)
        var result: [[[Float]]] = []
        result.reserveCapacity(starts.count)
        for batchStart in stride(from: 0, to: starts.count, by: Community1Config.segmentationBatch) {
            try cancellation.check()
            let batch = starts[batchStart..<min(batchStart + Community1Config.segmentationBatch, starts.count)]
            let audio = batch.flatMap { window(samples, start: $0) }
            let scores = try network.segmentation(
                MLXArray(audio, [batch.count, chunkSamples, 1]).asType(.float16), filters: filters, cancellation: cancellation)
            let classes = argMax(scores, axis: -1).asType(.int32)
            eval(classes)
            let frames = classes.dim(1), values = classes.asArray(Int32.self)
            for row in 0..<batch.count {
                result.append((0..<frames).map { Community1Config.powerset[Int(values[row * frames + $0])] })
            }
        }
        return result
    }

    /// [chunk][3][256] embeddings. Each speaker's mask excludes overlapped frames unless fewer than
    /// min_num_frames clean frames remain; ResNet frames are computed once per chunk and pooled per speaker.
    func embed(_ samples: [Float], segmentation: [[[Float]]], network: C1Network, cancellation: NativeCancellation) throws -> [[[Float]]] {
        let starts = chunkStarts(samples.count)
        var result: [[[Float]]] = []
        result.reserveCapacity(starts.count)
        for (chunk, start) in starts.enumerated() {
            try cancellation.check()
            let binary = segmentation[chunk], frames = binary.count
            let minimumFrames = Int(ceil(Double(frames * Community1Config.minimumEmbeddingSamples) / Double(chunkSamples)))
            var masks: [[Float]] = []
            for speaker in 0..<Community1Config.localSpeakers {
                let mask = binary.map { $0[speaker] }
                let clean = binary.map { $0.reduce(0, +) < 2 ? $0[speaker] : 0 }
                masks.append(clean.reduce(0, +) > Float(minimumFrames) ? clean : mask)
            }
            let features = try network.embedding.frames(try Community1Fbank.features(window(samples, start: start)), cancellation: cancellation)
            let pooled = features.dim(1)
            // F.interpolate(mode="nearest"): source = ⌊destination × (input / output)⌋ in Float32.
            let scale = Float(frames) / Float(pooled)
            let weights = masks.flatMap { mask in (0..<pooled).map { mask[min(frames - 1, Int(floor(Float($0) * scale)))] } }
            let vectors = network.embedding.embed(features, weights: MLXArray(weights, [Community1Config.localSpeakers, pooled]))
            eval(vectors)
            let values = vectors.asType(.float32).asArray(Float.self), dimension = Community1Config.embeddingDimension
            result.append((0..<Community1Config.localSpeakers).map { Array(values[($0 * dimension)..<(($0 + 1) * dimension)]) })
        }
        return result
    }

    static func labelOrder(_ clusters: Set<Int>) -> [Int] { clusters.sorted { String($0) < String($1) } }

    static func uri(_ path: String) -> String {
        let stem = URL(fileURLWithPath: path).deletingPathExtension().lastPathComponent
        let cleaned = stem.unicodeScalars.map { CharacterSet.whitespacesAndNewlines.contains($0) ? "_" : String($0) }.joined()
        return cleaned.isEmpty ? "audio" : cleaned
    }

    /// pyannote's DiarizeOutput.serialize (times rounded to milliseconds) plus RTTM lines as written by write_rttm.
    /// `decimal` rounds each time's exact binary value to milliseconds, as Python's round(x, 3) does (DiariZen's frame
    /// centres fall on half milliseconds, where rounding the scaled product to even goes the other way).
    static func result(
        _ regular: [(Double, Double, String)], _ exclusive: [(Double, Double, String)], speakers: [String], embeddings: [[Double]], uri: String,
        decimal: Bool = false
    ) -> JSONObject {
        func rounded(_ value: Double) -> Double {
            decimal ? Double(String(format: "%.3f", value)) ?? value : (value * 1000).rounded(.toNearestOrEven) / 1000
        }
        func segments(_ list: [(Double, Double, String)]) -> JSONValue {
            .array(list.map { .object(["start": .number(rounded($0.0)), "end": .number(rounded($0.1)), "speaker": .string($0.2)]) })
        }
        let rttm = regular.map { String(format: "SPEAKER %@ 1 %.3f %.3f <NA> <NA> %@ <NA> <NA>", uri, $0.0, $0.1 - $0.0, $0.2) }.joined(separator: "\n")
        return [
            "segments": segments(regular), "exclusive_segments": segments(exclusive), "speakers": .array(speakers.map(JSONValue.string)),
            "rttm": .string(rttm.isEmpty ? "" : rttm + "\n"),
            "speaker_embeddings": .array(embeddings.map { vector in vector.contains(where: { !$0.isFinite }) ? .null : .array(vector.map { .number($0) }) })
        ]
    }
}

/// Frame-level aggregation shared by speaker counting and reconstruction (Inference.aggregate over the
/// chunk sliding window into the PyanNet receptive-field frames), and Binarize(onset = offset = 0.5).
/// Frame geometry and local speaker count default to Community-1's PyanNet; DiariZen passes WavLM's 25 ms / 20 ms
/// frames and four local speakers (pyannote 3.1's aggregate uses the same arithmetic, with frames starting at 0).
struct Community1Timeline {
    let chunks: Int
    let chunkDuration: Double
    let chunkStep: Double
    var frameDuration = Community1Config.frameDuration
    var frameStep = Community1Config.frameStep
    var localSpeakers = Community1Config.localSpeakers
    var frames: Int { closestFrame(chunkDuration + Double(chunks - 1) * chunkStep + 0.5 * frameDuration) + 1 }

    /// SlidingWindow.closest_frame: rint((t − start − duration / 2) / step), start = 0.
    func closestFrame(_ time: Double) -> Int {
        Int(((time - 0.5 * frameDuration) / frameStep).rounded(.toNearestOrEven))
    }
    func startFrame(_ chunk: Int) -> Int { closestFrame(Double(chunk) * chunkStep + 0.5 * frameDuration) }

    /// speaker_count: mean over overlapping chunks of the number of active local speakers, rint to an integer.
    func speakerCount(_ segmentation: [[[Float]]]) -> [Int] {
        let total = frames
        var sums = [Float](repeating: 0, count: total), overlaps = [Float](repeating: 0, count: total)
        for (chunk, values) in segmentation.enumerated() {
            let offset = startFrame(chunk)
            for (frame, speakers) in values.enumerated() where offset + frame < total {
                sums[offset + frame] += speakers.reduce(0, +)
                overlaps[offset + frame] += 1
            }
        }
        return (0..<total).map { overlaps[$0] == 0 ? 0 : Int((sums[$0] / max(overlaps[$0], 1e-12)).rounded(.toNearestOrEven)) }
    }

    /// reconstruct + to_diarization: per-cluster activations summed over chunks (max over local speakers sharing a
    /// cluster), then the `count` most active clusters per frame (stable for ties, zero-padded when count exceeds them).
    func reconstruct(_ segmentation: [[[Float]]], hard: [[Int]], count: [Int]) -> [[Bool]] {
        let total = frames, clusters = max(0, (hard.flatMap { $0 }.max() ?? -1) + 1)
        let width = max(clusters, count.max() ?? 0)
        var activations = [[Float]](repeating: [Float](repeating: 0, count: width), count: total)
        for (chunk, values) in segmentation.enumerated() {
            let offset = startFrame(chunk)
            for cluster in Set(hard[chunk]) where cluster >= 0 {
                let members = (0..<localSpeakers).filter { hard[chunk][$0] == cluster }
                for (frame, speakers) in values.enumerated() where offset + frame < total {
                    activations[offset + frame][cluster] += members.map { speakers[$0] }.max() ?? 0
                }
            }
        }
        return (0..<total).map { t in
            let order = (0..<width).sorted { activations[t][$0] == activations[t][$1] ? $0 < $1 : activations[t][$0] > activations[t][$1] }
            var active = [Bool](repeating: false, count: width)
            for index in order.prefix(t < count.count ? count[t] : 0) { active[index] = true }
            return active
        }
    }

    /// Binarize over frame centres; a segment still active at the end closes at the last frame centre.
    func annotate(_ binary: [[Bool]]) -> [(start: Double, end: Double, cluster: Int)] {
        guard let width = binary.first?.count, width > 0 else { return [] }
        // SlidingWindow[i].middle = 0.5 · (s + (s + duration)) with s = i · step, bit for bit.
        let times = binary.indices.map { index -> Double in
            let start = Double(index) * frameStep
            return 0.5 * (start + (start + frameDuration))
        }
        var result: [(start: Double, end: Double, cluster: Int)] = []
        for cluster in 0..<width {
            var start = times[0], active = binary[0][cluster]
            for t in 1..<max(1, binary.count) {
                if active && !binary[t][cluster] {
                    result.append((start, times[t], cluster)); start = times[t]; active = false
                } else if !active && binary[t][cluster] {
                    start = times[t]; active = true
                }
            }
            if active, let last = times.last, last - start > 1e-6 { result.append((start, last, cluster)) }
        }
        // itertracks: segments in (start, end) order; tracks of one segment in string order of their original label.
        return result.sorted { ($0.start, $0.end, String($0.cluster)) < ($1.start, $1.end, String($1.cluster)) }
    }
}
