// SPDX-License-Identifier: MIT
// DiariZenPipeline.__call__ (diarizen/pipelines/inference.py) host steps with its pyannote 3.1.1 fork: hard powerset
// segmentation, a median filter, speaker counting, overlap-excluding embedding masks, DiariZen's VBxClustering
// (pipelines/clustering.py) and reconstruction. AHC, PLDA and VBx reuse Community-1's ports (the maker's VBx.py is the
// same algorithm, GMM update, Fa 0.07, Fb 0.8, 20 iterations). Double precision on the host.
import Foundation
import IntegrationKit

enum DiariZenPipeline {
    static let sampleRate = 16_000
    /// config.toml inference: 16 s windows every 1.6 s (segmentation_step 0.1), median filter on, batch 32.
    static let windowSamples = 256_000
    static let stepSamples = 25_600
    static let localSpeakers = 4
    /// Model._receptive_field (400 samples, 320-sample step); pyannote's aggregate re-anchors these frames at 0.
    static let frameDuration = 400.0 / 16_000
    static let frameStep = 320.0 / 16_000
    static let medianWindow = 11
    /// config.toml clustering.max_speakers caps the per-frame speaker count (the maker's clustering ignores bounds).
    static let maximumSpeakers = 20
    /// PretrainedSpeakerEmbedding.min_num_samples for WeSpeaker ResNet34 (one 25 ms fbank frame).
    static let minimumEmbeddingSamples = 400
    /// filter_embeddings(min_frames_ratio = 0.1): round(0.1 × frames) clean frames.
    static let minimumCleanRatio = 0.1
    static let speakerPriorCutoff = 1e-7

    /// pyannote.audio.utils.powerset.Powerset(4 classes, at most 4 per frame): {} then singletons, pairs, triples, all four
    /// in itertools.combinations order.
    static let powerset: [[Float]] = {
        var sets: [[Int]] = [[]]
        func combinations(_ k: Int, from start: Int, _ prefix: [Int]) {
            if prefix.count == k { sets.append(prefix); return }
            for i in start..<localSpeakers { combinations(k, from: i + 1, prefix + [i]) }
        }
        for k in 1...localSpeakers { combinations(k, from: 0, []) }
        return sets.map { set in (0..<localSpeakers).map { set.contains($0) ? 1 : 0 } }
    }()

    /// Inference.slide: complete windows every step, then one zero-padded window when samples remain.
    static func chunkStarts(_ count: Int, window: Int = windowSamples, step: Int = stepSamples) -> [Int] {
        let complete = count >= window ? (count - window) / step + 1 : 0
        let last = count < window || (count - window) % step > 0
        return (0..<(complete + (last ? 1 : 0))).map { $0 * step }
    }

    static func window(_ samples: [Float], start: Int, length: Int = windowSamples) -> [Float] {
        let end = min(start + length, samples.count)
        var audio = start < end ? Array(samples[start..<end]) : []
        if audio.count < length { audio += [Float](repeating: 0, count: length - audio.count) }
        return audio
    }

    /// scipy.ndimage.median_filter(size=(1, 11, 1), mode="reflect") on [chunk][frame][speaker] binary values: the
    /// window mirrors at the edges including the edge sample (d c b a | a b c d | d c b a).
    static func medianFilter(_ segmentation: [[[Float]]], size: Int = medianWindow) -> [[[Float]]] {
        let half = size / 2
        return segmentation.map { frames in
            let count = frames.count
            guard count > 0 else { return frames }
            func reflect(_ i: Int) -> Int {
                var j = i
                while j < 0 || j >= count {
                    if j < 0 { j = -j - 1 }
                    if j >= count { j = 2 * count - j - 1 }
                }
                return j
            }
            let speakers = frames[0].count
            return (0..<count).map { t in
                (0..<speakers).map { s in
                    var values = (-half...half).map { frames[reflect(t + $0)][s] }
                    values.sort()
                    return values[half]
                }
            }
        }
    }

    /// get_embeddings(exclude_overlap=True): each speaker's mask without overlapped frames, unless no more than
    /// min_num_frames clean frames remain, then the full mask.
    static func embeddingMasks(_ binary: [[Float]], samples: Int = windowSamples) -> [[Float]] {
        let frames = binary.count
        let minimumFrames = Int(ceil(Double(frames * minimumEmbeddingSamples) / Double(samples)))
        return (0..<localSpeakers).map { speaker in
            let mask = binary.map { $0[speaker] }
            let clean = binary.map { $0.reduce(0, +) < 2 ? $0[speaker] : 0 }
            return clean.reduce(0, +) > Float(minimumFrames) ? clean : mask
        }
    }

    struct Clusters {
        /// [chunk][speaker] after the maker's renumbering (np.unique over every value, -2 included), before inactive → -2.
        let hard: [[Int]]
        let train: [(chunk: Int, speaker: Int)]
        let ahc: [Int]
        let priors: [Double]
    }

    /// VBxClustering.__call__ of DiariZen's pyannote fork.
    static func cluster(
        embeddings: [[[Float]]], segmentations: [[[Float]]], plda: Community1PLDA, config: Community1Config = .default, cancellation: NativeCancellation
    ) throws -> Clusters {
        let chunks = embeddings.count
        guard segmentations.count == chunks else { throw audioFailure("DiariZen embedding and segmentation chunk counts differ.") }
        let speakers = embeddings.first?.count ?? 0
        let frames = segmentations.first?.count ?? 0
        // filter_embeddings: active, no NaN, at least round(0.1 × frames) single-speaker frames; retry with 0 if < 2 remain.
        func select(_ minimum: Int) -> [(chunk: Int, speaker: Int)] {
            var out: [(chunk: Int, speaker: Int)] = []
            for chunk in 0..<chunks {
                for speaker in 0..<speakers {
                    var active = false, clean = 0
                    for frame in segmentations[chunk] where frame[speaker] > 0 {
                        active = true
                        if frame.reduce(0, +) == 1 { clean += 1 }
                    }
                    if active, clean >= minimum, !embeddings[chunk][speaker].contains(where: \.isNaN) { out.append((chunk, speaker)) }
                }
            }
            return out
        }
        var train = select(Int((minimumCleanRatio * Double(frames)).rounded(.toNearestOrEven)))
        if train.count < 2 { train = select(0) }
        if train.count < 2 {
            return Clusters(hard: Array(repeating: Array(repeating: 0, count: speakers), count: chunks), train: train, ahc: [], priors: [])
        }
        try cancellation.check()
        let vectors = train.map { embeddings[$0.chunk][$0.speaker] }
        let normalized = vectors.map { row -> [Double] in
            let norm = sqrt(row.reduce(0.0) { $0 + Double($1) * Double($1) })
            return row.map { Double($0) / norm }
        }
        let ahc = try Community1Clustering.ahcLabels(normalized, threshold: config.threshold, cancellation: cancellation)
        let (q, priors) = try Community1Clustering.vbx(
            initialLabels: ahc, features: plda.transform(vectors), phi: plda.phi, config: config, cancellation: cancellation)
        // centroids = q[:, sp > 1e-7].T @ train (unnormalised sums; cosine distances do not need the division).
        let kept = priors.indices.filter { priors[$0] > speakerPriorCutoff }
        let dimension = vectors[0].count
        let centroids = kept.map { k -> [Double] in
            var sum = [Double](repeating: 0, count: dimension)
            for (row, vector) in vectors.enumerated() {
                let weight = q[row][k]
                for d in 0..<dimension { sum[d] += weight * Double(vector[d]) }
            }
            return sum
        }
        var soft = embeddings.map { chunk in chunk.map { e in centroids.map { 2 - Community1Clustering.cosineDistance(e, $0) } } }
        // constrained_argmax: NaN → nanmin, then a maximum-weight assignment per chunk; unmatched speakers stay -2.
        let floor = soft.flatMap { $0.flatMap { $0 } }.filter { !$0.isNaN }.min() ?? 0
        soft = soft.map { $0.map { $0.map { $0.isNaN ? floor : $0 } } }
        let raw = soft.map(DiariZenAssignment.maximize)
        // np.unique(hard, return_inverse=True): every value present, -2 included, renumbered in sorted order.
        let values = Array(Set(raw.flatMap { $0 })).sorted()
        let index = Dictionary(uniqueKeysWithValues: values.enumerated().map { ($0.element, $0.offset) })
        return Clusters(hard: raw.map { $0.map { index[$0]! } }, train: train, ahc: ahc, priors: priors)
    }
}

/// scipy.optimize.linear_sum_assignment(scores, maximize=True) for a rectangular score matrix: an exact
/// shortest-augmenting-path Hungarian algorithm; rows left without a column get -2.
enum DiariZenAssignment {
    static func maximize(_ scores: [[Double]]) -> [Int] {
        let rows = scores.count, columns = scores.first?.count ?? 0
        guard rows > 0, columns > 0 else { return Array(repeating: -2, count: rows) }
        if rows > columns {
            // Assign every column to a row instead, then read the matching back per row.
            let transposed = (0..<columns).map { c in (0..<rows).map { scores[$0][c] } }
            let byColumn = minimize(transposed.map { $0.map { -$0 } })
            var out = [Int](repeating: -2, count: rows)
            for (column, row) in byColumn.enumerated() where row >= 0 { out[row] = column }
            return out
        }
        return minimize(scores.map { $0.map { -$0 } })
    }

    /// Hungarian algorithm (potentials, O(n² m)) for n ≤ m: returns the column of each row.
    static func minimize(_ cost: [[Double]]) -> [Int] {
        let n = cost.count, m = cost[0].count
        var u = [Double](repeating: 0, count: n + 1), v = [Double](repeating: 0, count: m + 1)
        var p = [Int](repeating: 0, count: m + 1), way = [Int](repeating: 0, count: m + 1)
        for i in 1...n {
            p[0] = i
            var j0 = 0
            var minv = [Double](repeating: .infinity, count: m + 1), used = [Bool](repeating: false, count: m + 1)
            repeat {
                used[j0] = true
                let i0 = p[j0]
                var delta = Double.infinity, j1 = 0
                for j in 1...m where !used[j] {
                    let current = cost[i0 - 1][j - 1] - u[i0] - v[j]
                    if current < minv[j] { minv[j] = current; way[j] = j0 }
                    if minv[j] < delta { delta = minv[j]; j1 = j }
                }
                for j in 0...m {
                    if used[j] { u[p[j]] += delta; v[j] -= delta } else { minv[j] -= delta }
                }
                j0 = j1
            } while p[j0] != 0
            repeat {
                let j1 = way[j0]
                p[j0] = p[j1]
                j0 = j1
            } while j0 != 0
        }
        var out = [Int](repeating: -2, count: n)
        for j in 1...m where p[j] > 0 { out[p[j] - 1] = j - 1 }
        return out
    }
}
