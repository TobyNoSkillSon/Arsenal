// SPDX-License-Identifier: MIT
// Community-1 clustering on the host: pyannote.audio 4.0.0 VBxClustering (utils/vbx.py, pipelines/clustering.py),
// with SciPy 1.16 centroid linkage + fcluster(criterion="distance") and scikit-learn 1.9 KMeans
// (k-means++, n_init=3, random_state=42, Lloyd) ported for identical decisions. Double precision throughout.
import Foundation
import IntegrationKit

/// x-vector LDA transform and PLDA (xvec_transform.npz, plda.npz), as set up by pyannote's vbx_setup.
struct Community1PLDA {
    let mean1: [Double]
    let mean2: [Double]
    /// [256][128]
    let lda: [[Double]]
    let mu: [Double]
    /// [128][128]
    let tr: [[Double]]
    let psi: [Double]

    /// vbx_setup solves eigh(B, W) with W = inv(trᵀ·tr) and B = inv((trᵀ/psi)·tr). Its eigenvalues are psi and its
    /// W-normalised eigenvectors are the rows of tr, so plda_tr = rows of tr ordered by descending psi (eigenvector
    /// signs do not change VBx responsibilities).
    var order: [Int] { psi.indices.sorted { psi[$0] == psi[$1] ? $0 > $1 : psi[$0] > psi[$1] } }
    var phi: [Double] { order.map { psi[$0] } }

    /// PLDA(embeddings) = plda_tf(xvec_tf(x)): sqrt(128)·l2(lda.T·(sqrt(256)·l2(x − mean1)) − mean2), then (· − mu)·plda_trᵀ.
    func transform(_ embeddings: [[Float]]) -> [[Double]] {
        let rows = order, input = mean1.count, output = mean2.count
        return embeddings.map { embedding in
            var centered = (0..<input).map { Double(embedding[$0]) - mean1[$0] }
            let first = sqrt(centered.reduce(0) { $0 + $1 * $1 })
            centered = centered.map { $0 / first * sqrt(Double(input)) }
            var projected = [Double](repeating: 0, count: output)
            for i in 0..<input where centered[i] != 0 {
                let row = lda[i], value = centered[i]
                for j in 0..<output { projected[j] += value * row[j] }
            }
            projected = (0..<output).map { projected[$0] - mean2[$0] }
            let second = sqrt(projected.reduce(0) { $0 + $1 * $1 })
            let shifted = (0..<output).map { projected[$0] / second * sqrt(Double(output)) - mu[$0] }
            return rows.map { r in
                let vector = tr[r]
                var sum = 0.0
                for j in 0..<output { sum += shifted[j] * vector[j] }
                return sum
            }
        }
    }
}

struct Community1ClusterResult {
    /// [chunk][local speaker]: cluster index, or -2 when constrained assignment left the speaker out.
    let hardClusters: [[Int]]
    /// [cluster][256]; a cluster without members is NaN, as NumPy's mean of an empty selection.
    let centroids: [[Double]]
}

enum Community1Clustering {
    /// VBxClustering.__call__ with Community-1 parameters.
    static func cluster(
        embeddings: [[[Float]]], segmentations: [[[Float]]], plda: Community1PLDA, config: Community1Config, bounds: Community1SpeakerBounds,
        cancellation: NativeCancellation
    ) throws -> Community1ClusterResult {
        let chunks = embeddings.count, speakers = Community1Config.localSpeakers, dimension = Community1Config.embeddingDimension
        guard segmentations.count == chunks else { throw audioFailure("Community-1 embedding and segmentation chunk counts differ.") }
        // filter_embeddings: at least 20 % clean (single-speaker) frames and no NaN in the embedding.
        var train: [[Float]] = []
        for chunk in 0..<chunks {
            let frames = segmentations[chunk]
            for speaker in 0..<speakers {
                var clean = 0
                for frame in frames where frame.reduce(0, +) == 1 && frame[speaker] > 0 { clean += 1 }
                if Double(clean) >= config.minimumActiveRatio * Double(frames.count), !embeddings[chunk][speaker].contains(where: \.isNaN) {
                    train.append(embeddings[chunk][speaker])
                }
            }
        }
        if train.count < 2 {
            let mean = (0..<dimension).map { d in train.isEmpty ? Double.nan : train.reduce(0.0) { $0 + Double($1[d]) } / Double(train.count) }
            return Community1ClusterResult(hardClusters: Array(repeating: Array(repeating: 0, count: speakers), count: chunks), centroids: [mean])
        }
        try cancellation.check()
        let normalized = train.map { row -> [Double] in
            let norm = Float(sqrt(Double(row.reduce(Float(0)) { $0 + $1 * $1 })))
            return row.map { Double($0 / norm) }
        }
        let initial = try ahcLabels(normalized, threshold: config.threshold, cancellation: cancellation)
        let features = plda.transform(train)
        let (responsibilities, priors) = try vbx(initialLabels: initial, features: features, phi: plda.phi, config: config, cancellation: cancellation)
        let kept = priors.indices.filter { priors[$0] > config.speakerPriorCutoff }
        var centroids = kept.map { k -> [Double] in
            var sum = [Double](repeating: 0, count: dimension), total = 0.0
            for (row, embedding) in train.enumerated() {
                let weight = responsibilities[row][k]
                total += weight
                for d in 0..<dimension { sum[d] += weight * Double(embedding[d]) }
            }
            return sum.map { $0 / total }
        }
        var constrained = true
        var fixed = bounds.exact
        if centroids.count < bounds.minimum {
            fixed = bounds.minimum
        } else if let maximum = bounds.maximum, centroids.count > maximum {
            fixed = maximum
        }
        if let fixed, fixed != centroids.count {
            // Re-cluster the normalised training embeddings with scikit-learn's KMeans(n_init=3, random_state=42).
            constrained = false
            guard fixed <= normalized.count else {
                throw audioFailure("The recording yields \(normalized.count) usable speaker embeddings, fewer than the \(fixed) speakers requested.")
            }
            let labels = try Community1KMeans.fitPredict(normalized, clusters: fixed, cancellation: cancellation)
            centroids = (0..<fixed).map { k in
                let members = labels.indices.filter { labels[$0] == k }
                return (0..<dimension).map { d in members.isEmpty ? Double.nan : members.reduce(0.0) { $0 + Double(train[$1][d]) } / Double(members.count) }
            }
        }
        // soft_clusters = 2 − cosine distance between every (chunk, speaker) embedding and every centroid.
        var soft = embeddings.map { chunk in chunk.map { embedding in centroids.map { 2 - cosineDistance(embedding, $0) } } }
        if !constrained {
            return Community1ClusterResult(hardClusters: soft.map { $0.map(argmax) }, centroids: centroids)
        }
        let all = soft.flatMap { $0.flatMap { $0 } }
        let constant = all.contains(where: \.isNaN) ? Double.nan : (all.min() ?? 0) - 1
        for chunk in 0..<chunks {
            for speaker in 0..<speakers where segmentations[chunk].allSatisfy({ $0[speaker] == 0 }) {
                soft[chunk][speaker] = Array(repeating: constant, count: centroids.count)
            }
        }
        let floor = soft.flatMap { $0.flatMap { $0 } }.filter { !$0.isNaN }.min() ?? 0
        let hard = soft.map { chunk in assign(chunk.map { $0.map { $0.isNaN ? floor : $0 } }) }
        return Community1ClusterResult(hardClusters: hard, centroids: centroids)
    }

    /// np.argmax: first maximum, or the first NaN when one is present.
    static func argmax(_ row: [Double]) -> Int {
        if let nan = row.firstIndex(where: \.isNaN) { return nan }
        var best = 0
        for index in row.indices where row[index] > row[best] { best = index }
        return best
    }

    /// scipy.optimize.linear_sum_assignment(cost, maximize=True) for the three local speakers: an exact
    /// maximum-weight matching of min(speakers, clusters) pairs; unmatched speakers stay -2.
    static func assign(_ scores: [[Double]]) -> [Int] {
        let speakers = scores.count, clusters = scores.first?.count ?? 0
        var best = [Int](repeating: -2, count: speakers), bestScore = -Double.infinity
        guard clusters > 0 else { return best }
        let pairs = min(speakers, clusters)
        var current = [Int](repeating: -2, count: speakers), used = [Bool](repeating: false, count: clusters)
        func search(_ speaker: Int, _ assigned: Int, _ score: Double) {
            if speaker == speakers {
                if assigned == pairs && score > bestScore { bestScore = score; best = current }
                return
            }
            if assigned < pairs {
                for cluster in 0..<clusters where !used[cluster] {
                    used[cluster] = true; current[speaker] = cluster
                    search(speaker + 1, assigned + 1, score + scores[speaker][cluster])
                    used[cluster] = false; current[speaker] = -2
                }
            }
            if speakers - speaker - 1 >= pairs - assigned { search(speaker + 1, assigned, score) }
        }
        search(0, 0, 0)
        return best
    }

    static func cosineDistance(_ left: [Float], _ right: [Double]) -> Double {
        var dot = 0.0, a = 0.0, b = 0.0
        for index in left.indices {
            let x = Double(left[index]), y = right[index]
            dot += x * y; a += x * x; b += y * y
        }
        return 1 - dot / (sqrt(a) * sqrt(b))
    }

    /// linkage(method="centroid", metric="euclidean") then fcluster(threshold, criterion="distance") − 1.
    static func ahcLabels(_ points: [[Double]], threshold: Double, cancellation: NativeCancellation) throws -> [Int] {
        let tree = try Community1Linkage.centroid(points, cancellation: cancellation)
        return Community1Linkage.flatClusters(tree, count: points.count, cutoff: threshold).map { $0 - 1 }
    }

    /// cluster_vbx: AHC labels softened by softmax(7 · one-hot), then VBx (Fa, Fb, at most 20 iterations, ε 1e-4).
    static func vbx(initialLabels: [Int], features: [[Double]], phi: [Double], config: Community1Config, cancellation: NativeCancellation) throws -> (
        responsibilities: [[Double]], priors: [Double]
    ) {
        let frames = features.count, dimension = phi.count
        let speakers = (initialLabels.max() ?? 0) + 1
        var gamma = initialLabels.map { label in softmax((0..<speakers).map { $0 == label ? config.initialSmoothing : 0 }) }
        var priors = [Double](repeating: 1 / Double(speakers), count: speakers)
        let ratio = config.fa / config.fb
        let rho = features.map { feature in zip(feature, phi).map { $0 * sqrt($1) } }
        let constant = features.map { feature in -0.5 * (feature.reduce(0) { $0 + $1 * $1 } + Double(dimension) * log(2 * Double.pi)) }
        var previous: Double?
        for iteration in 0..<config.maxIterations {
            try cancellation.check()
            var inverseL = [[Double]](repeating: [Double](repeating: 0, count: dimension), count: speakers)
            var alpha = inverseL
            for speaker in 0..<speakers {
                var mass = 0.0, weighted = [Double](repeating: 0, count: dimension)
                for frame in 0..<frames {
                    let g = gamma[frame][speaker]
                    mass += g
                    for d in 0..<dimension { weighted[d] += g * rho[frame][d] }
                }
                for d in 0..<dimension {
                    inverseL[speaker][d] = 1 / (1 + ratio * mass * phi[d])
                    alpha[speaker][d] = ratio * inverseL[speaker][d] * weighted[d]
                }
            }
            let logPriors = priors.map { log($0 + 1e-8) }
            var total = 0.0
            for frame in 0..<frames {
                var logits = [Double](repeating: 0, count: speakers)
                for speaker in 0..<speakers {
                    var first = 0.0, second = 0.0
                    for d in 0..<dimension {
                        first += rho[frame][d] * alpha[speaker][d]
                        second += (inverseL[speaker][d] + alpha[speaker][d] * alpha[speaker][d]) * phi[d]
                    }
                    logits[speaker] = config.fa * (first - 0.5 * second + constant[frame]) + logPriors[speaker]
                }
                let marginal = logSumExp(logits)
                total += marginal
                gamma[frame] = logits.map { exp($0 - marginal) }
            }
            priors = (0..<speakers).map { speaker in gamma.reduce(0) { $0 + $1[speaker] } }
            let sum = priors.reduce(0, +)
            priors = priors.map { $0 / sum }
            var regularization = 0.0
            for speaker in 0..<speakers {
                for d in 0..<dimension { regularization += log(inverseL[speaker][d]) - inverseL[speaker][d] - alpha[speaker][d] * alpha[speaker][d] + 1 }
            }
            let elbo = total + config.fb * 0.5 * regularization
            if iteration > 0, let previous, elbo - previous < config.convergenceEpsilon { break }
            previous = elbo
        }
        return (gamma, priors)
    }

    static func softmax(_ values: [Double]) -> [Double] {
        let top = values.max() ?? 0
        let exponentials = values.map { exp($0 - top) }
        let total = exponentials.reduce(0, +)
        return exponentials.map { $0 / total }
    }

    static func logSumExp(_ values: [Double]) -> Double {
        let top = values.max() ?? 0
        guard top.isFinite else { return top }
        return top + log(values.reduce(0) { $0 + exp($1 - top) })
    }
}

/// SciPy `_hierarchy.fast_linkage` (Müllner's generic algorithm) for the centroid method, its binary heap,
/// and `cluster_dist` (max merge distance per subtree, then the top-down monocrit numbering).
enum Community1Linkage {
    struct Merge {
        let left: Int
        let right: Int
        let distance: Double
        let size: Int
    }

    private struct Heap {
        var indexByKey: [Int], keyByIndex: [Int], values: [Double], size: Int
        init(_ values: [Double]) {
            size = values.count
            indexByKey = Array(0..<size); keyByIndex = Array(0..<size); self.values = values
            for index in stride(from: size / 2 - 1, through: 0, by: -1) { siftDown(index) }
        }
        func min() -> (key: Int, value: Double) { (keyByIndex[0], values[0]) }
        mutating func removeMin() {
            swap(0, size - 1); size -= 1; siftDown(0)
        }
        mutating func change(_ key: Int, _ value: Double) {
            let index = indexByKey[key], old = values[index]
            values[index] = value
            if value < old { siftUp(index) } else { siftDown(index) }
        }
        mutating func siftUp(_ start: Int) {
            var index = start, parent = (index - 1) >> 1
            while index > 0 && values[parent] > values[index] {
                swap(index, parent); index = parent; parent = (index - 1) >> 1
            }
        }
        mutating func siftDown(_ start: Int) {
            var index = start, child = (index << 1) + 1
            while child < size {
                if child + 1 < size && values[child + 1] < values[child] { child += 1 }
                if values[index] > values[child] {
                    swap(index, child); index = child; child = (index << 1) + 1
                } else {
                    break
                }
            }
        }
        mutating func swap(_ i: Int, _ j: Int) {
            values.swapAt(i, j)
            let keyI = keyByIndex[i], keyJ = keyByIndex[j]
            keyByIndex[i] = keyJ; keyByIndex[j] = keyI
            indexByKey[keyI] = j; indexByKey[keyJ] = i
        }
    }

    static func centroid(_ points: [[Double]], cancellation: NativeCancellation) throws -> [Merge] {
        let n = points.count
        guard n > 1 else { return [] }
        func condensed(_ i: Int, _ j: Int) -> Int { i < j ? n * i - i * (i + 1) / 2 + (j - i - 1) : n * j - j * (j + 1) / 2 + (i - j - 1) }
        var distances = [Double](repeating: 0, count: n * (n - 1) / 2)
        for i in 0..<(n - 1) {
            if i % 64 == 0 { try cancellation.check() }
            let a = points[i]
            for j in (i + 1)..<n {
                let b = points[j]
                var sum = 0.0
                for d in a.indices { let difference = a[d] - b[d]; sum += difference * difference }
                distances[condensed(i, j)] = sqrt(sum)
            }
        }
        var size = [Int](repeating: 1, count: n), clusterID = Array(0..<n)
        var neighbor = [Int](repeating: 0, count: n - 1), minimum = [Double](repeating: 0, count: n - 1)
        func nearest(_ x: Int) -> (Int, Double) {
            var best = Double.infinity, y = -1
            for i in (x + 1)..<n where size[i] != 0 {
                let distance = distances[condensed(x, i)]
                if distance < best { best = distance; y = i }
            }
            return (y, best)
        }
        for x in 0..<(n - 1) { (neighbor[x], minimum[x]) = nearest(x) }
        var heap = Heap(minimum)
        var merges: [Merge] = []
        merges.reserveCapacity(n - 1)
        for k in 0..<(n - 1) {
            if k % 64 == 0 { try cancellation.check() }
            var x = 0, y = 0, distance = 0.0
            for _ in 0..<(n - k) {
                (x, distance) = heap.min()
                y = neighbor[x]
                if distance == distances[condensed(x, y)] { break }
                (y, distance) = nearest(x)
                neighbor[x] = y; minimum[x] = distance
                heap.change(x, distance)
            }
            heap.removeMin()
            var idX = clusterID[x], idY = clusterID[y]
            let nx = size[x], ny = size[y]
            if idX > idY { Swift.swap(&idX, &idY) }
            merges.append(Merge(left: idX, right: idY, distance: distance, size: nx + ny))
            size[x] = 0; size[y] = nx + ny; clusterID[y] = n + k
            for z in 0..<n {
                let nz = size[z]
                if nz == 0 || z == y { continue }
                let dzx = distances[condensed(z, x)], dzy = distances[condensed(z, y)]
                let sx = Double(nx), sy = Double(ny)
                distances[condensed(z, y)] = sqrt(((sx * dzx * dzx + sy * dzy * dzy) - (sx * sy * distance * distance) / (sx + sy)) / (sx + sy))
            }
            for z in 0..<x where size[z] > 0 && neighbor[z] == x { neighbor[z] = y }
            for z in 0..<y where size[z] != 0 {
                let candidate = distances[condensed(z, y)]
                if candidate < minimum[z] {
                    neighbor[z] = y; minimum[z] = candidate
                    heap.change(z, candidate)
                }
            }
            if y < n - 1 {
                let (z, candidate) = nearest(y)
                if z != -1 {
                    neighbor[y] = z; minimum[y] = candidate
                    heap.change(y, candidate)
                }
            }
        }
        return merges
    }

    /// fcluster(Z, cutoff, criterion="distance"): 1-based flat cluster numbers in SciPy's traversal order.
    static func flatClusters(_ merges: [Merge], count n: Int, cutoff: Double) -> [Int] {
        guard n > 1 else { return [Int](repeating: 1, count: n) }
        // get_max_dist_for_each_cluster
        var maxDistance = [Double](repeating: 0, count: n - 1)
        var stack = [2 * n - 2], visited = Set<Int>()
        while let node = stack.last {
            let root = node - n, left = merges[root].left, right = merges[root].right
            if left >= n && !visited.contains(left) { visited.insert(left); stack.append(left); continue }
            if right >= n && !visited.contains(right) { visited.insert(right); stack.append(right); continue }
            var value = merges[root].distance
            if left >= n { value = max(value, maxDistance[left - n]) }
            if right >= n { value = max(value, maxDistance[right - n]) }
            maxDistance[root] = value
            stack.removeLast()
        }
        // cluster_monocrit
        var labels = [Int](repeating: 0, count: n), clusters = 0, leader = -1
        stack = [2 * n - 2]; visited = []
        while let node = stack.last {
            let root = node - n, left = merges[root].left, right = merges[root].right
            if leader == -1 && maxDistance[root] <= cutoff { leader = root; clusters += 1 }
            if left >= n && !visited.contains(left) { visited.insert(left); stack.append(left); continue }
            if right >= n && !visited.contains(right) { visited.insert(right); stack.append(right); continue }
            if left < n {
                if leader == -1 { clusters += 1 }
                labels[left] = clusters
            }
            if right < n {
                if leader == -1 { clusters += 1 }
                labels[right] = clusters
            }
            if leader == root { leader = -1 }
            stack.removeLast()
        }
        return labels
    }
}

/// NumPy's legacy RandomState(seed): MT19937 seeded by init_genrand, 53-bit random_sample.
struct Community1RandomState {
    private var state = [UInt32](repeating: 0, count: 624)
    private var index = 624
    init(seed: UInt32) {
        state[0] = seed
        for i in 1..<624 { state[i] = 1_812_433_253 &* (state[i - 1] ^ (state[i - 1] >> 30)) &+ UInt32(i) }
    }
    mutating func next() -> UInt32 {
        if index >= 624 {
            for i in 0..<624 {
                let y = (state[i] & 0x8000_0000) | (state[(i + 1) % 624] & 0x7fff_ffff)
                var value = state[(i + 397) % 624] ^ (y >> 1)
                if y & 1 != 0 { value ^= 0x9908_b0df }
                state[i] = value
            }
            index = 0
        }
        var y = state[index]
        index += 1
        y ^= y >> 11
        y ^= (y << 7) & 0x9d2c_5680
        y ^= (y << 15) & 0xefc6_0000
        y ^= y >> 18
        return y
    }
    mutating func randomSample() -> Double {
        let a = Double(next() >> 5), b = Double(next() >> 6)
        return (a * 67_108_864.0 + b) / 9_007_199_254_740_992.0
    }
}

/// scikit-learn KMeans(n_clusters, n_init=3, random_state=42, copy_x=False).fit_predict on Float32 data:
/// mean-centred k-means++ (2 + ⌊ln k⌋ local trials), Lloyd iterations (max 300, tol 1e-4 × mean variance),
/// empty-cluster relocation, best inertia over the three initialisations.
enum Community1KMeans {
    static func fitPredict(_ points: [[Double]], clusters k: Int, cancellation: NativeCancellation) throws -> [Int] {
        let n = points.count, d = points.first?.count ?? 0
        guard k >= 1, n >= k else { throw audioFailure("KMeans needs at least as many embeddings as clusters.") }
        let mean = (0..<d).map { j in points.reduce(0.0) { $0 + $1[j] } / Double(n) }
        let variance = (0..<d).map { j in points.reduce(0.0) { $0 + ($1[j] - mean[j]) * ($1[j] - mean[j]) } / Double(n) }
        let tolerance = variance.reduce(0, +) / Double(d) * 1e-4
        let x = points.map { row in (0..<d).map { row[$0] - mean[$0] } }
        var random = Community1RandomState(seed: 42)
        var bestLabels: [Int]?, bestInertia = Double.infinity
        for _ in 0..<3 {
            try cancellation.check()
            let centers = plusPlus(x, k, &random)
            let (labels, inertia) = try lloyd(x, centers, tolerance: tolerance, cancellation: cancellation)
            if bestLabels == nil || (inertia < bestInertia && !sameClustering(labels, bestLabels!, k)) {
                bestLabels = labels; bestInertia = inertia
            }
        }
        return bestLabels!
    }

    static func squared(_ a: [Double], _ b: [Double]) -> Double {
        var sum = 0.0
        for i in a.indices { let difference = a[i] - b[i]; sum += difference * difference }
        return sum
    }

    static func plusPlus(_ x: [[Double]], _ k: Int, _ random: inout Community1RandomState) -> [[Double]] {
        let n = x.count, trials = 2 + Int(log(Double(k)))
        // RandomState.choice(n, p=weights/sum): searchsorted(cdf, u, side="right").
        let p = Double(Float(1) / Float(n))
        var cdf = [Double](repeating: 0, count: n), running = 0.0
        for i in 0..<n { running += p; cdf[i] = running }
        let last = cdf[n - 1]
        let u = random.randomSample()
        var first = n - 1
        for i in 0..<n where cdf[i] / last > u { first = i; break }
        var centers = [x[first]]
        var closest = x.map { Float(squared(x[first], $0)) }
        var potential = closest.reduce(Float(0), +)
        for _ in 1..<k {
            let draws = (0..<trials).map { _ in random.randomSample() * Double(potential) }
            var cumulative = [Float](repeating: 0, count: n), sum: Float = 0
            for i in 0..<n { sum += closest[i]; cumulative[i] = sum }
            let candidates = draws.map { draw -> Int in
                var index = n - 1
                for i in 0..<n where Double(cumulative[i]) >= draw { index = i; break }
                return index
            }
            var bestIndex = 0, bestPotential = Float.infinity, bestDistances = closest
            for (trial, candidate) in candidates.enumerated() {
                let distances = (0..<n).map { Swift.min(closest[$0], Float(squared(x[candidate], x[$0]))) }
                let value = distances.reduce(Float(0), +)
                if value < bestPotential || trial == 0 { bestPotential = value; bestIndex = candidate; bestDistances = distances }
            }
            potential = bestPotential; closest = bestDistances
            centers.append(x[bestIndex])
        }
        return centers
    }

    static func lloyd(_ x: [[Double]], _ initial: [[Double]], tolerance: Double, cancellation: NativeCancellation) throws -> ([Int], Double) {
        let n = x.count, k = initial.count, d = initial[0].count
        var centers = initial, labels = [Int](repeating: -1, count: n), previous = labels
        func assignLabels(_ centers: [[Double]]) -> [Int] {
            let norms = centers.map { $0.reduce(0) { $0 + $1 * $1 } }
            return x.map { point in
                var best = 0, bestValue = Double.infinity
                for (j, center) in centers.enumerated() {
                    var dot = 0.0
                    for i in 0..<d { dot += point[i] * center[i] }
                    let value = norms[j] - 2 * dot
                    if j == 0 || value < bestValue { best = j; bestValue = value }
                }
                return best
            }
        }
        var strict = false
        for _ in 0..<300 {
            try cancellation.check()
            labels = assignLabels(centers)
            var sums = [[Double]](repeating: [Double](repeating: 0, count: d), count: k), weights = [Double](repeating: 0, count: k)
            for (i, label) in labels.enumerated() {
                weights[label] += 1
                for j in 0..<d { sums[label][j] += x[i][j] }
            }
            // _relocate_empty_clusters_dense: the farthest points from their centres seed empty clusters.
            let empty = (0..<k).filter { weights[$0] == 0 }
            if !empty.isEmpty {
                let distances = x.indices.map { squared(x[$0], centers[labels[$0]]) }
                if (distances.max() ?? 0) > 0 {
                    let far = distances.indices.sorted { distances[$0] == distances[$1] ? $0 < $1 : distances[$0] > distances[$1] }
                    for (position, cluster) in empty.enumerated() {
                        let index = far[position], old = labels[index]
                        for j in 0..<d { sums[old][j] -= x[index][j]; sums[cluster][j] = x[index][j] }
                        weights[cluster] = 1; weights[old] -= 1
                    }
                }
            }
            let heaviest = weights.indices.max { weights[$0] < weights[$1] || weights[$0] == weights[$1] && $0 > $1 } ?? 0
            var updated = sums
            for j in 0..<k where weights[j] > 0 { updated[j] = sums[j].map { $0 / weights[j] } }
            for j in 0..<k where weights[j] <= 0 { updated[j] = updated[heaviest] }
            let shift = (0..<k).reduce(0.0) { $0 + squared(updated[$1], centers[$1]) }
            centers = updated
            if labels == previous { strict = true; break }
            if shift <= tolerance { break }
            previous = labels
        }
        if !strict { labels = assignLabels(centers) }
        let inertia = labels.indices.reduce(0.0) { $0 + squared(x[$1], centers[labels[$1]]) }
        return (labels, inertia)
    }

    static func sameClustering(_ a: [Int], _ b: [Int], _ k: Int) -> Bool {
        var mapping = [Int](repeating: -1, count: k)
        for i in a.indices {
            if mapping[a[i]] == -1 { mapping[a[i]] = b[i] } else if mapping[a[i]] != b[i] { return false }
        }
        return true
    }
}
