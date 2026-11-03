// SPDX-License-Identifier: MIT
// Mitra Classifier 2 (AutoGluon, Apache-2.0) ported to MLX from the maker's code in autogluon.tabular 1.6.3,
// models/mitra/_internal: models/tab2d.py (Tab2D, Layer, MultiheadAttention, the non-flash path),
// models/embedding.py (Tab2DQuantileEmbeddingX, Tab2DEmbeddingX, Tab2DEmbeddingYClasses) and
// data/preprocessor.py, as MitraClassifier.predict_proba runs them zero-shot (fine_tune=False, one estimator,
// no random class shuffles, feature shuffles or sign mirrors). The quantile embedding of the columns runs on the
// host in Float32 as torch does; the 12 row/column attention layers run on the float16 weights.
import Foundation
import MLX
import MLXFast
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.tabular"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["classify_rows"]

struct MitraConfig {
    var dim = 512
    var classes = 10
    var layers = 12
    var heads = 4
    init(_ json: [String: Any]) throws {
        dim = (json["dim"] as? NSNumber)?.intValue ?? dim
        classes = (json["dim_output"] as? NSNumber)?.intValue ?? classes
        layers = (json["n_layers"] as? NSNumber)?.intValue ?? layers
        heads = (json["n_heads"] as? NSNumber)?.intValue ?? heads
        guard (json["task"] as? String ?? "CLASSIFICATION") == "CLASSIFICATION" else {
            throw RunnerError.failed("Only the Mitra classifier is ported.")
        }
    }
}

/// LayerNorm computed in float32 (autocast runs layer_norm in float32), returned in the input's dtype.
final class Norm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(_ dims: Int) {
        _weight.wrappedValue = MLXArray.ones([dims])
        _bias.wrappedValue = MLXArray.zeros([dims])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: weight.asType(.float32), bias: bias.asType(.float32), eps: 1e-5)
            .asType(x.dtype)
    }
}

/// MultiheadAttention: q, k, v, o with bias; scaled dot-product attention with the default 1/sqrt(head) scale.
final class Attention: Module {
    let heads: Int
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    init(_ dim: Int, heads: Int) {
        self.heads = heads
        _q.wrappedValue = Linear(dim, dim)
        _k.wrappedValue = Linear(dim, dim)
        _v.wrappedValue = Linear(dim, dim)
        _o.wrappedValue = Linear(dim, dim)
    }
    /// Projected keys and values [B, heads, L, head] of `x` [B, L, dim].
    func keysValues(_ x: MLXArray) -> (MLXArray, MLXArray) { (split(k(x)), split(v(x))) }
    /// Attention of `x` [B, Lq, dim] over the given keys and values.
    func callAsFunction(_ x: MLXArray, keys: MLXArray, values: MLXArray) -> MLXArray {
        let queries = split(q(x))
        let head = queries.dim(-1)
        let out = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: values, scale: 1 / Float(head).squareRoot(), mask: .none)
        return o(out.transposed(0, 2, 1, 3).reshaped([x.dim(0), x.dim(1), x.dim(2)]))
    }
    private func split(_ t: MLXArray) -> MLXArray {
        t.reshaped([t.dim(0), t.dim(1), heads, t.dim(2) / heads]).transposed(0, 2, 1, 3)
    }
}

/// One Tab2D layer: attention across rows (queries attend the labelled rows), an MLP, attention across the
/// columns of each row, and a second MLP; every block pre-norm with a residual.
final class Layer: Module {
    @ModuleInfo(key: "layer_norm1") var norm1: Norm
    @ModuleInfo(key: "attention1") var attention1: Attention
    @ModuleInfo(key: "layer_norm2") var norm2: Norm
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "linear2") var linear2: Linear
    @ModuleInfo(key: "layer_norm3") var norm3: Norm
    @ModuleInfo(key: "attention2") var attention2: Attention
    @ModuleInfo(key: "layer_norm4") var norm4: Norm
    @ModuleInfo(key: "linear3") var linear3: Linear
    @ModuleInfo(key: "linear4") var linear4: Linear

    init(_ dim: Int, heads: Int) {
        _norm1.wrappedValue = Norm(dim)
        _attention1.wrappedValue = Attention(dim, heads: heads)
        _norm2.wrappedValue = Norm(dim)
        _linear1.wrappedValue = Linear(dim, 4 * dim)
        _linear2.wrappedValue = Linear(4 * dim, dim)
        _norm3.wrappedValue = Norm(dim)
        _attention2.wrappedValue = Attention(dim, heads: heads)
        _norm4.wrappedValue = Norm(dim)
        _linear3.wrappedValue = Linear(dim, 4 * dim)
        _linear4.wrappedValue = Linear(4 * dim, dim)
    }

    /// support [S, F+1, D] and query [Q, F+1, D] → both updated. `columnGroup` columns share one row-attention call.
    func callAsFunction(_ support: MLXArray, _ query: MLXArray, columnGroup: Int) -> (MLXArray, MLXArray) {
        // Attention across rows: one sequence per column, (S, F+1, D) → (F+1, S, D).
        let normedSupport = norm1(support).transposed(1, 0, 2)
        let normedQuery = norm1(query).transposed(1, 0, 2)
        let columns = normedSupport.dim(0)
        var supportParts: [MLXArray] = [], queryParts: [MLXArray] = []
        var start = 0
        while start < columns {
            let end = min(start + columnGroup, columns)
            let s = normedSupport[start ..< end], q = normedQuery[start ..< end]
            let (keys, values) = attention1.keysValues(s)
            supportParts.append(attention1(s, keys: keys, values: values))
            queryParts.append(attention1(q, keys: keys, values: values))
            start = end
        }
        var support = support + concatenated(supportParts, axis: 0).transposed(1, 0, 2)
        var query = query + concatenated(queryParts, axis: 0).transposed(1, 0, 2)
        support = support + linear2(gelu(linear1(norm2(support))))
        query = query + linear2(gelu(linear1(norm2(query))))

        // Attention across the columns of each row; query rows attend only themselves.
        let s3 = norm3(support), q3 = norm3(query)
        let (sk, sv) = attention2.keysValues(s3)
        let (qk, qv) = attention2.keysValues(q3)
        support = support + attention2(s3, keys: sk, values: sv)
        query = query + attention2(q3, keys: qk, values: qv)
        support = support + linear4(gelu(linear3(norm4(support))))
        query = query + linear4(gelu(linear3(norm4(query))))
        return (support, query)
    }
}

final class XEmbedding: Module {
    @ModuleInfo(key: "x_embedding") var linear: Linear
    init(_ dim: Int) { _linear.wrappedValue = Linear(1, dim) }
}

final class YEmbedding: Module {
    @ModuleInfo(key: "y_embedding") var classes: Embedding
    @ModuleInfo(key: "y_mask") var mask: Embedding
    init(_ dim: Int, classes: Int) {
        _classes.wrappedValue = Embedding(embeddingCount: classes, dimensions: dim)
        _mask.wrappedValue = Embedding(embeddingCount: 1, dimensions: dim)
    }
}

final class Tab2D: Module {
    @ModuleInfo(key: "x_embedding") var xEmbedding: XEmbedding
    @ModuleInfo(key: "y_embedding") var yEmbedding: YEmbedding
    @ModuleInfo(key: "layers") var layers: [Layer]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: Norm
    @ModuleInfo(key: "final_layer") var finalLayer: Linear
    init(_ c: MitraConfig) {
        _xEmbedding.wrappedValue = XEmbedding(c.dim)
        _yEmbedding.wrappedValue = YEmbedding(c.dim, classes: c.classes)
        _layers.wrappedValue = (0..<c.layers).map { _ in Layer(c.dim, heads: c.heads) }
        _finalLayerNorm.wrappedValue = Norm(c.dim)
        _finalLayer.wrappedValue = Linear(c.dim, c.classes)
    }
}

/// Tab2DQuantileEmbeddingX for one column, in Float32 as torch computes it: the 999 quantiles 0.001…0.999 of
/// the labelled values (torch.quantile, linear), each value's bucket (torch.bucketize, right=False) over the row
/// count, then standardised with the labelled rows' mean and population variance (0 when the variance is 0).
func quantileEmbed(support: [Float], query: [Float]) -> (support: [Float], query: [Float]) {
    let sorted = support.sorted()
    let n = sorted.count
    var boundaries = [Float](repeating: 0, count: 999)
    for k in 1...999 {
        let level = Float(k) / Float(1000)
        let rank = level * Float(n - 1)
        let below = Int(rank)  // truncation, as toType(kLong)
        let above = Int(rank.rounded(.up))
        let weight = rank - Float(below)
        let start = sorted[below], end = sorted[min(above, n - 1)]
        // torch.lerp
        boundaries[k - 1] = weight < 0.5 ? start + weight * (end - start) : end - (end - start) * (1 - weight)
    }
    func bucket(_ x: Float) -> Float {
        // The count of boundaries strictly below x (boundaries are non-decreasing).
        var low = 0, high = boundaries.count
        while low < high {
            let middle = (low + high) / 2
            if boundaries[middle] < x { low = middle + 1 } else { high = middle }
        }
        return Float(low)
    }
    let count = Float(n)
    let s = support.map { bucket($0) / count }
    let q = query.map { bucket($0) / count }
    let mean = Float(s.reduce(0.0) { $0 + Double($1) } / Double(n))
    let centered = s.map { $0 - mean }
    let variance = Float(centered.reduce(0.0) { $0 + Double($1) * Double($1) } / Double(n))
    guard variance != 0 else { return ([Float](repeating: 0, count: s.count), [Float](repeating: 0, count: q.count)) }
    let std = variance.squareRoot()
    return (centered.map { $0 / std }, q.map { ($0 - mean) / std })
}

final class Model {
    let config: MitraConfig
    let net: Tab2D
    let maxSupport = 2048
    let maxRows = 10000
    let maxColumns = 100

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = try MitraConfig(try Files.json(weightsDir))
        net = Tab2D(config)
        try Weights.apply(try Weights.load(weightsDir), to: net)
    }

    private func table(_ params: Params, _ key: String) throws -> [[Double]] {
        guard let list = params.raw[key] as? [Any], !list.isEmpty else { throw RunnerError.invalid("\(key) must be a nonempty list of rows.") }
        return try list.map { row in
            guard let cells = row as? [Any], !cells.isEmpty else { throw RunnerError.invalid("Each row of \(key) must be a nonempty list of numbers.") }
            return try cells.map { cell in
                guard let number = cell as? NSNumber, number.doubleValue.isFinite else {
                    throw RunnerError.invalid("\(key) must hold only finite numbers.")
                }
                return number.doubleValue
            }
        }
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let trainRows = try table(params, "train_rows")
        let rows = try table(params, "rows")
        let labels = try params.strings("train_labels")
        let columns = trainRows[0].count
        guard trainRows.allSatisfy({ $0.count == columns }), rows.allSatisfy({ $0.count == columns }) else {
            throw RunnerError.invalid("Every row of train_rows and rows must have the same number of columns.")
        }
        guard columns <= maxColumns else { throw RunnerError.invalid("At most \(maxColumns) columns.") }
        guard labels.count == trainRows.count else { throw RunnerError.invalid("train_labels needs one label per row of train_rows.") }
        guard trainRows.count >= 2, trainRows.count <= maxSupport else {
            throw RunnerError.invalid("train_rows holds 2 to \(maxSupport) rows.")
        }
        guard rows.count <= maxRows else { throw RunnerError.invalid("rows holds at most \(maxRows) rows.") }

        // Classes in sorted order (numerically when every label is a number), as AutoGluon's label encoder does.
        let unique = Array(Set(labels))
        let numeric = unique.allSatisfy { Double($0) != nil }
        let classes = numeric ? unique.sorted { Double($0)! < Double($1)! } : unique.sorted()
        guard classes.count >= 2 else { throw RunnerError.invalid("train_labels needs at least two different labels.") }
        guard classes.count <= config.classes else { throw RunnerError.invalid("At most \(config.classes) different labels.") }
        let classIndex = Dictionary(uniqueKeysWithValues: classes.enumerated().map { ($1, $0) })
        let y = labels.map { Int32(classIndex[$0]!) }

        // Preprocessor: drop columns with one value in train_rows; values become float32.
        let kept = (0..<columns).filter { column in Set(trainRows.map { $0[column] }).count > 1 }
        guard !kept.isEmpty else { throw RunnerError.invalid("Every column of train_rows holds a single value, so nothing can be learned.") }
        let s = trainRows.count, q = rows.count, f = kept.count
        var supportX = [Float](repeating: 0, count: s * f), queryX = [Float](repeating: 0, count: q * f)
        for (j, column) in kept.enumerated() {
            let embedded = quantileEmbed(support: trainRows.map { Float($0[column]) }, query: rows.map { Float($0[column]) })
            for i in 0..<s { supportX[i * f + j] = embedded.support[i] }
            for i in 0..<q { queryX[i * f + j] = embedded.query[i] }
        }

        // Tokens: [label ‖ columns] per row. Labelled rows carry their class embedding, query rows the mask embedding.
        func xTokens(_ values: [Float], _ count: Int) -> MLXArray {
            net.xEmbedding.linear(MLXArray(values, [count, f, 1]).asType(.float16))
        }
        let supportY = net.yEmbedding.classes(MLXArray(y, [s, 1]))
        let support = concatenated([supportY, xTokens(supportX, s)], axis: 1)
        let maskToken = net.yEmbedding.mask(MLXArray([Int32(0)], [1, 1]))

        // Keep each row-attention score tensor near 64M elements.
        let columnGroup = max(1, min(f + 1, (1 << 26) / max(1, config.heads * s * s)))
        var probabilities: [[Double]] = []
        var start = 0
        while start < q {
            // Query rows never attend each other, so chunks give the same answer as one pass.
            let end = min(start + 512, q)
            let count = end - start
            var query = concatenated([broadcast(maskToken, to: [count, 1, config.dim]),
                                      xTokens(Array(queryX[(start * f) ..< (end * f)]), count)], axis: 1)
            var supportState = support
            for layer in net.layers {
                (supportState, query) = layer(supportState, query, columnGroup: columnGroup)
                try params.checkCancelled()
            }
            let logits = net.finalLayer(net.finalLayerNorm(query[0..., 0])).asType(.float32)[0..., ..<classes.count]
            let p = softmax(logits, axis: -1)
            eval(p)
            let flat = p.asArray(Float.self)
            for r in 0..<count { probabilities.append((0..<classes.count).map { Double(flat[r * classes.count + $0]) }) }
            start = end
        }
        let predicted = probabilities.map { row in classes[row.indices.max { row[$0] < row[$1] }!] }
        return ["labels": predicted, "classes": classes, "probabilities": probabilities]
    }
}
