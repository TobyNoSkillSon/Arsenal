// SPDX-License-Identifier: MIT
// ULTRA 50g (mgalkin/ultra_50g), a knowledge-graph foundation model, ported from the repository's ultra/ package
// (models.py Ultra, RelNBFNet, EntityNBFNet; layers.py GeneralizedRelationalConv; tasks.py build_relation_graph).
// The request carries the graph itself as (head, relation, tail) triples with any names. As the maker's datasets do,
// every triple also gets its inverse (tail, relation^-1, head). ULTRA then
//   1. builds the graph of relations (two relations are linked head-head, tail-tail, head-tail or tail-head when
//      some entity plays those roles in both) and runs RelNBFNet on it from the query relation: relation features
//      relative to the query, with nothing learned per graph;
//   2. runs EntityNBFNet from the query head over the entity graph with those relation features and scores every
//      entity as the missing tail (a logit; the model was trained with binary cross-entropy, so sigmoid is its
//      probability).
// A head query (?, relation, tail) becomes a tail query from `tail` with the inverse relation, as the maker's
// negative_sample_to_tail does (the relation graph still runs from the relation as given, as Ultra.forward does). Message passing computes what the maker's rspmm kernel computes (sum of distmult
// messages plus the boundary condition) with MLX gathers and scatter-adds. Weights and messages are float16; the sums over neighbours and the layer
// update before LayerNorm run in float32 so a node with many neighbours cannot overflow float16.
import Foundation
import MLX
import MLXNN

let modelKind = "local.graph"
let modelOps = ["predict_links"]

let maxTriples = 1_000_000
/// The relation graph has up to 4 x (2 x relations)^2 edges, so distinct relations are capped; entities likewise.
let maxRelations = 4096, maxEntities = 1_000_000

final class Model {
    let net: UltraNet

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        func dims(_ key: String) throws -> (Int, Int) {
            guard let cfg = config[key] as? [String: Any], let input = cfg["input_dim"] as? Int, let hidden = cfg["hidden_dims"] as? [Int],
                  Set(hidden) == [input], (cfg["message_func"] as? String) == "distmult", (cfg["aggregate_func"] as? String) == "sum",
                  (cfg["short_cut"] as? Bool) == true, (cfg["layer_norm"] as? Bool) == true
            else { throw RunnerError.failed("Only ULTRA checkpoints with distmult messages, sum aggregation, shortcut and LayerNorm are supported.") }
            return (input, hidden.count)
        }
        let (relDim, relLayers) = try dims("relation_model_cfg")
        let (entDim, entLayers) = try dims("entity_model_cfg")
        guard relDim == entDim else { throw RunnerError.failed("ULTRA relation and entity widths differ.") }
        net = UltraNet(dim: relDim, relationLayers: relLayers, entityLayers: entLayers)
        // nn.Sequential indices: relation_projection.{0,2} and mlp.{0,2} (1 is the ReLU) become .0 and .1 here.
        try Weights.apply(try Weights.load(weightsDir), to: net) { key in
            guard key.hasPrefix("model.") else { return nil }
            return String(key.dropFirst(6)).replacingOccurrences(of: "relation_projection.2.", with: "relation_projection.1.")
                .replacingOccurrences(of: "mlp.2.", with: "mlp.1.")
        }
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard let rows = params.raw["triples"] as? [Any], !rows.isEmpty else {
            throw RunnerError.invalid("triples must be a nonempty list of [head, relation, tail] lists.")
        }
        guard rows.count <= maxTriples else { throw RunnerError.invalid("triples holds \(rows.count) triples; at most \(maxTriples).") }
        var entities: [String: Int] = [:], entityNames: [String] = []
        var relations: [String: Int] = [:], relationNames: [String] = []
        func id(_ name: String, _ table: inout [String: Int], _ names: inout [String]) -> Int {
            if let i = table[name] { return i }
            table[name] = names.count; names.append(name); return names.count - 1
        }
        func name(_ value: Any) -> String? {
            if let s = value as? String { return s.isEmpty ? nil : s }
            if let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() { return n.stringValue }
            return nil
        }
        var heads: [Int] = [], tails: [Int] = [], types: [Int] = []
        var known = Set<[Int]>()
        for (i, row) in rows.enumerated() {
            guard let triple = row as? [Any], triple.count == 3, let h = name(triple[0]), let r = name(triple[1]), let t = name(triple[2]) else {
                throw RunnerError.invalid("triples[\(i)] must be [head, relation, tail]: three nonempty names (strings or numbers).")
            }
            let hi = id(h, &entities, &entityNames), ri = id(r, &relations, &relationNames), ti = id(t, &entities, &entityNames)
            guard relationNames.count <= maxRelations else {
                throw RunnerError.invalid("triples hold more than \(maxRelations) distinct relations; at most \(maxRelations).")
            }
            guard entityNames.count <= maxEntities else {
                throw RunnerError.invalid("triples hold more than \(maxEntities) distinct entities; at most \(maxEntities).")
            }
            if known.insert([hi, ri, ti]).inserted { heads.append(hi); types.append(ri); tails.append(ti) }
        }
        let relationName = try params.string("relation")
        guard let relation = relations[relationName] else {
            throw RunnerError.invalid("relation \"\(relationName)\" does not appear in triples.")
        }
        let headName = params.raw["head"].flatMap(name), tailName = params.raw["tail"].flatMap(name)
        guard (headName == nil) != (tailName == nil) else {
            throw RunnerError.invalid("Give head (to find missing tails) or tail (to find missing heads), not both.")
        }
        let given = headName ?? tailName!
        guard let anchor = entities[given] else { throw RunnerError.invalid("entity \"\(given)\" does not appear in triples.") }
        let predictTails = headName != nil
        let numRelations = relationNames.count
        let queryRelation = predictTails ? relation : relation + numRelations

        // The fact graph with inverse edges, as the maker's datasets build it.
        let src = heads + tails, dst = tails + heads
        let edgeType = types + types.map { $0 + numRelations }
        let scores = try net(
            numEntities: entityNames.count, numRelations: 2 * numRelations, src: src, dst: dst, edgeType: edgeType,
            anchor: anchor, relation: relation, queryRelation: queryRelation, params: params)
        let values = scores.asType(.float32).asArray(Float.self)

        let excludeKnown = params.bool("exclude_known", default: true)
        let topK = max(1, params.int("top_k", default: 10))
        var ranked: [[String: Any]] = []
        for i in values.indices.sorted(by: { values[$0] > values[$1] }) {
            let isKnown = predictTails ? known.contains([anchor, relation, i]) : known.contains([i, relation, anchor])
            if excludeKnown && isKnown { continue }
            ranked.append([
                "entity": entityNames[i], "score": Double(values[i]), "probability": Double(1 / (1 + exp(-values[i]))), "known": isKnown,
            ])
            if ranked.count == topK { break }
        }
        return [
            "predictions": ranked, "predicting": predictTails ? "tail" : "head",
            "entities": entityNames.count, "relations": numRelations, "triples": heads.count,
        ]
    }
}

// MARK: network (names follow the checkpoint, prefix "model." removed)

final class UltraNet: Module {
    @ModuleInfo(key: "relation_model") var relationModel: RelationModel
    @ModuleInfo(key: "entity_model") var entityModel: EntityModel

    init(dim: Int, relationLayers: Int, entityLayers: Int) {
        _relationModel.wrappedValue = RelationModel(dim: dim, layers: relationLayers)
        _entityModel.wrappedValue = EntityModel(dim: dim, layers: entityLayers)
    }

    /// Logits [numEntities] for every entity as the tail of (anchor, queryRelation, ?). As Ultra.forward does, the
    /// relation graph runs from the request's relation as given (`relation`, before a head query becomes a tail query
    /// with the inverse), and the entity model's query is row `queryRelation` of its output.
    func callAsFunction(
        numEntities: Int, numRelations: Int, src: [Int], dst: [Int], edgeType: [Int], anchor: Int, relation: Int, queryRelation: Int,
        params: Params
    ) throws -> MLXArray {
        let relationGraph = buildRelationGraph(numEntities: numEntities, numRelations: numRelations, src: src, dst: dst, edgeType: edgeType)
        let relationFeatures = try relationModel(graph: relationGraph, numRelations: numRelations, query: relation, params: params)
        let graph = Graph(src: src, dst: dst, type: edgeType)
        return try entityModel(graph: graph, numEntities: numEntities, relationFeatures: relationFeatures,
                               anchor: anchor, queryRelation: queryRelation, params: params)
    }
}

struct Graph {
    let src: MLXArray, dst: MLXArray, type: MLXArray
    init(src: [Int], dst: [Int], type: [Int]) {
        self.src = MLXArray(src.map(Int32.init)); self.dst = MLXArray(dst.map(Int32.init)); self.type = MLXArray(type.map(Int32.init))
    }
}

/// tasks.build_relation_graph: an edge r1 -> r2 of type 0 (head-head), 1 (tail-tail), 2 (head-tail: an entity is a
/// head of r1 and a tail of r2) or 3 (tail-head) whenever some entity plays those roles. Only the edges matter: the
/// maker's degree-normalised values are dropped (its message passing uses edge weight 1).
func buildRelationGraph(numEntities: Int, numRelations: Int, src: [Int], dst: [Int], edgeType: [Int]) -> Graph {
    var headRels = [Set<Int>](repeating: [], count: numEntities), tailRels = [Set<Int>](repeating: [], count: numEntities)
    for i in src.indices {
        headRels[src[i]].insert(edgeType[i]); tailRels[dst[i]].insert(edgeType[i])
    }
    var edges = [Set<Int>](repeating: [], count: 4)  // r1 * numRelations + r2
    for e in 0 ..< numEntities {
        let h = headRels[e], t = tailRels[e]
        for a in h { for b in h { edges[0].insert(a * numRelations + b) } }
        for a in t { for b in t { edges[1].insert(a * numRelations + b) } }
        for a in h { for b in t { edges[2].insert(a * numRelations + b) } }
        for a in t { for b in h { edges[3].insert(a * numRelations + b) } }
    }
    var s: [Int] = [], d: [Int] = [], k: [Int] = []
    for (kind, set) in edges.enumerated() {
        for code in set.sorted() { s.append(code / numRelations); d.append(code % numRelations); k.append(kind) }
    }
    return Graph(src: s, dst: d, type: k)
}

/// One GeneralizedRelationalConv step with distmult messages and sum aggregation, as the maker's fused rspmm kernel
/// computes it (ultra/rspmm/source/rspmm.cpp, the path its training used): for every edge (u, v, r) of edge_index,
/// update[u] += input[v] * relation[r], plus boundary[u]; output = relu(LayerNorm(linear([input, update]))).
/// (PyG's generic message/aggregate path, which the maker uses only for gradient visualisation, sums the other way.)
class RelationalConv: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm

    init(dim: Int) {
        _linear.wrappedValue = Linear(2 * dim, dim)
        _layerNorm.wrappedValue = LayerNorm(dimensions: dim)  // nn.LayerNorm default eps 1e-5
    }

    func step(_ input: MLXArray, relation: MLXArray, boundary: MLXArray, graph: Graph) -> MLXArray {
        // rspmm: row = edge_index[0] gathers from col = edge_index[1].
        let messages = input.take(graph.dst, axis: 0) * relation.take(graph.type, axis: 0)
        let update = boundary.asType(.float32).at[graph.src].add(messages.asType(.float32))
        let w = linear.weight.asType(.float32), b = linear.bias!.asType(.float32)
        let z = matmul(concatenated([input.asType(.float32), update], axis: -1), w.transposed()) + b
        let normed = MLXFast.layerNorm(z, weight: layerNorm.weight!.asType(.float32), bias: layerNorm.bias!.asType(.float32), eps: 1e-5)
        return relu(normed).asType(input.dtype)
    }
}

final class RelationModel: Module {
    @ModuleInfo(key: "layers") var layers: [RelationalConvWithEmbedding]
    init(dim: Int, layers count: Int) { _layers.wrappedValue = (0 ..< count).map { _ in RelationalConvWithEmbedding(dim: dim) } }

    /// RelNBFNet.bellmanford from the query relation: the query node starts as all ones; returns [numRelations, dim].
    func callAsFunction(graph: Graph, numRelations: Int, query: Int, params: Params) throws -> MLXArray {
        let dtype = layers[0].relation.weight.dtype
        let dim = layers[0].relation.weight.dim(1)
        let isQuery = (MLXArray(Int32(0) ..< Int32(numRelations)) .== Int32(query)).expandedDimensions(axis: 1)
        let boundary = MLX.where(isQuery, MLXArray.ones([numRelations, dim], dtype: dtype), MLXArray.zeros([numRelations, dim], dtype: dtype))
        var x = boundary
        for layer in layers {
            x = layer.step(x, relation: layer.relation.weight, boundary: boundary, graph: graph) + x
            try params.checkCancelled()
        }
        return x
    }
}

/// GeneralizedRelationalConv for the relation graph (an independent relation embedding per layer).
final class RelationalConvWithEmbedding: RelationalConv {
    @ModuleInfo(key: "relation") var relation: Embedding
    override init(dim: Int) {
        _relation.wrappedValue = Embedding(embeddingCount: 4, dimensions: dim)
        super.init(dim: dim)
    }
}

/// GeneralizedRelationalConv for the entity graph (project_relations=True): this layer's relation features are an
/// MLP of the relation graph's output.
final class RelationalConvWithProjection: RelationalConv {
    @ModuleInfo(key: "relation_projection") var relationProjection: [Linear]
    override init(dim: Int) {
        _relationProjection.wrappedValue = [Linear(dim, dim), Linear(dim, dim)]
        super.init(dim: dim)
    }
    func project(_ r: MLXArray) -> MLXArray { relationProjection[1](relu(relationProjection[0](r))) }
}

final class EntityModel: Module {
    @ModuleInfo(key: "layers") var layers: [RelationalConvWithProjection]
    @ModuleInfo(key: "mlp") var mlp: [Linear]

    init(dim: Int, layers count: Int) {
        _layers.wrappedValue = (0 ..< count).map { _ in RelationalConvWithProjection(dim: dim) }
        _mlp.wrappedValue = [Linear(2 * dim, 2 * dim), Linear(2 * dim, 1)]
    }

    /// EntityNBFNet.bellmanford from the anchor entity, then the MLP on [final state, query] for every entity.
    func callAsFunction(
        graph: Graph, numEntities: Int, relationFeatures: MLXArray, anchor: Int, queryRelation: Int, params: Params
    ) throws -> MLXArray {
        let query = relationFeatures[queryRelation]  // [dim]
        let dim = query.dim(0)
        let isAnchor = (MLXArray(Int32(0) ..< Int32(numEntities)) .== Int32(anchor)).expandedDimensions(axis: 1)
        let boundary = MLX.where(isAnchor, broadcast(query.expandedDimensions(axis: 0), to: [numEntities, dim]),
                                 MLXArray.zeros([numEntities, dim], dtype: query.dtype))
        var x = boundary
        for layer in layers {
            x = layer.step(x, relation: layer.project(relationFeatures), boundary: boundary, graph: graph) + x
            try params.checkCancelled()
        }
        let feature = concatenated([x, broadcast(query.expandedDimensions(axis: 0), to: [numEntities, dim])], axis: -1)
        return mlp[1](relu(mlp[0](feature))).squeezed(axis: -1)
    }
}
