// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import Testing
@testable import GeometryWorker

/// DA3 Small checkpoints carry unported camera heads (cam_enc/cam_dec) whose names Load maps to nil. Get must not
/// pack those matrices: Load drops the weight but cannot drop its stored scales/biases ("Orphan affine companions").
@Suite(.serialized)
struct DA3StoredTierTests {
    final class Attention: Module {
        @ModuleInfo(key: "qkv") var qkv: Linear
        override init() { _qkv.wrappedValue = Linear(64, 192); super.init() }
    }
    final class Block: Module {
        @ModuleInfo(key: "attn") var attn: Attention
        override init() { _attn.wrappedValue = Attention(); super.init() }
    }
    final class Backbone: Module {
        @ModuleInfo(key: "blocks") var blocks: [Block]
        override init() { _blocks.wrappedValue = [Block()]; super.init() }
    }
    /// The mapped names of one DA3 backbone matrix: backbone.blocks.0.attn.qkv.{weight,bias}.
    final class Graph: Module {
        @ModuleInfo(key: "backbone") var backbone: Backbone
        override init() { _backbone.wrappedValue = Backbone(); super.init() }
    }

    @Test func storageEligibilitySkipsDroppedNames() {
        let eligible = GeometryWeights.storageEligible(moge: false)
        #expect(eligible("model.backbone.pretrained.blocks.0.attn.qkv.weight", [192, 64]))
        #expect(!eligible("model.cam_enc.fc.weight", [128, 64]))
        #expect(!eligible("model.cam_dec.mlp.0.weight", [64, 64]))
        #expect(!eligible("model.gs_head.proj.weight", [64, 64]))
    }

    @Test(.heavy, .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] != nil))
    func int8SnapshotFromSourceLayoutNamesLoads() throws {
        NativeRuntime.prepare()
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("geometry-da3-tier-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let source = temporary.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        func half(_ shape: [Int]) -> MLXArray { MLXRandom.uniform(Float(-0.1)..<Float(0.1), shape, key: MLXRandom.key(UInt64(shape.reduce(1, *)))).asType(.float16) }
        try MLX.save(
            arrays: [
                "model.backbone.pretrained.blocks.0.attn.qkv.weight": half([192, 64]),
                "model.backbone.pretrained.blocks.0.attn.qkv.bias": half([192]),
                "model.cam_enc.fc.weight": half([128, 64]), "model.cam_enc.fc.bias": half([128]),
                "model.cam_dec.mlp.0.weight": half([64, 64])
            ], url: source.appendingPathComponent("model.safetensors"))
        func convert(_ eligible: @escaping (String, [Int]) -> Bool, into name: String) throws -> URL {
            let destination = temporary.appendingPathComponent(name)
            _ = try NativeConversion(recipe: "cast:geometry-da3-v1", eligible: eligible).run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(destination.path), "dtype": "int8", "group_size": 64,
                    "recipe": "cast:geometry-da3-v1"
                ], cancellation: NativeCancellation())
            return destination.appendingPathComponent("model.safetensors")
        }
        func load(_ archive: URL) throws -> Int {
            let graph = Graph()
            try NativeParameters.initializeRandom(graph)
            return try NativePrecisionTier("int8").load(
                graph, archives: [archive], map: { GeometryWeights.load($0, $1, moge: false) }, eligible: GeometryWeights.eligible
            ).quantizedTensors
        }
        // The production Get policy packs only the mapped backbone matrix; the camera heads stay half and are dropped.
        #expect(try load(convert(GeometryWeights.storageEligible(moge: false), into: "int8")) == 1)
        // The former policy packed the camera heads too, and Load refused their orphan companions.
        #expect(throws: (any Error).self) { try load(convert(GeometryWeights.eligible, into: "int8-unfiltered")) }
    }
}
