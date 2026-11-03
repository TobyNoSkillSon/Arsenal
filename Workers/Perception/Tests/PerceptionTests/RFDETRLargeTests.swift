// SPDX-License-Identifier: MIT
import CryptoKit
import Foundation
import ImageIO
import IntegrationKit
import MLX
import MLXNN
import MLXRandom
import ArsenalWire
import Testing
@testable import MLXRFDETR

/// RF-DETR Large (2026, Roboflow/rf-detr-large) runs the same graph as Nano at 704 px with four decoder layers.
/// The runner used to hard-code Nano's 384 px and two layers; it now reads them from config.json.
struct RFDETRLargeTests {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    private static func config(_ name: String) throws -> JSONObject {
        let url = try #require(Bundle.module.url(forResource: name, withExtension: "json", subdirectory: "Fixtures"))
        return try JSONDecoder().decode(JSONObject.self, from: Data(contentsOf: url))
    }

    @Test func shapeFollowsEachPinnedConfig() throws {
        let nano = NativeRFDETR.architectureShape(try Self.config("rf-detr-nano-config"))
        #expect(nano == .init(resolution: 384, patchSize: 16, embedDim: 384, depth: 12, heads: 6, windows: 2, decoderLayers: 2, queries: 300))
        let large = NativeRFDETR.architectureShape(try Self.config("rf-detr-large-config"))
        #expect(large == .init(resolution: 704, patchSize: 16, embedDim: 384, depth: 12, heads: 6, windows: 2, decoderLayers: 4, queries: 300))
        #expect(NativeRFDETR.backboneLayout(try Self.config("rf-detr-large-config")).features == [2, 5, 8, 11])
        // A config without these keys still loads as Nano did before.
        #expect(NativeRFDETR.architectureShape([:]) == nano)
    }

    /// The maker's selection is torch.topk: highest score first, equal scores in position order. 104 of the 299
    /// adjacent pairs among the cats photo's top 300 are exact FP16 ties, so this order decides the detections.
    @Test func topKSelectionBreaksTiesByPositionLikeTorch() {
        #expect(Transformer.topKIndices([1, 3, 3, 2, 3, .nan, 0], k: 5) == [1, 2, 4, 3, 0])
        #expect(Transformer.topKIndices([0.5, 0.5, 0.5], k: 2) == [0, 1])
        #expect(Transformer.topKIndices([], k: 3).isEmpty)
    }

    /// The header-only check (pinned safetensors header against the worker's full-size parameter inventory, through
    /// gate/header-map.json) passed, and the map it used is the committed one.
    @Test func headerCheckPassedAgainstTheCommittedMap() throws {
        let gate = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("models/rf-detr-large/gate")
        let report = try JSONDecoder().decode(JSONObject.self, from: Data(contentsOf: gate.appendingPathComponent("header-check.json")))
        #expect(report["status"]?.string == "passed-with-planned-get-casts")
        #expect(report["source_tensors"]?.number == 509 && report["runtime_parameters"]?.number == 508 && report["planned_get_casts"]?.number == 508)
        #expect(report["errors"]?.array?.isEmpty == true && report["tensor_payload_bytes_downloaded"]?.number == 0)
        let map = try Data(contentsOf: gate.appendingPathComponent("header-map.json"))
        let digest = SHA256.hash(data: map).map { String(format: "%02x", $0) }.joined()
        #expect(report["mapping_sha256"]?.string == digest)
    }

    /// A batch selects each image's own top queries: positions are within that image's row.
    @Test func batchedSelectionKeepsEachImageApart() {
        let scores: [Float] = [1, 3, 2, 3, 5, 5, 0, -1]
        #expect(Transformer.batchedTopK(scores, batches: 2, k: 2) == [1, 3, 0, 1])
        #expect(Transformer.batchedTopK(Array(scores[4...]), batches: 1, k: 2) == [0, 1])
    }

    /// Weight-free checks on a tiny random keypoint transformer: an injected selection reaches the keypoint branch,
    /// and a batch of two images selects and decodes each image as it would alone. Heavy only because the deformable
    /// attention is a Metal kernel (GPU only); 16 positions and 4 queries.
    @Test(.heavy) func keypointBranchUsesTheInjectedSelectionAndBatchesMatchSingleImages() throws {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
        do {
            MLXRandom.seed(7)
            let config = RFDETRConfig(
                hiddenDim: 32, decLayers: 1, saNheads: 2, caNheads: 2, dimFeedforward: 64, numQueries: 4, groupDetr: 1, numClasses: 3,
                useGroupposeKeypoints: true, numKeypointsPerClass: [2, 2, 2])
            let transformer = Transformer(config: config)
            let bbox = DecoderMLP(inputDim: 32, hiddenDim: 32, outputDim: 4, numLayers: 3)
            let memory = MLXRandom.normal([2, 16, 32]), keypoints = MLXRandom.normal([2, 16, 32])
            let queryFeat = MLXRandom.normal([4, 32]) * 0.1, refpoint = MLXRandom.normal([4, 4]) * 0.1
            func run(_ memory: MLXArray, _ keypoints: MLXArray, _ selection: MLXArray?) -> MLXArray {
                transformer.callWithKeypoints(
                    memory, keypointMemory: keypoints, spatialShapes: [(4, 4)], queryFeat: queryFeat, refpointEmbed: refpoint,
                    bboxEmbed: bbox, selection: selection
                ).1.asType(.float32)
            }
            let computed = run(memory, keypoints, nil)
            let scores = transformer.encOutClassEmbed[0](transformer.encOutputNorm[0](transformer.encOutput[0](memory))).max(axis: -1)
            let host = Transformer.batchedTopK(scores.asType(.float32).asArray(Float.self), batches: 2, k: 4)
            #expect(allClose(run(memory, keypoints, MLXArray(host, [2, 4])), computed).item(Bool.self))
            let other = MLXArray(host.map { (Int32(15) - $0) }, [2, 4])
            #expect(!allClose(run(memory, keypoints, other), computed).item(Bool.self), "keypoint branch ignored the injected selection")
            let alone = run(memory[1..<2], keypoints[1..<2], nil)
            #expect(allClose(alone, computed[1..<2], atol: 1e-5).item(Bool.self), "second image differs inside the batch")
        }
    }

    /// Review (2 Nov): a random-weight load takes its sizes from the request, so Large's no-weights fixture carries
    /// Large's sizes; a full-inventory smoke then builds the 704 px, four-layer graph rather than Nano's.
    @Test func noWeightsFixtureCarriesLargeSizes() throws {
        let url = Self.repo.appendingPathComponent("models/rf-detr-large/gate/no-weights.json")
        let fixture = try JSONDecoder().decode(JSONObject.self, from: Data(contentsOf: url))
        let config = try #require(fixture["config"]?.object)
        #expect(NativeRFDETR.architectureShape(config) == NativeRFDETR.architectureShape(try Self.config("rf-detr-large-config")))
    }

    /// Preprocessing matches the checkpoints' own processor (transformers RfDetrImageProcessor: scale to [0, 1], then
    /// PyTorch bilinear with half-pixel centres and no antialias, then ImageNet normalisation), captured once on two PNGs
    /// (an odd-sized upscale with noise, a downscale with hard edges) at Large's 704 and Nano's 384 px.
    @Test func preprocessingMatchesTheMakerProcessor() throws {
        let fixtures = try #require(Bundle.module.url(forResource: "rf-detr-preprocess-golden", withExtension: "json", subdirectory: "Fixtures"))
        let golden = try #require(JSONSerialization.jsonObject(with: Data(contentsOf: fixtures)) as? [String: Any])
        let cases = try #require(golden["cases"] as? [String: [String: Any]])
        #expect(cases.count == 4)
        for (name, entry) in cases {
            let file = try #require(entry["file"] as? String), size = try #require(entry["size"] as? Int)
            let url = try #require(Bundle.module.url(forResource: (file as NSString).deletingPathExtension, withExtension: "png", subdirectory: "Fixtures"))
            let source = try #require(CGImageSourceCreateWithURL(url as CFURL, nil))
            let image = try #require(CGImageSourceCreateImageAtIndex(source, 0, nil))
            let rgb = try RFDETRPreprocess.rgb(image)
            let pixels = RFDETRPreprocess.pixels(rgb.bytes, width: rgb.width, height: rgb.height, size: size)
            #expect(pixels.count == size * size * 3, "\(name)")
            let points = (entry["samples"] as? [[Any]] ?? []) + (entry["corners"] as? [[Any]] ?? [])
            #expect(points.count == 68, "\(name)")
            var worst: Float = 0
            for point in points {
                let y = try #require(point[0] as? Int), x = try #require(point[1] as? Int), values = try #require(point[2] as? [Double])
                for c in 0..<3 { worst = max(worst, abs(pixels[(y * size + x) * 3 + c] - Float(values[c]))) }
            }
            #expect(worst <= 1e-5, "\(name): largest difference \(worst)")
            let means = try #require(entry["channel_mean"] as? [Double])
            for c in 0..<3 {
                let mean = stride(from: c, to: pixels.count, by: 3).reduce(0.0) { $0 + Double(pixels[$1]) } / Double(size * size)
                #expect(abs(mean - means[c]) <= 1e-5, "\(name) channel \(c)")
            }
        }
    }

    /// The golden binds the fixture, the maker's detections and the frozen tolerances (no weights needed).
    @Test func goldenIsComplete() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().deletingLastPathComponent()
        let data = try Data(contentsOf: root.appendingPathComponent("models/rf-detr-large/gate/golden/golden.json"))
        let golden = try JSONDecoder().decode(JSONObject.self, from: data)
        #expect(golden["fixture"]?.object?["sha256"]?.string == "dea9e7ef97386345f7cff32f9055da4982da5471c48d575146c796ab4563b04e")
        let maker = try #require(golden["maker"]?.object?["detections"]?.array)
        #expect(maker.count == 5)
        #expect(maker.compactMap { $0.object?["label"]?.string } == ["cat", "remote", "cat", "remote", "couch"])
        let tolerance = try #require(golden["tolerance"]?.object)
        #expect(Set(tolerance.keys) == ["component_relative_l2", "score_abs", "box_abs"])
        #expect(golden["arsenal"]?.object?["detections"]?.array?.count == 5)
    }

    /// Per-component parity against the maker's transformers code (FP16, MPS), captured once in
    /// lab/evidence/rf-detr-large. Each component consumes the maker's own input tensor.
    /// `ARSENAL_RFDETR_LARGE_EVIDENCE=<that dir> scripts/test.sh --heavy --filter RFDETRLargeTests`
    @Test(.heavy) func matchesMakerPerComponent() throws {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
        let evidence = try #require(ProcessInfo.processInfo.environment["ARSENAL_RFDETR_LARGE_EVIDENCE"].map(URL.init(fileURLWithPath:)))
        // A plain FP16 cast of the pinned checkpoint from the Hugging Face cache, made here and removed afterwards, so no
        // second copy of the weights stays on disk.
        let source = URL(
            fileURLWithPath: ProcessInfo.processInfo.environment["ARSENAL_RFDETR_LARGE_SOURCE"]
                ?? NSHomeDirectory() + "/.cache/huggingface/hub/models--Roboflow--rf-detr-large/snapshots/f62f7dd5252b61097cbace33886045816dadbde9")
        let half = FileManager.default.temporaryDirectory.appendingPathComponent("rf-detr-large-fp16-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: half, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: half) }
        let original = try MLX.loadArrays(url: source.appendingPathComponent("model.safetensors"))
        try MLX.save(arrays: original.mapValues { $0.asType(.float16) }, url: half.appendingPathComponent("model.safetensors"))
        try Data(contentsOf: source.appendingPathComponent("config.json")).write(to: half.appendingPathComponent("config.json")) // the cache holds symlinks
        let native = NativeRFDETR()
        try native.load(
            NativeLoadContext([
                "weights_dir": .string(half.path), "model_dir": .string(evidence.path), "dtype": "fp16", "config": [:]
            ]), cancellation: NativeCancellation())
        let model = try #require(native.model)
        let reference = try MLX.loadArrays(url: evidence.appendingPathComponent("reference.safetensors"))
        func nhwc(_ key: String) throws -> MLXArray { try #require(reference[key]).transposed(0, 2, 3, 1) }
        func relL2(_ a: MLXArray, _ b: MLXArray) -> Float {
            let x = a.asType(.float32), y = b.asType(.float32)
            return (sqrt(sum((x - y) * (x - y))) / sqrt(sum(y * y))).item(Float.self)
        }
        var report: [String: Float] = [:]
        // Counts and shapes first.
        let features = model.backbone(try nhwc("pixel_values"))
        #expect(features.count == 4)
        for (index, feature) in features.enumerated() {
            let expected = try nhwc("feature_\(index)")
            #expect(feature.shape == expected.shape)
            report["backbone.\(index)"] = relL2(feature, expected)
        }
        let memories = model.projector(try (0..<4).map { try nhwc("feature_\($0)") })
        #expect(memories.count == 1 && memories[0].shape == [1, 44, 44, 256])
        report["projector"] = relL2(memories[0], try nhwc("projector"))
        let makerTopK = try #require(reference["enc.topk"])
        let heads = model.detect(memories: [try nhwc("projector")], features: [], imageSize: (704, 704), selection: makerTopK)
        let logits = try #require(heads["pred_logits"]), boxes = try #require(heads["pred_boxes"])
        #expect(logits.shape == [1, 300, 91] && boxes.shape == [1, 300, 4])
        report["decoder.logits"] = relL2(logits, try #require(reference["logits"]))
        // All 300 boxes include background queries (maker score < 0.02) whose boxes move with half-precision rounding:
        // the maker's own BF16 run differs from its FP16 run by relative L2 0.26 there. Confident queries are held tight.
        report["decoder.boxes_all"] = relL2(boxes, try #require(reference["boxes"]))
        let confident = MLXArray(
            sigmoid(try #require(reference["logits"]).asType(.float32)).max(axis: -1).asArray(Float.self).enumerated().filter { $0.element >= 0.1 }
                .map { Int32($0.offset) })
        report["decoder.boxes_confident"] = relL2(boxes[0, confident], try #require(reference["boxes"])[0, confident])
        let whole = model(try nhwc("pixel_values"))
        // The full path selects its own top 300; one-ULP score differences reorder near-ties, and each rank has its
        // own learned query, so the whole run is compared order-free: detections matched by label and nearest box.
        let ours = Self.detections(logits: whole["pred_logits"]!, boxes: whole["pred_boxes"]!)
        let maker = Self.detections(logits: try #require(reference["logits"]), boxes: try #require(reference["boxes"]))
        #expect(ours.map(\.label).sorted() == maker.map(\.label).sorted())
        var scoreGap: Float = 0, boxGap: Float = 0
        for expected in maker {
            let match = try #require(ours.filter { $0.label == expected.label }.min { Self.gap($0.box, expected.box) < Self.gap($1.box, expected.box) })
            scoreGap = max(scoreGap, abs(match.score - expected.score)); boxGap = max(boxGap, Self.gap(match.box, expected.box))
        }
        report["end_to_end.max_score_abs"] = scoreGap
        report["end_to_end.max_box_abs"] = boxGap
        let memoryFlat = try nhwc("projector").reshaped([1, -1, 256])
        let transformer = model.transformer
        let encScores = transformer.encOutClassEmbed[0](transformer.encOutputNorm[0](transformer.encOutput[0](memoryFlat))).max(axis: -1)
        report["encoder.scores"] = relL2(encScores, try #require(reference["enc.scores"]))
        let makerSelection = try #require(reference["enc.topk"]).asArray(Int32.self).map(Int.init)
        let selection = Transformer.topKIndices(encScores.asType(.float32).asArray(Float.self), k: 300)
        let sameRank = zip(selection, makerSelection).filter { $0 == $1 }.count
        print("RFDETR-LARGE-SELECTION same_rank=\(sameRank) same_set=\(Set(selection).intersection(makerSelection).count)")
        // Same 300 positions, except where the 300th score is itself tied: then either side may hold any of the tied ones.
        let ourScores = encScores.asType(.float32).asArray(Float.self), theirScores = try #require(reference["enc.scores"]).asType(.float32).asArray(Float.self)
        let ourCut = selection.map { ourScores[$0] }.min()!, theirCut = makerSelection.map { theirScores[$0] }.min()!
        let onlyOurs = Set(selection).subtracting(makerSelection), onlyTheirs = Set(makerSelection).subtracting(selection)
        #expect(onlyOurs.count == onlyTheirs.count)
        #expect(onlyOurs.allSatisfy { ourScores[$0] == ourCut } && onlyTheirs.allSatisfy { theirScores[$0] == theirCut }, "selection differs above the cut")
        try MLX.save(
            arrays: [
                "enc.scores": encScores,
                "heads.logits": logits, "heads.boxes": boxes, "whole.logits": whole["pred_logits"]!, "whole.boxes": whole["pred_boxes"]!
            ], url: evidence.appendingPathComponent("arsenal.safetensors"))
        print("RFDETR-LARGE-PARITY " + report.sorted { $0.key < $1.key }.map { "\($0.key)=\($0.value)" }.joined(separator: " "))
        // Bounds and their history: models/rf-detr-large/gate/golden/golden.json.
        for (key, value) in report where !key.hasPrefix("end_to_end") && key != "decoder.boxes_all" {
            #expect(value <= 0.03, "\(key) relative L2 \(value)")
        }
        #expect(report["decoder.boxes_all"]! <= 0.1)
        #expect(report["end_to_end.max_score_abs"]! <= 0.02)
        #expect(report["end_to_end.max_box_abs"]! <= 0.01)
    }

    struct Detection { let label: Int; let score: Float; let box: [Float] }
    static func detections(logits: MLXArray, boxes: MLXArray, threshold: Float = 0.5) -> [Detection] {
        let scores = sigmoid(logits.asType(.float32)).asArray(Float.self), coordinates = boxes.asType(.float32).asArray(Float.self)
        let classes = logits.dim(-1)
        return scores.indices.filter { scores[$0] >= threshold }.map { index in
            Detection(label: index % classes, score: scores[index], box: Array(coordinates[(index / classes * 4)..<(index / classes * 4 + 4)]))
        }
    }
    static func gap(_ a: [Float], _ b: [Float]) -> Float { zip(a, b).map { abs($0 - $1) }.max() ?? .infinity }

    /// A minimal input-only ICC profile (device class 'scnr', RGB to XYZ by an A2B0 lut8 only), as some scanners and
    /// cameras embed: CoreGraphics reads it but cannot draw into it.
    static func inputOnlyProfile() -> Data {
        func be32(_ v: UInt32) -> [UInt8] { [UInt8(v >> 24), UInt8(v >> 16 & 0xFF), UInt8(v >> 8 & 0xFF), UInt8(v & 0xFF)] }
        func text(_ s: String) -> [UInt8] { Array(s.utf8) }
        let white = [0.9642, 1.0, 0.8249].flatMap { be32(UInt32(($0 * 65536).rounded())) }
        let ramp = [UInt8](0...255)
        var lut = text("mft1") + [0, 0, 0, 0, 3, 3, 2, 0]
        for row in 0..<3 { for column in 0..<3 { lut += be32(row == column ? 65536 : 0) } }
        lut += ramp + ramp + ramp
        for r in 0..<2 {
            for g in 0..<2 { for b in 0..<2 { lut += [UInt8(min(255, r * 200 + g * 40 + b * 15)), UInt8(r * 54 + g * 182 + b * 18), UInt8(min(255, b * 230 + g * 25))] } }
        }
        lut += ramp + ramp + ramp
        let tags: [(String, [UInt8])] = [
            ("desc", text("desc") + [0, 0, 0, 0] + be32(5) + text("scan\0") + [UInt8](repeating: 0, count: 78)),
            ("wtpt", text("XYZ ") + [0, 0, 0, 0] + white), ("A2B0", lut), ("cprt", text("text") + [0, 0, 0, 0] + text("none\0"))
        ]
        var table = be32(UInt32(tags.count)), body: [UInt8] = [], offset = 128 + 4 + 12 * tags.count
        for (signature, data) in tags {
            table += text(signature) + be32(UInt32(offset)) + be32(UInt32(data.count))
            let padded = data + [UInt8](repeating: 0, count: (4 - data.count % 4) % 4)
            body += padded; offset += padded.count
        }
        var header = be32(UInt32(128 + table.count + body.count)) + text("appl") + [2, 0x10, 0, 0] + text("scnrRGB XYZ ")
        header += [UInt8](repeating: 0, count: 12) + text("acspAPPL") + [UInt8](repeating: 0, count: 24) + white + [0, 0, 0, 0]
        return Data(header + [UInt8](repeating: 0, count: 128 - header.count) + table + body)
    }

    /// An image whose ICC profile cannot be a drawing target (input-only scanner and camera profiles) used to fail with a
    /// plain NSError (503). Its stored values are read unconverted, as PIL hands them to the processor; when no space
    /// can be drawn into at all the caller gets invalid_request (422).
    @Test func pixelsOfAnInputOnlyProfileAreTheStoredValues() throws {
        let scanner = try #require(CGColorSpace(iccData: Self.inputOnlyProfile() as CFData))
        #expect(scanner.model == .rgb && !scanner.supportsOutput)
        let bytes: [UInt8] = [250, 10, 20, 255, 30, 240, 40, 255, 50, 60, 230, 255, 128, 128, 128, 255]
        func image(_ space: CGColorSpace) throws -> CGImage {
            let provider = try #require(CGDataProvider(data: Data(bytes) as CFData))
            return try #require(
                CGImage(
                    width: 2, height: 2, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 8, space: space,
                    bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider,
                    decode: nil, shouldInterpolate: false, intent: .defaultIntent))
        }
        let stored: [UInt8] = [250, 10, 20, 30, 240, 40, 50, 60, 230, 128, 128, 128]
        #expect(try RFDETRPreprocess.rgb(image(scanner)).bytes == stored)
        let srgb = try #require(CGColorSpace(name: CGColorSpace.sRGB)), cmyk = try #require(CGColorSpace(name: CGColorSpace.genericCMYK))
        #expect(try RFDETRPreprocess.rgb(image(srgb)).bytes == stored)
        do {
            _ = try RFDETRPreprocess.rgb(image(srgb), spaces: [cmyk])
            Issue.record("A space CoreGraphics cannot draw into was accepted")
        } catch let failure as WorkerFailure { #expect(failure.code == "invalid_request") }
    }

}
