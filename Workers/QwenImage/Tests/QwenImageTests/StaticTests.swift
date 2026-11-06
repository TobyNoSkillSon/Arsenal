// SPDX-License-Identifier: MIT
// Static checks on small fixed tensors whose reference values are derived from diffusers' QwenImage21 code by hand.
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom
import Testing
@testable import QwenImageNative

@Suite struct ScheduleTests {
    @Test func baseScheduleShiftsAndEndsAtTheTerminal() throws {
        let c = SchedulerConfig() // the base checkpoint's scheduler_config.json values
        // mu = 0.5 + (4096 − 256)·(0.9 − 0.5)/(8192 − 256)
        #expect(abs(FlowSchedule.mu(imageTokens: 4096, c) - 0.6935483870967742) < 1e-12)
        let s = try FlowSchedule(c, steps: 40, imageTokens: 4096)
        #expect(s.sigmas.count == 41 && s.steps == 40)
        #expect(s.sigmas[0] == 1 && s.sigmas[40] == 0)
        #expect(abs(s.sigmas[39] - 0.02) < 1e-6)
        for i in 0..<40 { #expect(s.sigmas[i] > s.sigmas[i + 1]) }
        // Step 1 before the stretch: e^mu / (e^mu + (1/0.975 − 1)); then 1 − (1 − x)/scale, scale = (1 − last)/0.98.
        let e = exp(0.6935483870967742), first = e / (e + (1 / 0.975 - 1)), last = e / (e + 39)
        let expected = 1 - (1 - first) / ((1 - last) / 0.98)
        #expect(abs(Double(s.sigmas[1]) - expected) < 2e-6)
        #expect(abs(s.timestep(1) - Float(expected * 1000)) < 2e-3)
    }

    @Test func turboRunsItsSavedSigmasUnchanged() throws {
        var c = SchedulerConfig()
        c.dynamicShifting = false; c.shiftTerminal = nil
        c.sampleSigmas = [1.0, 0.978453, 0.95418, 0.926626, 0.89508, 0.845148, 0.704534, 0.414568]
        let s = try FlowSchedule(c, steps: 40, imageTokens: 4096)
        #expect(s.steps == 8)
        #expect(s.sigmas == c.sampleSigmas!.map(Float.init) + [0])
        #expect(s.delta(7) == -Float(0.414568))
    }

    @Test(.heavy) func schedulerConfigsParse() throws {
        let base = try SchedulerConfig(
            json: [
                "_class_name": "FlowMatchEulerDiscreteScheduler", "base_image_seq_len": 256, "base_shift": 0.5, "max_image_seq_len": 8192, "max_shift": 0.9,
                "shift": 1.0, "shift_terminal": 0.02, "use_dynamic_shifting": true, "time_shift_type": "exponential"
            ], pipeline: [:])
        #expect(base == SchedulerConfig())
        let turbo = try SchedulerConfig(
            json: ["_class_name": "FlowMatchEulerDiscreteScheduler", "shift_terminal": NSNull(), "use_dynamic_shifting": false],
            pipeline: ["sample_sigmas": [1.0, 0.5]])
        #expect(turbo.shiftTerminal == nil && turbo.sampleSigmas == [1.0, 0.5] && !turbo.dynamicShifting)
    }
}

@Suite struct LayoutTests {
    @Test(.heavy) func imageBlocksAreCentredOnZero() {
        let p = DiTRope.image(frame: 5, height: 4, width: 2)
        #expect(p.height == [-2, -2, -1, -1, 0, 0, 1, 1])
        #expect(p.width == [-1, 0, -1, 0, -1, 0, -1, 0])
        #expect(p.frame == Array(repeating: 5, count: 8))
        // Odd sides: H = 3 → −2, −1, 0.
        #expect(Set(DiTRope.image(frame: 0, height: 3, width: 1).height) == [-2, -1, 0])
    }

    /// QwenImage21Rope.forward walked by hand: text advances one position per token, an image block freezes the frame
    /// at the current position and advances it by max(h, w).
    @Test func layoutFollowsTheReferencePositionWalk() throws {
        let slots = [false, false] + Array(repeating: true, count: 4) + [false]
        let layout = try SequenceLayout(imageSlots: slots, conditions: [(4, 4)], target: (2, 2))
        #expect(layout.runs == [.text(range: 0..<2), .image(index: 0, height: 4, width: 4), .text(range: 6..<7), .target(height: 2, width: 2)])
        #expect(layout.prefixLength == 2 + 16 + 1)
        #expect(Array(layout.prefixPositions.frame.prefix(2)) == [0, 1])
        #expect(layout.prefixPositions.frame[2..<18].allSatisfy { $0 == 2 })
        #expect(layout.prefixPositions.frame[18] == 6 && layout.prefixPositions.height[18] == 6 && layout.prefixPositions.width[18] == 6)
        #expect(layout.targetPositions.frame == [7, 7, 7, 7])
        #expect(layout.targetPositions.height == [-1, -1, 0, 0] && layout.targetPositions.width == [-1, 0, -1, 0])
        #expect(layout.segments.map { [$0.start, $0.end] } == [[0, 2], [2, 18], [18, 19]])
        #expect(layout.segments.map(\.text) == [true, false, true])
    }

    @Test func mismatchedSlotsAreRejected() {
        #expect(throws: QwenImageError.self) { try SequenceLayout(imageSlots: [false, true, true, false], conditions: [(4, 4)], target: (2, 2)) }
        #expect(throws: QwenImageError.self) { try SequenceLayout(imageSlots: [false], conditions: [(4, 4)], target: (2, 2)) }
    }

    @Test func promptTemplatesMatchThePipeline() {
        let t2i = PromptLayout(prompt: "a cat", imageSlots: [])
        #expect(t2i.text == "<|im_start|>system\nComprehend and analyze the provided prompt.<|im_end|>\n<|im_start|>user\na cat<|im_end|>\n<|im_start|>assistant\n")
        let edit = PromptLayout(prompt: "swap them", imageSlots: [2, 1])
        #expect(
            edit.text
                == PromptLayout.systemTurn + "<|im_start|>user\n<image1><|vision_start|><|image_pad|><|image_pad|><|vision_end|> "
                + "<image2><|vision_start|><|image_pad|><|vision_end|>swap them<|im_end|>\n<|im_start|>assistant\n"
        )
        #expect(PromptLayout(prompt: "", imageSlots: []).text.contains("user\n <|im_end|>"))
        let bytes = ByteTokens(config: SmallConfigs.encoder)
        let ids = edit.tokens(bytes.encode, imagePad: 300)
        #expect(ids.filter { $0 == 300 }.count == 3)
    }

    @Test func calculateDimensionsRoundsToThirtyTwo() {
        #expect(ImageSizing.dimensions(area: 1024 * 1024, ratio: 1) == (1024, 1024))
        #expect(ImageSizing.dimensions(area: 1024 * 1024, ratio: 16.0 / 9) == (1376, 768))
        #expect(ImageSizing.dimensions(area: 1024 * 1024, ratio: 2.0 / 3) == (832, 1248))
        var r = GenerationRequest(prompt: "x")
        #expect(QwenImagePipeline.outputSize(r) == (1024, 1024))
        r.width = 1000; r.height = 700
        #expect(QwenImagePipeline.outputSize(r) == (992, 672))
    }

    @Test(.heavy) func mropePositionsAdvancePastImages() {
        var c = Qwen3VLConfig(); c.imageToken = 9
        // text text [image 4×6 patches → 2×3 merged] text
        let tokens: [Int32] = [1, 2, 9, 9, 9, 9, 9, 9, 3]
        let p = Qwen3VLEncoder.ropeIndex(tokens: tokens, grids: [VisionGrid(h: 4, w: 6)], config: c)
        #expect(p[0] == [0, 1, 2, 2, 2, 2, 2, 2, 5])
        #expect(p[1] == [0, 1, 2, 2, 2, 3, 3, 3, 5])
        #expect(p[2] == [0, 1, 2, 3, 4, 2, 3, 4, 5])
    }
}

@Suite struct TensorTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }
    func close(_ a: MLXArray, _ b: MLXArray, _ tolerance: Float = 1e-5) -> Bool { abs(a.asType(.float32) - b.asType(.float32)).max().item(Float.self) <= tolerance }

    /// AvgDown3D on one frame against the reference's view/permute/mean, written as explicit loops.
    @Test(.heavy) func averageDownMatchesTheReferenceChannelOrder() {
        let (h, w, c) = (4, 4, 3)
        let values = (0..<(h * w * c)).map { Float($0) * 0.37 - 5 }
        let x = MLXArray(values, [1, h, w, c])
        for (temporal, out) in [(1, 3), (2, 6), (2, 3)] {
            let y = vaeAverageDown(x, outChannels: out, temporal: temporal, spatial: 2)
            let group = c * temporal * 4 / out
            var expected = [Float]()
            for i in 0..<(h / 2) {
                for j in 0..<(w / 2) {
                    for o in 0..<out {
                        var total: Float = 0
                        for g in 0..<group {
                            let index = o * group + g // ((ch·ft + t)·2 + a)·2 + b
                            let b = index % 2, a = (index / 2) % 2, t = (index / 4) % temporal, ch = index / (4 * temporal)
                            // The zero-padded frame comes first: t = 0 is zero when temporal = 2.
                            if temporal == 1 || t == 1 { total += values[((2 * i + a) * w + (2 * j + b)) * c + ch] }
                        }
                        expected.append(total / Float(group))
                    }
                }
            }
            #expect(close(y, MLXArray(expected, [1, h / 2, w / 2, out])))
        }
    }

    /// DupUp3D(first_chunk) against the reference's repeat_interleave/view/permute and last temporal slice.
    @Test(.heavy) func duplicateUpMatchesTheReferenceChannelOrder() {
        let (h, w, c) = (2, 3, 4)
        let values = (0..<(h * w * c)).map { Float($0) + 1 }
        let x = MLXArray(values, [1, h, w, c])
        for (temporal, out) in [(2, 4), (1, 2), (2, 8)] {
            let y = vaeDuplicateUp(x, outChannels: out, temporal: temporal)
            let repeats = out * temporal * 4 / c
            var expected = [Float](repeating: 0, count: 4 * h * w * out)
            for i in 0..<(2 * h) {
                for j in 0..<(2 * w) {
                    for o in 0..<out {
                        let a = i % 2, b = j % 2, t = temporal - 1
                        let index = ((o * temporal + t) * 2 + a) * 2 + b
                        expected[(i * 2 * w + j) * out + o] = values[((i / 2) * w + j / 2) * c + index / repeats]
                    }
                }
            }
            #expect(close(y, MLXArray(expected, [1, 2 * h, 2 * w, out])))
        }
    }

    /// The segment-wise prefill equals one attention call with the explicit mask (q ≥ k) or same image block.
    @Test(.heavy) func segmentAttentionEqualsTheBlockCausalMask() throws {
        let slots = [false, false, false] + Array(repeating: true, count: 2) + [false, false] + Array(repeating: true, count: 1) + [false]
        let layout = try SequenceLayout(imageSlots: slots, conditions: [(2, 4), (2, 2)], target: (2, 2))
        let p = layout.prefixLength
        MLXRandom.seed(3)
        let q = MLXRandom.normal([1, 2, p, 8]), k = MLXRandom.normal([1, 2, p, 8]), v = MLXRandom.normal([1, 2, p, 8])
        var block = [Int](repeating: -1, count: p), id = 0
        for s in layout.segments where !s.text { for i in s.start..<s.end { block[i] = id }; id += 1 }
        var mask = [Bool]()
        for i in 0..<p { for j in 0..<p { mask.append(j <= i || (block[i] >= 0 && block[i] == block[j])) } }
        let reference = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.35, mask: .array(MLXArray(mask, [p, p])))
        let parts = layout.segments.map { s in
            MLXFast.scaledDotProductAttention(
                queries: q[0..., 0..., s.start..<s.end], keys: k[0..., 0..., 0..<s.end], values: v[0..., 0..., 0..<s.end], scale: 0.35,
                mask: s.text ? .causal : .none)
        }
        // Multi-token runs reuse the full kernel and match exactly (a top-left aligned causal mask would hide most
        // keys from the 2-token text run); a single query takes MLX's vector kernel, which rounds differently (~6e-4).
        #expect(close(concatenated(parts, axis: 2), reference, 2e-3))
    }

    /// Complex-pair RoPE: (x0 + i·x1)·e^{iθ} with θ = position · 10000^(−2k/dim) per axis.
    @Test(.heavy) func ropeRotatesInterleavedPairs() {
        var p = DiTRope.Positions(); p.frame = [3]; p.height = [-2]; p.width = [5]
        let axes = [4, 4, 4]
        let table = DiTRope.table(p, axes: axes)
        let x = MLXArray((0..<12).map { Float($0) - 5.5 }, [1, 1, 1, 12])
        let y = DiTRope.apply(x, table).reshaped(12).asArray(Float.self)
        let xs = (0..<12).map { Float($0) - 5.5 }
        var expected = [Float]()
        for (axis, position) in [(0, 3.0), (1, -2.0), (2, 5.0)] {
            for k in 0..<2 {
                let theta = Float(position / pow(10000.0, Double(2 * k) / 4))
                let pair = axis * 2 + k, x0 = xs[2 * pair], x1 = xs[2 * pair + 1]
                expected += [x0 * cos(theta) - x1 * sin(theta), x0 * sin(theta) + x1 * cos(theta)]
            }
        }
        for i in 0..<12 { #expect(abs(y[i] - expected[i]) < 1e-5) }
    }

    /// The Qwen2-VL patch order: merge group (row, column), then intra-group (row, column); each patch (C, T, P, P).
    @Test(.heavy) func visionPatchesFollowTheProcessorLayout() throws {
        var v = Qwen3VLConfig.Vision(); v.patch = 2; v.merge = 2; v.temporalPatch = 2
        let (w, h) = (8, 4)
        let rgb = (0..<(w * h * 3)).map { UInt8($0 % 251) }
        let (patches, grid) = try VisionPatches.make(rgb: rgb, width: w, height: h, config: v)
        #expect(grid == VisionGrid(t: 1, h: 2, w: 4) && patches.shape == [8, 24])
        let values = patches.asArray(Float.self)
        // Patch 5: merge block (0, 1) [groups of 2×2 patches across 4 columns → 2 blocks], intra (0, 1) → patch row 0, column 3.
        let row = 0, column = 3
        for c in 0..<3 {
            for t in 0..<2 {
                for i in 0..<2 {
                    for j in 0..<2 {
                        let byte = rgb[((row * 2 + i) * w + (column * 2 + j)) * 3 + c]
                        let got = values[5 * 24 + ((c * 2 + t) * 2 + i) * 2 + j]
                        #expect(abs(got - (Float(byte) / 127.5 - 1)) < 1e-6)
                    }
                }
            }
        }
    }

    @Test(.heavy) func timestepSinusoidIsCosineFirst() {
        let e = TimeTextEmbed.sinusoid(MLXArray([Float(250)])).asArray(Float.self)
        #expect(e.count == 256)
        for k in [0, 1, 17, 127] {
            let f = Float(exp(-log(10000.0) * Double(k) / 128))
            #expect(abs(e[k] - cos(250 * f)) < 1e-4 && abs(e[128 + k] - sin(250 * f)) < 1e-4)
        }
    }

    @Test(.heavy) func zeroCentredNormScalesByWeightPlusOne() {
        var c = SmallConfigs.transformer; c.contextDim = 4; c.heads = 1; c.headDim = 32
        let projection = TextProjection(c)
        let x = MLXArray([Float(1), 2, 3, 4], [1, 1, 4])
        let normed = MLXFast.rmsNorm(x, weight: projection.norm.weight + 1, eps: 1e-6)
        let rms = (Float(1 + 4 + 9 + 16) / 4).squareRoot()
        #expect(close(normed, x / rms, 1e-5))
    }
}
