// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import Testing
import MLX
import ArsenalWire
import IntegrationKit
@testable import LanguageWorker

/// LFM2.5-VL-450M processor layout against Transformers' Lfm2VlImageProcessor/Lfm2VlProcessor arithmetic
/// (smart_resize, _is_image_too_large, find_closest_aspect_ratio, _build_image_tokens), evaluated once in Python for
/// the pinned preprocessor_config.json (64–256 tokens, 512-px tiles, 2–10 tiles, thumbnail, tolerance 2).
struct LFMProcessorTests {
    init() { MLXReady.ensure() }
    private func processor() throws -> LFMVisionProcessor {
        let config: JSONObject = [
            "do_image_splitting": true, "downsample_factor": 2, "encoder_patch_size": 16, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5],
            "max_image_tokens": 256, "max_pixels_tolerance": 2.0, "max_tiles": 10, "min_image_tokens": 64, "min_tiles": 2, "tile_size": 512,
            "use_thumbnail": true
        ]
        return LFMVisionProcessor(
            try JSONDecoder().decode(LFMVisionConfiguration.self, from: WireJSON.encode(config)),
            tokenizer: RandomTokenizer(imageID: 396, startID: 261, endID: 262, imageToken: "<image>"), imageTokenID: 396)
    }

    @Test func layoutMatchesTransformers() throws {
        let lfm = try processor()
        // (width, height) → rows, columns, resized width, resized height, single-image tokens, total image tokens.
        let cases: [([Int], [Int])] = [
            ([100, 80], [1, 1, 288, 256, 72, 72]), ([256, 256], [1, 1, 256, 256, 64, 64]), ([512, 512], [1, 1, 512, 512, 256, 256]),
            ([640, 480], [1, 1, 576, 416, 234, 234]), ([1024, 768], [2, 3, 576, 416, 234, 1770]), ([300, 1200], [1, 1, 256, 1024, 256, 256]),
            ([2000, 500], [1, 4, 1024, 256, 256, 1280]), ([4032, 3024], [2, 3, 576, 416, 234, 1770]), ([731, 731], [2, 2, 512, 512, 256, 1280])
        ]
        for (size, expected) in cases {
            let layout = lfm.layout(width: size[0], height: size[1])
            #expect(
                [layout.rows, layout.columns, layout.width, layout.height, layout.imageTokens, layout.totalTokens] == expected, "\(size)")
            #expect(lfm.featureCount(layout) == expected[5], "\(size)")
        }
    }

    @Test func expansionUsesMarkersAndChecksCounts() throws {
        let lfm = try processor()
        let tiled = lfm.layout(width: 1024, height: 768)
        let text = lfm.replacement(tiled)
        #expect(text.hasPrefix("<|image_start|><|img_row_1_col_1|><image>"))
        #expect(text.contains("<|img_row_2_col_3|>") && text.contains("<|img_thumbnail|>") && text.hasSuffix("<image><|image_end|>"))
        let single = lfm.layout(width: 256, height: 256)
        #expect(lfm.replacement(single) == "<|image_start|>" + String(repeating: "<image>", count: 64) + "<|image_end|>")
        // Prompt "a <image> b" with one image: the placeholder becomes the full expansion.
        let prompt = [98, 396, 99]
        let expanded = try lfm.expand(prompt, layouts: [tiled])
        #expect(expanded.first == 98 && expanded.last == 99)
        #expect(expanded.filter { $0 == 396 }.count == 1770)
        // Placeholder/image mismatches throw instead of reaching the model's feature merge.
        #expect(throws: (any Error).self) { try lfm.expand(prompt, layouts: [tiled, single]) }
    }

    @Test func tilesAndThumbnailBecomeEncoderInputs() throws {
        NativeRuntime.prepare()
        let lfm = try processor()
        let image = CIImage(color: CIColor(red: 0.1, green: 0.5, blue: 0.9)).cropped(to: CGRect(x: 0, y: 0, width: 1024, height: 768))
        let (layout, inputs) = lfm.encoderInputs(image)
        #expect(layout.rows == 2 && layout.columns == 3)
        #expect(inputs.count == 7)
        #expect(inputs.prefix(6).allSatisfy { $0.0.shape == [1024, 768] && $0.1.h == 32 && $0.1.w == 32 })
        #expect(inputs[6].0.shape == [26 * 36, 768] && inputs[6].1.h == 26 && inputs[6].1.w == 36)
    }

    /// Siglip2 position-embedding resize (bilinear, antialias) against Pillow's float BILINEAR on a 16×16 ramp.
    @Test(.heavy) func positionResizeMatchesAntialiasedBilinear() {
        // Precondition: float32 matmuls are exact only with MLX_ENABLE_TF32=0, which MLXReady (run by init) sets before any
        // suite's MLX work. If this fails, the process was changed by another path; the tolerance is not the problem.
        #expect(getenv("MLX_ENABLE_TF32").map { String(cString: $0) } == "0")
        NativeRuntime.prepare()
        let ramp = MLXArray((0..<256).map { Float($0) }, [16, 16, 1])
        let cases: [([Int], [Float], Float)] = [
            ([2, 3], [47.1479, 53.6121, 124.2679, 130.7321, 201.3878, 207.8521], 207.8521),
            ([5, 20], [1.4066, 4.3235, 7.5, 10.6765, 13.5934, 12.6066, 15.5235, 18.7, 21.8765, 24.7934, 25.4066, 28.3235], 253.5934),
            ([36, 26], [0, 0.1667, 0.6111, 1.0556, 1.5, 1.9444, 2.3889, 2.8333, 3.2778, 3.7222, 4.1667, 4.6111], 255)
        ]
        for (size, prefix, last) in cases {
            let values = Siglip2PositionResize.resize(ramp, height: size[1], width: size[0]).asArray(Float.self)
            #expect(values.count == size[0] * size[1])
            #expect(zip(values, prefix).allSatisfy { abs($0 - $1) < 2e-3 }, "\(size)")
            #expect(abs((values.last ?? 0) - last) < 2e-3, "\(size)")
        }
    }
}
