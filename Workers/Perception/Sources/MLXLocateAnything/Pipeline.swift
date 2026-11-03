// Vendored model implementation; source and adaptations: Workers/Perception/licenses/MLXLocateAnything/PORTED-FROM.md
import CoreGraphics
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import Tokenizers
import IntegrationKit

// High-level pipeline mirroring locateanything_worker.py: predict(image,
// question, visualPromptBox?) -> detected boxes. Phase 1 uses standard
// autoregressive (greedy) decoding of the box grammar; Parallel Box Decoding is
// added in Phase 2.

/// One detection: a bounding box (or point) with an optional referring-expression label.
public struct DetectedBox: Sendable {
    /// The referring phrase this detection belongs to (e.g. "car"), if the model emitted one.
    public let label: String?
    /// Normalized `[x1, y1, x2, y2]` (or `[x, y]` for a point) in `0...1`.
    public let box: [Float]
    /// Pixel coordinates in the original image (`[x1, y1, x2, y2]` or `[x, y]`).
    public let boxPixels: [Float]
    /// `true` when this is a point rather than a box.
    public let isPoint: Bool
}

/// Sampling controls for autoregressive decoding. Defaults to greedy (deterministic:
/// same image + prompt → same boxes). Only affects the `.ar` decoding path — the
/// Parallel Box Decoding modes select coordinates by threshold, deterministically.
public struct SamplingParameters: Sendable {
    /// `0` = greedy (argmax). Higher = flatter distribution / more random.
    public var temperature: Float
    /// Nucleus sampling: keep the smallest set of tokens summing to `topP`. `1` = off.
    public var topP: Float
    /// Keep only the `topK` most-likely tokens. `0` = off.
    public var topK: Int

    public init(temperature: Float = 0, topP: Float = 1.0, topK: Int = 0) {
        self.temperature = temperature
        self.topP = topP
        self.topK = topK
    }

    /// Deterministic argmax decoding.
    public static let greedy = SamplingParameters()

    public var isGreedy: Bool { temperature <= 0 }
}

/// The result of a ``LocateAnythingSession/predict(image:question:maxNewTokens:mode:visualPromptBox:visualPromptBoxFormat:)`` call.
public struct LocateAnythingResult: Sendable {
    /// The raw decoded box grammar, e.g. `<ref>car</ref><box><0><111><74><213></box>`.
    public let text: String
    /// The parsed detections.
    public let boxes: [DetectedBox]
    /// Number of tokens generated.
    public let tokenCount: Int
}

/// The single library-side driver both the CLI and the SwiftUI demo consume.
///
/// `@unchecked Sendable` under a **single-writer** contract: at most one caller
/// drives `predict(...)` at a time. The CLI is single-threaded; the GUI runs it
/// on one `Task.detached` while the UI only reads results afterward. Sharing a
/// session across two concurrent `predict` callers is unsupported — build a
/// serializing wrapper for that case. Inputs are materialized (`CGImage`), never
/// file paths, so a sandboxed frontend works identically to an unsandboxed CLI.
public final class LocateAnythingSession: @unchecked Sendable {
    public let model: LocateAnythingModel
    public let config: LocateAnythingConfiguration
    let tokenizer: any Tokenizers.Tokenizer
    let mergeLength: Int
    var cancellation = NativeCancellation()

    init(model: LocateAnythingModel, config: LocateAnythingConfiguration, tokenizer: any Tokenizers.Tokenizer) {
        self.model = model
        self.config = config
        self.tokenizer = tokenizer
        self.mergeLength = config.visionConfig.mergeKernelSize[0] * config.visionConfig.mergeKernelSize[1]
    }

    // MARK: - Inference

    /// Ground the instruction in the image and return the detected boxes.
    /// - Parameters:
    ///   - image: the input image (already decoded — see ``LocateAnythingImageProcessor/baked(_:)``).
    ///   - question: a LocateAnything instruction, e.g. `"Locate all the instances that matches the following description: car."`.
    ///   - maxNewTokens: generation cap.
    ///   - mode: decoding strategy — ``PBDMode/hybrid`` (default) uses Parallel Box Decoding.
    ///   - visualPromptBox: optional box to crop as a visual prompt (released weights don't support this yet).
    ///   - visualPromptBoxFormat: how `visualPromptBox` coordinates are expressed.
    public func predict(
        image: CGImage,
        question: String,
        maxNewTokens: Int = 512,
        mode: PBDMode = .hybrid,
        sampling: SamplingParameters = .greedy,
        visualPromptBox: [Float]? = nil,
        visualPromptBoxFormat: BoxFormat = .normalized1000
    ) throws -> LocateAnythingResult {
        var processed: [ProcessedImage] = [try LocateAnythingImageProcessor.preprocess(image)]
        var numImages = 1
        if let vp = visualPromptBox {
            let crop = cropVisualPrompt(image, box: vp, format: visualPromptBoxFormat)
            processed.append(try LocateAnythingImageProcessor.preprocess(crop))
            numImages = 2
        }

        let (pixelValues, grids) = LocateAnythingImageProcessor.batch(processed)
        let pv = pixelValues.asType(.float16)

        let prompt = LocateAnythingPrompt.chatML(question: question, numImages: numImages)
        let expanded = LocateAnythingPrompt.expandPlaceholders(prompt, grids: grids, mergeLength: mergeLength)
        let ids = tokenizer.encode(text: expanded, addSpecialTokens: false)

        return try runGeneration(
            ids: ids, pixelValues: pv, grids: grids,
            displayWidth: image.width, displayHeight: image.height,
            maxNewTokens: maxNewTokens, mode: mode, sampling: sampling)
    }

    /// Ground an instruction across sampled video frames.
    ///
    /// The frontend extracts the frames (see ``LocateAnythingVideo/plan(totalFrames:videoFPS:)``)
    /// and passes them as materialized images. Frames are globally downscaled to the
    /// reference pixel budget, preprocessed per-frame, and grounded together.
    /// - Parameters:
    ///   - frames: the sampled frames, in order.
    ///   - timestamps: each frame's time in seconds (used in the prompt).
    ///   - sampledFPS: effective sample rate, reported in the prompt.
    ///   - displayWidth: width used to map boxes to pixels (defaults to the frame's).
    ///   - displayHeight: height used to map boxes to pixels (defaults to the frame's).
    private func runGeneration(
        ids: [Int], pixelValues: MLXArray, grids: [(Int, Int)],
        displayWidth: Int, displayHeight: Int,
        maxNewTokens: Int, mode: PBDMode, sampling: SamplingParameters
    ) throws -> LocateAnythingResult {
        let inputIds = MLXArray(ids.map { Int32($0) }, [1, ids.count])
        let cache = model.newCache()
        let embeds = model.inputEmbeddings(inputIds: inputIds, pixelValues: pixelValues, grids: grids)

        let generated: [Int]
        if mode == .ar {
            generated = try generateAR(ids: ids, embeds: embeds, cache: cache, maxNewTokens: maxNewTokens, sampling: sampling)
        } else {
            let decoder = PBDDecoder(model: model, mode: mode)
            decoder.cancellation = cancellation
            generated = try decoder.generate(
                prompt: ids, inputsEmbeds: embeds, cache: cache, maxTokens: maxNewTokens)
        }

        let text = tokenizer.decode(tokens: generated)
        let boxes = try boxes(generated: generated, imageWidth: displayWidth, imageHeight: displayHeight)
        return LocateAnythingResult(text: text, boxes: boxes, tokenCount: generated.count)
    }

    /// The generated tokens' detections: explicit termination, then the complete reference/box grammar (Arsenal: shared
    /// by serving and the parser fixtures, `NativeLocateAnything.parseOutput`).
    func boxes(generated: [Int], imageWidth: Int, imageHeight: Int) throws -> [DetectedBox] {
        try parseBoxes(try LocateAnythingBoxes.completed(generated, config: config), imageWidth: imageWidth, imageHeight: imageHeight)
    }

    /// The token IDs of decoded model output with its special tokens kept, as the maker's `generate` returns it.
    func tokens(decoded text: String) -> [Int] { tokenizer.encode(text: text, addSpecialTokens: false) }

    /// Standard autoregressive decoding (Phase 1 baseline / PBD fallback).
    /// Greedy by default; `sampling` enables temperature / top-p / top-k.
    private func generateAR(
        ids: [Int], embeds: MLXArray, cache: [KVCache], maxNewTokens: Int,
        sampling: SamplingParameters
    ) throws -> [Int] {
        let sampler: any LogitSampler =
            sampling.isGreedy
            ? ArgMaxSampler()
            : TopPSampler(temperature: sampling.temperature, topP: sampling.topP, topK: sampling.topK)

        let prefillLogits = model.languageLogits(
            inputsEmbeds: embeds, mask: ids.count > 1 ? .causal : .none, cache: cache)
        var nextId = sampleLast(prefillLogits, sampler)
        let eos = Set(config.eosTokenIds)
        var generated: [Int] = []
        generated.reserveCapacity(maxNewTokens)
        var loop = LocateAnythingRepetitionGuard()
        for _ in 0..<maxNewTokens {
            try cancellation.check()
            generated.append(nextId)
            if eos.contains(nextId) { break }
            if loop.append([nextId]) { throw LocateAnythingOutputError.repetition }
            let emb1 = model.embed(MLXArray([Int32(nextId)], [1, 1]))
            let logits = model.languageLogits(inputsEmbeds: emb1, mask: .none, cache: cache)
            nextId = sampleLast(logits, sampler)
        }
        return generated
    }

    private func sampleLast(_ logits: MLXArray, _ sampler: any LogitSampler) -> Int {
        let l = logits.dim(1)
        let last = logits[0..., (l - 1)..<l, 0...].reshaped([1, -1]) // [1, V]
        return sampler.sample(logits: last).item(Int.self)
    }

    // MARK: - Helpers

    public enum BoxFormat: Sendable {
        case normalized1000, normalized1, pixel
    }

    private func cropVisualPrompt(_ image: CGImage, box: [Float], format: BoxFormat) -> CGImage {
        guard box.count == 4 else { return image }
        let w = Float(image.width)
        let h = Float(image.height)
        var x1 = box[0], y1 = box[1], x2 = box[2], y2 = box[3]
        switch format {
        case .normalized1000:
            x1 = x1 / 1000 * w; y1 = y1 / 1000 * h; x2 = x2 / 1000 * w; y2 = y2 / 1000 * h
        case .normalized1:
            x1 *= w; y1 *= h; x2 *= w; y2 *= h
        case .pixel:
            break
        }
        let rect = CGRect(
            x: CGFloat(x1), y: CGFloat(y1),
            width: CGFloat(max(1, x2 - x1)), height: CGFloat(max(1, y2 - y1)))
        return image.cropping(to: rect) ?? image
    }

    private func parseBoxes(_ tokens: [Int], imageWidth: Int, imageHeight: Int) throws -> [DetectedBox] {
        try LocateAnythingBoxes.parse(tokens, config: config, imageWidth: imageWidth, imageHeight: imageHeight) { tokenizer.decode(tokens: $0, skipSpecialTokens: true) }
    }
}
