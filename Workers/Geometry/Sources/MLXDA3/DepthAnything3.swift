// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
import MLX
import MLXNN

public struct DA3Outputs: OptionSet, Sendable {
    public let rawValue: Int

    public init(rawValue: Int) {
        self.rawValue = rawValue
    }

    public static let depth = DA3Outputs(rawValue: 1 << 0)
    public static let depthConfidence = DA3Outputs(rawValue: 1 << 1)
    public static let ray = DA3Outputs(rawValue: 1 << 2)
    public static let rayConfidence = DA3Outputs(rawValue: 1 << 3)
    public static let sky = DA3Outputs(rawValue: 1 << 4)

    public static let depthOnly: DA3Outputs = [.depth]
    public static let depthWithConfidence: DA3Outputs = [.depth, .depthConfidence]
    public static let all: DA3Outputs = [.depth, .depthConfidence, .ray, .rayConfidence, .sky]
}

/// Depth Anything 3 — depth estimation from single or multi-view images.
///
/// Combines a DinoV2 backbone with a DPT/DualDPT decoder head.
/// For inference-only; does not include camera encoder/decoder or GS heads.
public class DepthAnything3: Module {
    public static let patchSize = 14

    @ModuleInfo(key: "backbone") public var backbone: DinoVisionTransformer
    @ModuleInfo(key: "head") private var head: Module

    private let headType: String

    public init(backbone: DinoVisionTransformer, head: DPT) {
        self.headType = "dpt"
        self._backbone.wrappedValue = backbone
        self._head.wrappedValue = head
    }

    public init(backbone: DinoVisionTransformer, head: DualDPT) {
        self.headType = "dualdpt"
        self._backbone.wrappedValue = backbone
        self._head.wrappedValue = head
    }

    /// Forward pass.
    ///
    /// - Parameters:
    ///   - x: Input images `[B, S, H, W, C]` (NHWC) where S is number of views.
    ///     For single image: `[1, 1, H, W, 3]`
    ///   - refViewStrategy: Reference-view selection, matching python's
    ///     `ref_view_strategy` (default `saddle_balanced`). Only affects `S >= 3`.
    /// - Returns: Dict with keys like "depth", "depth_conf", "ray", "ray_conf", "sky"
    public func callAsFunction(
        _ x: MLXArray,
        refViewStrategy: RefViewStrategy = .saddleBalanced
    ) -> [String: MLXArray] {
        let H = x.dim(2)
        let W = x.dim(3)

        let feats = DA3Profiler.measure("  backbone", sync: { MLX.eval($0.map { $0.0 }) }) {
            backbone(x, refViewStrategy: refViewStrategy, checkpoint: {}).0
        }

        let output: [String: MLXArray] = DA3Profiler.measure("  head", sync: { MLX.eval($0) }) {
            if let dualHead = head as? DualDPT {
                return dualHead.callAsFunction(feats: feats, H: H, W: W)
            } else if let dptHead = head as? DPT {
                return dptHead.callAsFunction(feats: feats, H: H, W: W)
            } else {
                fatalError("Unknown head type")
            }
        }
        return Self.processMonoSky(output)
    }

    /// Forward pass that observes `checkpoint` between backbone blocks and around the head,
    /// with the reference's mono sky post-processing.
    public func predict(
        _ x: MLXArray,
        refViewStrategy: RefViewStrategy = .saddleBalanced,
        checkpoint: () throws -> Void
    ) rethrows -> [String: MLXArray] {
        let H = x.dim(2)
        let W = x.dim(3)
        try checkpoint()
        let feats = try backbone(x, refViewStrategy: refViewStrategy, checkpoint: checkpoint).0
        MLX.eval(feats.flatMap { [$0.0, $0.1] })
        try checkpoint()
        let output: [String: MLXArray]
        if let dualHead = head as? DualDPT {
            output = dualHead.callAsFunction(feats: feats, H: H, W: W)
        } else if let dptHead = head as? DPT {
            output = dptHead.callAsFunction(feats: feats, H: H, W: W)
        } else {
            fatalError("Unknown head type")
        }
        MLX.eval(Array(output.values))
        try checkpoint()
        return Self.processMonoSky(output)
    }

    /// Port of `DepthAnything3Net._process_mono_sky_estimation`
    /// (`src/depth_anything_3/model/da3.py`): pixels with sky probability ≥ 0.3 take the
    /// 99th percentile (linear interpolation, as `torch.quantile`) of the non-sky depth.
    /// Skipped unless both the non-sky and sky populations exceed 10 pixels. The mask and
    /// percentile span the whole batch, as upstream. Upstream draws a random 100,000-pixel
    /// sample above that population; this uses every non-sky pixel, which is deterministic.
    /// The sky tensor itself is returned unchanged.
    public static func processMonoSky(_ output: [String: MLXArray]) -> [String: MLXArray] {
        guard let sky = output["sky"], let depth = output["depth"], sky.shape == depth.shape else { return output }
        let skyValues = sky.asType(.float32).asArray(Float.self)
        let depthValues = depth.asType(.float32).asArray(Float.self)
        let nonSky = skyValues.map { $0 < 0.3 }
        let nonSkyCount = nonSky.lazy.filter { $0 }.count
        guard nonSkyCount > 10, depthValues.count - nonSkyCount > 10 else { return output }
        var population = [Float]()
        population.reserveCapacity(nonSkyCount)
        for (index, keep) in nonSky.enumerated() where keep { population.append(depthValues[index]) }
        // A NaN would make the percentile undefined; the caller's numeric validation rejects it.
        guard population.allSatisfy({ !$0.isNaN }) else { return output }
        population.sort()
        let position = 0.99 * Double(population.count - 1)
        let lower = Int(position.rounded(.down)), upper = min(lower + 1, population.count - 1)
        let fraction = Float(position - Double(lower))
        let percentile = population[lower] + (population[upper] - population[lower]) * fraction
        var replaced = depthValues
        for index in replaced.indices where !nonSky[index] { replaced[index] = percentile }
        var result = output
        result["depth"] = MLXArray(replaced, depth.shape).asType(depth.dtype)
        return result
    }

    public func callAsFunction(
        _ x: MLXArray,
        outputs requestedOutputs: DA3Outputs = .all,
        refViewStrategy: RefViewStrategy = .saddleBalanced
    ) -> [String: MLXArray] {
        if requestedOutputs == .all {
            return callAsFunction(x, refViewStrategy: refViewStrategy)
        }

        let H = x.dim(2)
        let W = x.dim(3)

        // Backbone: extract multi-scale features
        let feats = DA3Profiler.measure("  backbone", sync: { MLX.eval($0.map { $0.0 }) }) {
            backbone(x, refViewStrategy: refViewStrategy, checkpoint: {}).0
        }

        // Head: decode features to depth
        let output: [String: MLXArray] = DA3Profiler.measure("  head", sync: { MLX.eval($0) }) {
            if let dualHead = head as? DualDPT {
                return dualHead.callAsFunction(feats: feats, H: H, W: W, outputs: requestedOutputs)
            } else if let dptHead = head as? DPT {
                return dptHead.callAsFunction(feats: feats, H: H, W: W, outputs: requestedOutputs)
            } else {
                fatalError("Unknown head type")
            }
        }
        return Self.processMonoSky(output)
    }
}

// MARK: - Factory

/// Build a DepthAnything3 model from a config name.
///
/// - Parameter configName: One of "da3-small", "da3-base", "da3-large", "da3-giant", "da3mono-large"
/// - Returns: Uninitialized model (use `loadModel` for weights)
public func buildModel(configName: String) -> DepthAnything3 {
    return buildModel(config: DA3Config.fromName(configName))
}

public func buildModel(config cfg: DA3Config) -> DepthAnything3 {
    let bb = cfg.backbone

    let backbone = DinoVisionTransformer(
        imgSize: 518,
        patchSize: 14,
        inChannels: 3,
        embedDim: bb.embedDim,
        depth: bb.depth,
        numHeads: bb.numHeads,
        mlpRatio: bb.mlpRatio,
        qkvBias: true,
        projBias: true,
        ffnBias: true,
        initValues: 1.0,
        useSwiglu: bb.useSwiglu,
        altStart: cfg.altStart,
        qknormStart: cfg.qknormStart,
        ropeStart: cfg.ropeStart,
        catToken: cfg.catToken,
        outLayers: cfg.outLayers
    )

    let hc = cfg.head

    if cfg.headType == "dualdpt" {
        let head = DualDPT(
            dimIn: hc.dimIn,
            patchSize: 14,
            outputDim: hc.outputDim,
            features: hc.features,
            outChannels: hc.outChannels,
            posEmbed: hc.posEmbed
        )
        return DepthAnything3(backbone: backbone, head: head)
    } else {
        let head = DPT(
            dimIn: hc.dimIn,
            patchSize: 14,
            outputDim: hc.outputDim,
            features: hc.features,
            outChannels: hc.outChannels,
            posEmbed: hc.posEmbed,
            useSkyHead: hc.useSkyHead,
            normType: hc.normType
        )
        return DepthAnything3(backbone: backbone, head: head)
    }
}

// swiftlint:enable all
