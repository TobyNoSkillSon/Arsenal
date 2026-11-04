import Testing
import MLX
import DDColorNative
import LaMaNative
@testable import RestorationWorker
struct ContractTests {
    @Test func registrationsAndPrecisionPolicies() {
        #expect(RestorationWorker.registrations.count == 6)
        #expect(RestorationWorker.registrations.compactMap(\.conversion).map(\.recipe).contains("cast:restoration-lama-v1"))
        #expect(!DDColorAdapter.eligible("encoder.arch.stages.0.0.pwconv1.weight", [768, 192]))
        #expect(!LaMaAdapter.eligible("model.5.conv1.ffc.convl2l.weight", [128, 3, 3, 128]))
        #expect(!RIFEAdapter.eligible("block0.conv.weight", [192, 3, 3, 96]))
        #expect(SeedVR2Adapter.eligible("blocks.0.attn.q.weight", [128, 128]))
        #expect(!SeedVR2Adapter.eligible("vid_in.proj.weight", [128, 132]))
    }
    /// The inventories match the pinned checkpoints' safetensors headers: 575 DDColor tensors, and 378 LaMa tensors
    /// after folding 152 BatchNorms out of the 837 float32 tensors.
    @Test func photoInventories() {
        let ddcolor = DDColorWeights.shapes()
        #expect(ddcolor.count == 575)
        #expect(ddcolor.values.reduce(0) { $0 + $1.reduce(1, *) } == 227_881_750)
        let lama = LaMaFold.foldedShapes()
        #expect(lama.count == 378)
        #expect(lama["model.5.conv1.ffc.convg2g.fu.conv_layer.weight"] == [384, 1, 1, 384])
        #expect(lama["model.24.weight"] == [256, 3, 3, 512])
    }

    /// The BatchNorm fold on a synthetic full-inventory source (CPU, no model run): every BatchNorm folds into the
    /// right convolutions with scale = gamma / sqrt(var + eps) and shift = beta - mean * scale; a leftover tensor fails.
    @Test func lamaFoldAlgebraAndCoverage() throws {
        try Device.withDefaultDevice(.cpu) {
            var source: [String: MLXArray] = [:]
            for (key, shape) in LaMaFold.foldedShapes() where !key.hasSuffix(".bias") || ["model.24.bias", "model.27.bias", "model.30.bias", "model.34.bias"].contains(key) {
                source[key] = MLX.full(shape, values: MLXArray(Float(1)))
            }
            func bn(_ p: String, _ n: Int) {
                source[p + ".weight"] = MLX.full([n], values: MLXArray(Float(2))); source[p + ".bias"] = MLX.full([n], values: MLXArray(Float(0.25)))
                source[p + ".running_mean"] = MLX.full([n], values: MLXArray(Float(0.5))); source[p + ".running_var"] = MLX.full([n], values: MLXArray(Float(3)))
            }
            for i in 1...4 { bn("model.\(i).bn_l", [64, 128, 256, 128][i - 1]) }
            bn("model.4.bn_g", 384)
            for i in 5...22 {
                for c in ["conv1", "conv2"] {
                    let p = "model.\(i).\(c)"
                    bn(p + ".bn_l", 128); bn(p + ".bn_g", 384); bn(p + ".ffc.convg2g.conv1.1", 192); bn(p + ".ffc.convg2g.fu.bn", 384)
                }
            }
            for (b, n) in [(25, 256), (28, 128), (31, 64)] { bn("model.\(b)", n) }
            #expect(source.count == 837)
            let folded = try LaMaFold.fold(source)
            #expect(Set(folded.keys) == Set(LaMaFold.foldedShapes().keys))
            let scale = 2 / (3 + 1e-5 as Float).squareRoot(), shift = 0.25 - 0.5 * scale
            func value(_ k: String) -> Float { folded[k]!.reshaped([-1])[0].item(Float.self) }
            for key in [
                "model.9.conv2.ffc.convl2l.weight", "model.9.conv2.ffc.convg2l.weight", "model.9.conv2.ffc.convg2g.conv2.weight",
                "model.9.conv2.ffc.convg2g.fu.conv_layer.weight", "model.1.ffc.convl2l.weight", "model.24.weight"
            ] {
                #expect(abs(value(key) - scale) < 1e-6, "\(key)")
            }
            for key in ["model.9.conv2.ffc.convl2l.bias", "model.9.conv2.ffc.convl2g.bias", "model.9.conv2.ffc.convg2g.conv1.0.bias"] {
                #expect(abs(value(key) - shift) < 1e-6, "\(key)")
            }
            #expect(abs(value("model.24.bias") - (scale + shift)) < 1e-6)
            #expect(value("model.34.weight") == 1)
            var extra = source; extra["model.99.weight"] = MLXArray(Float(1))
            #expect(throws: LaMaWeightError.self) { try LaMaFold.fold(extra) }
        }
    }
}
