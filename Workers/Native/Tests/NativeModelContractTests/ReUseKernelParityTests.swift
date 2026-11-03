// SPDX-License-Identifier: MIT
// Fixed-tensor parity of RE-USE's Metal causal conv + SiLU and selective scan against the stock
// MLX path, in both half activation dtypes. The stock path is the gate's reference: the kernels
// must round where its half ops round (review M1, 16 Oct 2026). Inputs are seeded and nonconstant.
import Foundation
import Testing
import MLX
import MLXNN
import MLXRandom
@testable import SoundEngine

struct ReUseKernelParityTests {
    static let halves: [DType] = [.bfloat16, .float16]

    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
    }

    /// Bitwise comparison of two equally shaped half arrays.
    struct Parity: CustomStringConvertible {
        let count: Int
        let identical: Int
        let maxULP: Int
        let relativeL2: Double
        let bothNaN: Int
        var mismatchShare: Double { Double(count - identical - bothNaN) / Double(count) }
        var description: String {
            String(
                format: "%d values, %d bit-identical, %.3g%% differ, max %d ULP, relative L2 %.3g",
                count, identical, 100 * mismatchShare, maxULP, relativeL2)
        }

        init(_ candidate: MLXArray, _ reference: MLXArray) {
            precondition(candidate.shape == reference.shape && candidate.dtype == reference.dtype)
            let a = candidate.view(dtype: .uint16).asArray(UInt16.self)
            let b = reference.view(dtype: .uint16).asArray(UInt16.self)
            let af = candidate.asType(.float32).asArray(Float.self), bf = reference.asType(.float32).asArray(Float.self)
            func ordered(_ h: UInt16) -> Int { h & 0x8000 == 0 ? Int(h) : -Int(h & 0x7fff) }
            var identical = 0, maxULP = 0, nan = 0
            var noise = 0.0, signal = 0.0
            for i in a.indices {
                if af[i].isNaN && bf[i].isNaN { nan += 1; continue }
                if a[i] == b[i] { identical += 1 }
                if af[i].isNaN != bf[i].isNaN || af[i].isFinite != bf[i].isFinite {
                    maxULP = .max
                } else if af[i].isFinite {
                    maxULP = max(maxULP, abs(ordered(a[i]) - ordered(b[i])))
                    noise += Double(af[i] - bf[i]) * Double(af[i] - bf[i])
                    signal += Double(bf[i]) * Double(bf[i])
                } else if a[i] != b[i] {
                    maxULP = .max
                }
            }
            self.count = a.count
            self.identical = identical
            self.maxULP = maxULP
            self.relativeL2 = signal == 0 ? (noise == 0 ? 0 : .infinity) : (noise / signal).squareRoot()
            self.bothNaN = nan
        }
    }

    @Test(.heavy) func causalConvAndSiLUMatchStockInBothHalfDtypes() throws {
        let keys = MLXRandom.split(key: MLXRandom.key(20261016), into: 4)
        let (batch, length, d) = (2, 257, 256)
        for half in Self.halves {
            let conv = Conv1d(inputChannels: d, outputChannels: d, kernelSize: 4, padding: 3, groups: d, bias: true)
            let weight = (MLXRandom.normal([d, 4, 1], key: keys[0]) * 0.5).asType(half)
            let bias = (MLXRandom.normal([d], key: keys[1]) * 0.1).asType(half)
            try conv.update(parameters: ModuleParameters.unflattened(["weight": weight, "bias": bias]), verify: .all)
            let x = (MLXRandom.normal([batch, length, d], key: keys[2]) * 1.5 + MLXRandom.uniform(low: -1, high: 1, [1, 1, d], key: keys[3])).asType(half)

            let metalConv = try ReUseMetalCausalConv1d.run(x: x, weight: conv.weight, bias: try #require(conv.bias))
            let stockConv = ReUseMamba.stockCausalConv(x, weight: conv.weight, bias: conv.bias)
            #expect(metalConv.dtype == half && stockConv.dtype == half)
            let convParity = Parity(metalConv, stockConv)
            let siluParity = Parity(MLXNN.silu(metalConv), MLXNN.silu(stockConv))
            // The stock causal conv is the same convolution as the module's padded Conv1d, which
            // routes through a grouped GEMM with coarser product rounding (never the served path).
            let module = Parity(stockConv, conv(x)[0..., ..<length, 0...])
            print("RE-USE parity \(half) causal conv: \(convParity); conv+SiLU: \(siluParity); stock vs padded Conv1d: \(module)")
            #expect(convParity.identical == convParity.count && siluParity.identical == siluParity.count)
            #expect(module.relativeL2 <= (half == .bfloat16 ? 0x1p-7 : 0x1p-10))
        }
    }

    /// The kernel's in-register SiLU(z) must equal MLXNN.silu bit for bit over every half value:
    /// with B = C = 0, u = D = 1 the scan output is exactly SiLU(z) in the activation dtype.
    @Test(.heavy) func scanSiLUEqualsMLXSiLUOnEveryHalfBitPattern() {
        for half in Self.halves {
            let patterns = MLXArray((0...UInt16.max).map { $0 }, [1, 256, 256]).view(dtype: half)
            let ones = MLXArray.ones([1, 256, 256], dtype: half)
            let zerosN = MLXArray.zeros([1, 256, reUseMambaStateSize], dtype: half)
            let scan = ReUseMetalSelectiveScan.run(
                deltaRaw: MLXArray.zeros([1, 256, 256], dtype: half),
                A_log: MLXArray.zeros([256, reUseMambaStateSize], dtype: half),
                B: zerosN, C: zerosN, u: ones, D: MLXArray.ones([256], dtype: half), z: patterns)
            let parity = Parity(scan, MLXNN.silu(patterns))
            print("RE-USE parity \(half) in-kernel SiLU over all 65,536 patterns: \(parity)")
            #expect(scan.dtype == half)
            #expect(parity.identical + parity.bothNaN == parity.count)
        }
    }

    @Test(.heavy) func selectiveScanMatchesStockInBothHalfDtypes() {
        let keys = MLXRandom.split(key: MLXRandom.key(20261017), into: 8)
        let (batch, length, d, n) = (2, 300, 256, reUseMambaStateSize)
        for half in Self.halves {
            // Model-like magnitudes: softplus(deltaRaw) mostly 1e-3…1, A = −(1…16), D ≈ 1.
            let deltaRaw = (MLXRandom.normal([batch, length, d], key: keys[0]) * 2 - 3).asType(half)
            let aLog = MLX.log(MLXRandom.uniform(low: 1, high: 16, [d, n], key: keys[1])).asType(half)
            let b = MLXRandom.normal([batch, length, n], key: keys[2]).asType(half)
            let c = MLXRandom.normal([batch, length, n], key: keys[3]).asType(half)
            let u = (MLXRandom.normal([batch, length, d], key: keys[4]) * 2).asType(half)
            let dParam = (1 + MLXRandom.normal([d], key: keys[5]) * 0.3).asType(half)
            let z = (MLXRandom.normal([batch, length, d], key: keys[6]) * 3).asType(half)

            let metal = ReUseMetalSelectiveScan.run(deltaRaw: deltaRaw, A_log: aLog, B: b, C: c, u: u, D: dParam, z: z)
            let stock = StockSelectiveScan.run(deltaRaw: deltaRaw, A_log: aLog, B: b, C: c, u: u, D: dParam, z: z).asType(half)
            let parity = Parity(metal, stock)
            print("RE-USE parity \(half) selective scan: \(parity)")
            #expect(metal.dtype == half)
            // Every operation and rounding point matches the stock graph, including MLX's precise
            // exp (the JIT build's metal::exp is the fast form and differed on 74% of FP32 inputs).
            #expect(parity.identical == parity.count)
        }
    }

    /// One whole ReUseMamba layer (inProj → conv → SiLU → xProj/dtProj → scan → outProj) with
    /// seeded, nonconstant half weights: Metal against stock.
    @Test(.heavy) func mambaLayerMetalMatchesStockInBothHalfDtypes() throws {
        let keys = MLXRandom.split(key: MLXRandom.key(20261018), into: 12)
        for half in Self.halves {
            let metal = ReUseMamba(dModel: 64, kernels: .metal), stock = ReUseMamba(dModel: 64, kernels: .stock)
            func normal(_ shape: [Int], _ k: Int, _ scale: Float) -> MLXArray { (MLXRandom.normal(shape, key: keys[k]) * scale).asType(half) }
            let dtTarget = MLX.exp(MLXRandom.uniform(low: log(Float(1e-3)), high: log(Float(0.1)), [256], key: keys[6]))
            let weights: [String: MLXArray] = [
                "inProj.weight": normal([512, 64], 0, 1 / 8),
                "conv1d.weight": normal([256, 4, 1], 1, 0.5),
                "conv1d.bias": normal([256], 2, 0.1),
                "xProj.weight": normal([36, 256], 3, 1 / 16),
                "dtProj.weight": normal([256, 4], 4, 0.5),
                "dtProj.bias": (dtTarget + MLX.log(-MLX.expm1(-dtTarget))).asType(half),
                "A_log": MLX.log(MLX.broadcast(MLXArray((1...16).map { Float($0) }), to: [256, 16])).asType(half),
                "D": (1 + MLXRandom.normal([256], key: keys[7]) * 0.1).asType(half),
                "outProj.weight": normal([64, 256], 5, 1 / 16)
            ]
            for model in [metal, stock] { try model.update(parameters: ModuleParameters.unflattened(weights), verify: .all) }
            let x = normal([2, 200, 64], 8, 1)
            let (a, b) = (metal(x), stock(x))
            eval(a, b)
            let parity = Parity(a, b)
            let snr = -20 * Foundation.log10(max(parity.relativeL2, 1e-12))
            print("RE-USE parity \(half) ReUseMamba layer: \(parity), SNR \(String(format: "%.1f", snr)) dB")
            #expect(a.dtype == half && b.dtype == half)
            #expect(metal.parameters().flattened().allSatisfy { $0.1.dtype == half })
            #expect(parity.identical == parity.count)
        }
    }

    /// The load-time kernel unit self-test must pass on its own fixed inputs in every dtype; its
    /// first half version overflowed FP16 at scale 8 and failed every 16 Oct intake load.
    @Test(.heavy) func kernelGateUnitSelfTestPassesInFP32AndBothHalfDtypes() {
        let result = ReUseKernelGate.selfTest()
        print("RE-USE kernel gate unit self-test: \(result)")
        #expect(result.passed && result.scanMaxAbsError.isFinite && result.convMaxAbsError.isFinite)
    }
}
