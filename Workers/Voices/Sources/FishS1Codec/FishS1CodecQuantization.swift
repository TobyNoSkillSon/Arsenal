// SPDX-License-Identifier: MIT
// Fish S1 DAC vector quantizers. Derived from mlx-audio-swift FishS1DACQuantization.swift at 01dec7c9 (MIT); numerics and
// the streaming decode follow mlx-audio Python fish_s1_dac.py at dad2087f. See lab/drama3-port/codec-notes.md.
import Foundation
import MLX
import MLXNN

final class FishS1VectorQuantize: Module {
    @ModuleInfo(key: "in_proj") var inProj: FishS1WNConv1d
    @ModuleInfo(key: "out_proj") var outProj: FishS1WNConv1d
    @ModuleInfo(key: "codebook") var codebook: Embedding
    let codebookSize: Int

    init(inputDim: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        _inProj.wrappedValue = FishS1WNConv1d(inChannels: inputDim, outChannels: codebookDim, kernelSize: 1)
        _outProj.wrappedValue = FishS1WNConv1d(inChannels: codebookDim, outChannels: inputDim, kernelSize: 1)
        _codebook.wrappedValue = Embedding(embeddingCount: codebookSize, dimensions: codebookDim)
    }

    /// Codes [B, T] -> latents [B, D, T].
    func decodeCode(_ ids: MLXArray) -> MLXArray {
        codebook.weight.take(ids, axis: 0).transposed(0, 2, 1)
    }

    /// Nearest code by cosine distance, exactly as the reference writes it.
    func decodeLatents(_ latents: MLXArray) -> (MLXArray, MLXArray) {
        let b = latents.dim(0), d = latents.dim(1), t = latents.dim(2)
        let encodings = latents.transposed(0, 2, 1).reshaped(b * t, d)
        let book = codebook.weight
        let encNorm = encodings / maximum(sqrt(sum(pow(encodings, 2), axis: 1, keepDims: true)), 1e-12)
        let codeNorm = book / maximum(sqrt(sum(pow(book, 2), axis: 1, keepDims: true)), 1e-12)
        let dist =
            sum(pow(encNorm, 2), axis: 1, keepDims: true)
            - 2 * matmul(encNorm, codeNorm.T)
            + sum(pow(codeNorm, 2), axis: 1, keepDims: true).T
        let indices = argMax(-dist, axis: 1).reshaped(b, t)
        return (decodeCode(indices), indices)
    }

    /// Encoder path: (projected z_q [B, input_dim, T], indices [B, T]).
    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        let zE = inProj(z)
        let (zQ, indices) = decodeLatents(zE)
        // Straight-through form kept for bit-compatibility with the reference.
        let straight = zE + (zQ - zE)
        return (outProj(straight), indices)
    }
}

final class FishS1ResidualVectorQuantize: Module {
    @ModuleInfo(key: "quantizers") var quantizers: [FishS1VectorQuantize]
    let codebookSize: Int

    init(inputDim: Int, nCodebooks: Int, codebookSize: Int, codebookDim: Int) {
        self.codebookSize = codebookSize
        _quantizers.wrappedValue = (0 ..< nCodebooks).map { _ in
            FishS1VectorQuantize(inputDim: inputDim, codebookSize: codebookSize, codebookDim: codebookDim)
        }
    }

    /// (z_q, codes [B, n, T]) for every quantizer.
    func callAsFunction(_ z: MLXArray) -> (MLXArray, MLXArray) {
        var zQ: MLXArray?
        var residual = z
        var codes: [MLXArray] = []
        for quantizer in quantizers {
            let (zQi, indices) = quantizer(residual)
            zQ = zQ.map { $0 + zQi } ?? zQi
            residual = residual - zQi
            codes.append(indices)
        }
        return (zQ!, stacked(codes, axis: 1))
    }

    /// Codes [B, n, T] -> summed projected latents [B, input_dim, T].
    func fromCodes(_ codes: MLXArray) -> MLXArray {
        var zQ: MLXArray?
        for i in 0 ..< codes.dim(1) {
            let projected = quantizers[i].outProj(quantizers[i].decodeCode(codes[0..., i, 0...]))
            zQ = zQ.map { $0 + projected } ?? projected
        }
        return zQ!
    }
}

/// Python DownsampleResidualVectorQuantize: semantic + residual RVQ between a 4x downsample and upsample, with windowed
/// transformers before and after.
final class FishS1DownsampleResidualVectorQuantize: Module {
    @ModuleInfo(key: "semantic_quantizer") var semanticQuantizer: FishS1ResidualVectorQuantize
    @ModuleInfo(key: "quantizer") var quantizer: FishS1ResidualVectorQuantize
    @ModuleInfo(key: "downsample") var downsample: [[Module]]
    @ModuleInfo(key: "upsample") var upsample: [[Module]]
    @ModuleInfo(key: "pre_module") var preModule: FishS1WindowLimitedTransformer
    @ModuleInfo(key: "post_module") var postModule: FishS1WindowLimitedTransformer

    init(
        inputDim: Int, nCodebooks: Int, codebookDim: Int, codebookSize: Int, semanticCodebookSize: Int, downsampleFactor: [Int],
        preModule: FishS1WindowLimitedTransformer, postModule: FishS1WindowLimitedTransformer
    ) {
        let dims = Array(repeating: inputDim, count: downsampleFactor.count + 1)
        _semanticQuantizer.wrappedValue = FishS1ResidualVectorQuantize(
            inputDim: inputDim, nCodebooks: 1, codebookSize: semanticCodebookSize, codebookDim: codebookDim)
        _quantizer.wrappedValue = FishS1ResidualVectorQuantize(
            inputDim: inputDim, nCodebooks: nCodebooks, codebookSize: codebookSize, codebookDim: codebookDim)
        _downsample.wrappedValue = downsampleFactor.enumerated().map { index, factor in
            [
                FishS1CausalConvNet(inChannels: dims[index], outChannels: dims[index + 1], kernelSize: factor, stride: factor),
                FishS1ConvNeXtBlock(dim: dims[index + 1]),
            ]
        }
        _upsample.wrappedValue = downsampleFactor.enumerated().reversed().map { index, factor in
            [
                FishS1CausalTransConvNet(inChannels: dims[index + 1], outChannels: dims[index], kernelSize: factor, stride: factor),
                FishS1ConvNeXtBlock(dim: dims[index]),
            ]
        }
        _preModule.wrappedValue = preModule
        _postModule.wrappedValue = postModule
    }

    /// Encoder path: latents [B, D, T*4] -> codes [B, 1 + n, T] (semantic first).
    func encodeCodes(_ z: MLXArray) -> MLXArray {
        var x = z
        for block in downsample {
            for layer in block { x = fishS1Call(layer, x) }
        }
        x = preModule(x)
        let (semanticZ, semanticCodes) = semanticQuantizer(x)
        let (_, codes) = quantizer(x - semanticZ)
        return concatenated([semanticCodes, codes], axis: 1)
    }

    func decode(_ indices: MLXArray) -> MLXArray {
        var x = postModule(codesToLatents(indices))
        for block in upsample {
            for layer in block { x = fishS1Call(layer, x) }
        }
        return x
    }

    /// Streaming decode of new frames; see FishS1Codec.streamingStep.
    func decodeStep(_ indices: MLXArray) -> MLXArray {
        var x = postModule.step(codesToLatents(indices))
        for block in upsample {
            for layer in block { x = fishS1Step(layer, x) }
        }
        return x
    }

    /// Codes [B, 1 + n, T], clipped to each codebook's range -> summed latents [B, D, T].
    func codesToLatents(_ indices: MLXArray) -> MLXArray {
        let semantic = clip(indices[0..., 0 ..< 1, 0...], min: 0, max: semanticQuantizer.codebookSize - 1)
        let zSemantic = semanticQuantizer.fromCodes(semantic)
        guard indices.dim(1) > 1 else { return zSemantic + zeros(like: zSemantic) }
        let residual = clip(indices[0..., 1..., 0...], min: 0, max: quantizer.codebookSize - 1)
        return zSemantic + quantizer.fromCodes(residual)
    }
}
