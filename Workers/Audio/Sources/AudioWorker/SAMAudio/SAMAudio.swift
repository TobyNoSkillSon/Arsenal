import Foundation
import MLX
import MLXNN

public enum SAMAudioError: Error, LocalizedError {
    case notImplemented(String)
    case invalidAudioShape([Int])
    case mismatchedBatchCounts
    case invalidStepSize(Float)
    case missingTextMask
    case invalidRepoID(String)
    case modelFilesNotFound(String)
    case noCompatibleWeights
    case missingModelWeights(Int)
    case invalidChunkConfiguration(chunkSeconds: Float, overlapSeconds: Float)
    case unsupportedBatchSize(Int)
    case chunkedAnchorsNotSupported

    public var errorDescription: String? {
        switch self {
        case .notImplemented(let method):
            return "\(method) is not implemented yet."
        case .invalidAudioShape(let shape):
            return "Expected audio shape (batch, 1, samples), got \(shape)."
        case .mismatchedBatchCounts:
            return "Audio, descriptions, and optional conditioning tensors must share the same batch size."
        case .invalidStepSize(let stepSize):
            return "Step size must be between 0 and 1 (exclusive), got \(stepSize)."
        case .missingTextMask:
            return "When passing precomputed text features, you must also pass a text mask."
        case .invalidRepoID(let repo):
            return "Invalid repository ID: \(repo)."
        case .modelFilesNotFound(let path):
            return "Could not find required model files in: \(path)."
        case .noCompatibleWeights:
            return "No compatible weights were found for this SAMAudio model."
        case .missingModelWeights(let count):
            return "Missing \(count) model parameters while strict weight loading is enabled."
        case .invalidChunkConfiguration(let chunkSeconds, let overlapSeconds):
            return "Invalid chunk configuration: chunkSeconds=\(chunkSeconds), overlapSeconds=\(overlapSeconds). Require chunkSeconds > 0 and 0 <= overlapSeconds < chunkSeconds."
        case .unsupportedBatchSize(let batchSize):
            return "This method currently supports batchSize == 1, got \(batchSize)."
        case .chunkedAnchorsNotSupported:
            return "Chunked SAMAudio inference does not currently support anchor IDs/alignment."
        }
    }
}

public final class SAMAudio: Module, @unchecked Sendable {
    var cancellationCheck: (() throws -> Void)?
    public let config: SAMAudioConfig
    public let processor: SAMAudioProcessor

    @ModuleInfo(key: "audio_codec") var audioCodec: DACVAE
    @ModuleInfo(key: "transformer") var transformer: DiT
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "embed_anchors") var embedAnchors: EmbedAnchors
    @ModuleInfo(key: "memory_proj") var memoryProj: Linear

    private let _timestepInvFreq: MLXArray

    public init(config: SAMAudioConfig = SAMAudioConfig()) {
        self.config = config
        self.processor = SAMAudioProcessor(config: config)
        let codebookDim = config.audioCodec.codebookDim
        let transformerDim = config.transformer.dim

        precondition(
            config.inChannels == 6 * codebookDim,
            "SAMAudio expects in_channels = 6 * codebook_dim (\(6 * codebookDim)), got \(config.inChannels)"
        )
        precondition(
            config.transformer.outChannels == 2 * codebookDim,
            "SAMAudio expects transformer.out_channels = 2 * codebook_dim (\(2 * codebookDim)), got \(config.transformer.outChannels)"
        )
        precondition(
            config.transformer.contextDim == config.transformer.dim,
            "SAMAudio expects transformer.context_dim to match transformer.dim (got \(config.transformer.contextDim) vs \(config.transformer.dim))"
        )
        precondition(transformerDim % 2 == 0, "SAMAudio timestep embedding requires an even transformer dim")

        self._audioCodec.wrappedValue = DACVAE(config: config.audioCodec)
        self._transformer.wrappedValue = DiT(config: config.transformer)
        self._proj.wrappedValue = Linear(config.inChannels, transformerDim)
        self._embedAnchors.wrappedValue = EmbedAnchors(
            numEmbeddings: config.numAnchors,
            embeddingDim: config.anchorEmbeddingDim,
            outDim: transformerDim
        )
        self._memoryProj.wrappedValue = Linear(config.textEncoder.dim, transformerDim)

        let halfDim = transformerDim / 2
        self._timestepInvFreq = exp(
            -Foundation.log(Float(10_000))
                * MLXArray(Array(0..<halfDim).map(Float.init)).asType(.float32)
                / Float(halfDim)
        )
    }

    public var sampleRate: Int { audioCodec.sampleRate }

    private func validateChunking(
        totalSamples: Int,
        chunkSeconds: Float,
        overlapSeconds: Float
    ) throws -> (chunkSamples: Int, overlapSamples: Int, hopSamples: Int, numChunks: Int) {
        guard chunkSeconds > 0, overlapSeconds >= 0, overlapSeconds < chunkSeconds else {
            throw SAMAudioError.invalidChunkConfiguration(
                chunkSeconds: chunkSeconds,
                overlapSeconds: overlapSeconds
            )
        }

        let chunkSamples = max(1, Int(Float(sampleRate) * chunkSeconds))
        let overlapSamples = max(0, Int(Float(sampleRate) * overlapSeconds))
        let hopSamples = max(1, chunkSamples - overlapSamples)
        let numChunks = max(
            1,
            Int(ceil(Double(max(totalSamples - overlapSamples, 0)) / Double(hopSamples)))
        )
        return (chunkSamples, overlapSamples, hopSamples, numChunks)
    }

    private func cosineCrossfadeWeights(count: Int) -> (fadeIn: MLXArray, fadeOut: MLXArray) {
        if count <= 0 {
            let empty = MLXArray([] as [Float]).reshaped([0, 1])
            return (empty, empty)
        }

        var fadeInValues: [Float] = []
        fadeInValues.reserveCapacity(count)
        let denom = max(1, count - 1)
        for i in 0..<count {
            let t = Float(i) / Float(denom)
            let value = 0.5 * (1 - Foundation.cos(Float.pi * t))
            fadeInValues.append(value)
        }

        let fadeOutValues = fadeInValues.map { 1 - $0 }
        return (
            MLXArray(fadeInValues).reshaped([count, 1]),
            MLXArray(fadeOutValues).reshaped([count, 1])
        )
    }

    private func appendWithCosineCrossfade(
        chunks: inout [MLXArray],
        newChunk: MLXArray,
        overlapSamples: Int
    ) {
        guard overlapSamples > 0, let previous = chunks.last else {
            chunks.append(newChunk)
            return
        }

        _ = chunks.popLast()

        let prevLen = previous.shape[0]
        let currLen = newChunk.shape[0]
        let overlap = min(overlapSamples, prevLen, currLen)
        guard overlap > 0 else {
            chunks.append(previous)
            chunks.append(newChunk)
            return
        }

        let prevMainLen = prevLen - overlap
        if prevMainLen > 0 {
            chunks.append(previous[0..<prevMainLen, 0...])
        }

        let prevTail = previous[prevMainLen..<prevLen, 0...]
        let currHead = newChunk[0..<overlap, 0...]
        let (fadeIn, fadeOut) = cosineCrossfadeWeights(count: overlap)
        let blended = prevTail * fadeOut + currHead * fadeIn
        chunks.append(blended)

        if currLen > overlap {
            chunks.append(newChunk[overlap..<currLen, 0...])
        }
    }

    private func concatNoiseChunks(_ chunks: [MLXArray]) -> MLXArray? {
        if chunks.isEmpty {
            return nil
        }
        if chunks.count == 1 {
            return chunks[0]
        }
        return MLX.concatenated(chunks, axis: 1)
    }

    private func sinusoidalTimeEmbedding(_ positions: MLXArray) -> MLXArray {
        let pos = positions.asType(.float32)
        let emb = pos.expandedDimensions(axis: 1) * _timestepInvFreq.expandedDimensions(axis: 0)
        return MLX.concatenated([MLX.cos(emb), MLX.sin(emb)], axis: -1)
    }

    public func alignInputs(
        noisyAudio: MLXArray,
        audioFeatures: MLXArray,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil
    ) -> MLXArray {
        let x = MLX.concatenated(
            [noisyAudio, MLXArray.zeros(like: audioFeatures), audioFeatures],
            axis: 2
        )
        let projected = proj(x)
        return embedAnchors(projected, anchorIDs: anchorIDs, anchorAlignment: anchorAlignment)
    }

    public func callAsFunction(
        noisyAudio: MLXArray,
        audioFeatures: MLXArray,
        textFeatures: MLXArray?,
        time: MLXArray,
        textMask: MLXArray? = nil,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil,
        audioPadMask: MLXArray? = nil
    ) -> MLXArray {
        let alignedInputs = alignInputs(
            noisyAudio: noisyAudio,
            audioFeatures: audioFeatures,
            anchorIDs: anchorIDs,
            anchorAlignment: anchorAlignment
        )

        let timestepEmb = sinusoidalTimeEmbedding(time).asType(alignedInputs.dtype).expandedDimensions(axis: 1)
        let memory: MLXArray
        if let textFeatures {
            memory = memoryProj(textFeatures) + timestepEmb
        } else {
            memory = timestepEmb
        }

        return transformer(
            alignedInputs,
            time: time,
            paddingMask: audioPadMask,
            memory: memory,
            memoryPaddingMask: textMask
        )
    }

    private func getAudioFeatures(_ audios: MLXArray) -> MLXArray {
        let audioFeatures = audioCodec(audios).transposed(0, 2, 1)
        return MLX.concatenated([audioFeatures, audioFeatures], axis: 2)
    }

    private func odeStepEuler(
        t: Float,
        dt: Float,
        noisyAudio: MLXArray,
        audioFeatures: MLXArray,
        textFeatures: MLXArray,
        textMask: MLXArray,
        anchorIDs: MLXArray?,
        anchorAlignment: MLXArray?,
        audioPadMask: MLXArray?
    ) -> MLXArray {
        let timeT = MLXArray(Array(repeating: t, count: noisyAudio.shape[0])).asType(.float32)
        let velocity = self(
            noisyAudio: noisyAudio,
            audioFeatures: audioFeatures,
            textFeatures: textFeatures,
            time: timeT,
            textMask: textMask,
            anchorIDs: anchorIDs,
            anchorAlignment: anchorAlignment,
            audioPadMask: audioPadMask
        )
        return noisyAudio + MLXArray(dt) * velocity
    }

    private func odeStepMidpoint(
        t: Float,
        dt: Float,
        noisyAudio: MLXArray,
        audioFeatures: MLXArray,
        textFeatures: MLXArray,
        textMask: MLXArray,
        anchorIDs: MLXArray?,
        anchorAlignment: MLXArray?,
        audioPadMask: MLXArray?
    ) -> MLXArray {
        let batchSize = noisyAudio.shape[0]
        let timeT = MLXArray(Array(repeating: t, count: batchSize)).asType(.float32)
        let velocityT = self(
            noisyAudio: noisyAudio,
            audioFeatures: audioFeatures,
            textFeatures: textFeatures,
            time: timeT,
            textMask: textMask,
            anchorIDs: anchorIDs,
            anchorAlignment: anchorAlignment,
            audioPadMask: audioPadMask
        )

        let midpoint = noisyAudio + MLXArray(0.5 * dt) * velocityT
        let timeMid = MLXArray(Array(repeating: t + 0.5 * dt, count: batchSize)).asType(.float32)
        let velocityMid = self(
            noisyAudio: midpoint,
            audioFeatures: audioFeatures,
            textFeatures: textFeatures,
            time: timeMid,
            textMask: textMask,
            anchorIDs: anchorIDs,
            anchorAlignment: anchorAlignment,
            audioPadMask: audioPadMask
        )

        return noisyAudio + MLXArray(dt) * velocityMid
    }

    public func separate(
        audios: MLXArray,
        descriptions: [String],
        sizes: MLXArray? = nil,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil,
        audioPadMask: MLXArray? = nil,
        noise: MLXArray? = nil,
        ode: SAMAudioODEOptions = .default,
        odeDecodeChunkSize: Int? = nil,
        _textFeatures: MLXArray? = nil,
        _textMask: MLXArray? = nil
    ) throws -> SAMAudioSeparationResult {
        guard audios.ndim == 3, audios.shape[1] == 1 else {
            throw SAMAudioError.invalidAudioShape(audios.shape)
        }
        guard audios.shape[0] == descriptions.count else {
            throw SAMAudioError.mismatchedBatchCounts
        }
        if !(ode.stepSize > 0 && ode.stepSize < 1) {
            throw SAMAudioError.invalidStepSize(ode.stepSize)
        }

        let audioFeatures = getAudioFeatures(audios)
        eval(audioFeatures)

        let batchSize = audioFeatures.shape[0]
        let seqLen = audioFeatures.shape[1]
        let effectiveSizes = (sizes ?? MLXArray(Array(repeating: seqLen, count: batchSize))).asType(.int32)
        let effectiveAudioPadMask = audioPadMask ?? SAMAudioProcessor.maskFromSizes(effectiveSizes)

        let textFeatures: MLXArray
        let textMask: MLXArray
        if let cachedFeatures = _textFeatures {
            guard let cachedMask = _textMask else {
                throw SAMAudioError.missingTextMask
            }
            textFeatures = cachedFeatures
            textMask = cachedMask
        } else {
            throw SAMAudioError.missingTextMask
        }
        eval(textFeatures, textMask)

        var noisyAudio = (noise ?? MLXRandom.normal(audioFeatures.shape)).asType(.float16)
        let numSteps = Int((1.0 / ode.stepSize).rounded())

        for i in 0..<numSteps {
            try cancellationCheck?()
            let t = Float(i) * ode.stepSize
            switch ode.method {
            case .euler:
                noisyAudio = odeStepEuler(
                    t: t,
                    dt: ode.stepSize,
                    noisyAudio: noisyAudio,
                    audioFeatures: audioFeatures,
                    textFeatures: textFeatures,
                    textMask: textMask,
                    anchorIDs: anchorIDs,
                    anchorAlignment: anchorAlignment,
                    audioPadMask: effectiveAudioPadMask
                )
            case .midpoint:
                noisyAudio = odeStepMidpoint(
                    t: t,
                    dt: ode.stepSize,
                    noisyAudio: noisyAudio,
                    audioFeatures: audioFeatures,
                    textFeatures: textFeatures,
                    textMask: textMask,
                    anchorIDs: anchorIDs,
                    anchorAlignment: anchorAlignment,
                    audioPadMask: effectiveAudioPadMask
                )
            }
            eval(noisyAudio)

            if (i + 1) % 4 == 0 {
                Memory.clearCache()
            }
        }

        let generatedFeatures = noisyAudio.transposed(0, 2, 1) // (B, 2*C, T)
        let channels = generatedFeatures.shape[1] / 2
        let targetFeatures = generatedFeatures[0..., 0..<channels, 0...]
        let residualFeatures = generatedFeatures[0..., channels..<generatedFeatures.shape[1], 0...]

        let targetWavs = audioCodec.decode(targetFeatures, chunkSize: odeDecodeChunkSize)
        let residualWavs = audioCodec.decode(residualFeatures, chunkSize: odeDecodeChunkSize)
        eval(targetWavs, residualWavs)

        let featureSizes = effectiveSizes.asArray(Int32.self).map(Int.init)
        var target: [MLXArray] = []
        var residual: [MLXArray] = []
        target.reserveCapacity(batchSize)
        residual.reserveCapacity(batchSize)

        for b in 0..<batchSize {
            let wavSize = min(audioCodec.featureIdxToWavIdx(featureSizes[b]), targetWavs.shape[1])
            target.append(targetWavs[b, 0..<wavSize, 0...])
            residual.append(residualWavs[b, 0..<wavSize, 0...])
        }

        return SAMAudioSeparationResult(
            target: target,
            residual: residual,
            noise: noisyAudio,
            peakMemoryGB: Float(Double(Memory.peakMemory) / 1e9)
        )
    }

    public func separateLong(
        audios: MLXArray,
        descriptions: [String],
        chunkSeconds: Float = 10.0,
        overlapSeconds: Float = 3.0,
        anchorIDs: MLXArray? = nil,
        anchorAlignment: MLXArray? = nil,
        ode: SAMAudioODEOptions = .default,
        odeDecodeChunkSize: Int? = nil,
        _textFeatures: MLXArray? = nil,
        _textMask: MLXArray? = nil
    ) throws -> SAMAudioSeparationResult {
        guard audios.ndim == 3, audios.shape[1] == 1 else {
            throw SAMAudioError.invalidAudioShape(audios.shape)
        }
        guard audios.shape[0] == 1 else {
            throw SAMAudioError.unsupportedBatchSize(audios.shape[0])
        }

        let totalSamples = audios.shape[2]
        let (chunkSamples, overlapSamples, hopSamples, numChunks) = try validateChunking(
            totalSamples: totalSamples,
            chunkSeconds: chunkSeconds,
            overlapSeconds: overlapSeconds
        )

        if totalSamples <= chunkSamples {
            let featureLength = audioCodec.wavIdxToFeatureIdx(totalSamples)
            let sizes = MLXArray([featureLength]).asType(.int32)
            return try separate(
                audios: audios,
                descriptions: descriptions,
                sizes: sizes,
                anchorIDs: anchorIDs,
                anchorAlignment: anchorAlignment,
                ode: ode,
                odeDecodeChunkSize: odeDecodeChunkSize,
                _textFeatures: _textFeatures,
                _textMask: _textMask
            )
        }

        if anchorIDs != nil || anchorAlignment != nil {
            throw SAMAudioError.chunkedAnchorsNotSupported
        }

        guard let features = _textFeatures, let mask = _textMask else { throw SAMAudioError.missingTextMask }
        let text = (features: features, mask: mask)

        var targetChunks: [MLXArray] = []
        var residualChunks: [MLXArray] = []
        var noiseChunks: [MLXArray] = []
        targetChunks.reserveCapacity(numChunks)
        residualChunks.reserveCapacity(numChunks)
        noiseChunks.reserveCapacity(numChunks)

        for i in 0..<numChunks {
            let start = i * hopSamples
            let end = min(start + chunkSamples, totalSamples)
            let chunk = audios[0..., 0..., start..<end]

            let result = try separate(
                audios: chunk,
                descriptions: descriptions,
                ode: ode,
                odeDecodeChunkSize: odeDecodeChunkSize,
                _textFeatures: text.features,
                _textMask: text.mask
            )

            let targetChunk = result.target[0]
            let residualChunk = result.residual[0]
            appendWithCosineCrossfade(chunks: &targetChunks, newChunk: targetChunk, overlapSamples: overlapSamples)
            appendWithCosineCrossfade(chunks: &residualChunks, newChunk: residualChunk, overlapSamples: overlapSamples)

            if let noise = result.noise {
                noiseChunks.append(noise)
            }

            Memory.clearCache()
        }

        let fullTarget = targetChunks.count == 1 ? targetChunks[0] : MLX.concatenated(targetChunks, axis: 0)
        let fullResidual = residualChunks.count == 1 ? residualChunks[0] : MLX.concatenated(residualChunks, axis: 0)
        let fullNoise = concatNoiseChunks(noiseChunks)
        eval(fullTarget, fullResidual)

        return SAMAudioSeparationResult(
            target: [fullTarget],
            residual: [fullResidual],
            noise: fullNoise,
            peakMemoryGB: Float(Double(Memory.peakMemory) / 1e9)
        )
    }

}
