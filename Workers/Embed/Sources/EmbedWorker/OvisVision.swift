// swiftlint:disable shorthand_operator
// Copyright © 2026 Apple Inc.
import Foundation
import ArsenalWire
import IntegrationKit
import MLX
import MLXNN
import MLXLMCommon
import MLXVLM
enum OvisVision {

    static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let first = x[.ellipsis, 0..<half]
        let second = x[.ellipsis, half...]
        return concatenated([-second, first], axis: -1)
    }

    static func applyRotary(_ tensor: MLXArray, freqs: MLXArray) -> MLXArray {
        var cosVals = cos(freqs)
        var sinVals = sin(freqs)

        cosVals = expandedDimensions(cosVals, axis: 1)
        cosVals = tiled(cosVals, repetitions: [1, 1, 2])
        cosVals = expandedDimensions(cosVals, axis: 0)

        sinVals = expandedDimensions(sinVals, axis: 1)
        sinVals = tiled(sinVals, repetitions: [1, 1, 2])
        sinVals = expandedDimensions(sinVals, axis: 0)

        let rotated = (tensor * cosVals) + (rotateHalf(tensor) * sinVals)
        return rotated.asType(tensor.dtype)
    }

    final class VisionRotaryEmbedding {
        let dimension: Int
        let theta: Float

        init(dimension: Int, theta: Float = 10_000) {
            self.dimension = dimension
            self.theta = theta
        }

        func callAsFunction(sequenceLength: Int) -> MLXArray {
            let invFreq =
                1.0
                / pow(
                    MLXArray(theta),
                    MLXArray(stride(from: 0, to: dimension, by: 2)).asType(.float32)
                        / Float(dimension)
                )
            let seq = MLXArray(0..<sequenceLength).asType(invFreq.dtype)
            return outer(seq, invFreq)
        }
    }

    final class PatchEmbed: Module, UnaryLayer {
        @ModuleInfo(key: "proj") var proj: Conv3d

        let patchSize: Int
        let temporalPatchSize: Int
        let inChannels: Int
        let hiddenSize: Int

        init(
            patchSize: Int,
            temporalPatchSize: Int,
            inChannels: Int,
            hiddenSize: Int
        ) {
            self.patchSize = patchSize
            self.temporalPatchSize = temporalPatchSize
            self.inChannels = inChannels
            self.hiddenSize = hiddenSize

            let kernel = IntOrTriple([temporalPatchSize, patchSize, patchSize])
            _proj.wrappedValue = Conv3d(
                inputChannels: inChannels,
                outputChannels: hiddenSize,
                kernelSize: kernel,
                stride: kernel,
                bias: true
            )
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x.reshaped(
                -1,
                inChannels,
                temporalPatchSize,
                patchSize,
                patchSize
            ).movedAxis(source: 1, destination: 4)

            states = proj(states)
            states = states.reshaped(-1, hiddenSize)
            return states
        }
    }

    final class PatchMerger: Module, UnaryLayer {
        let hiddenSize: Int
        let usePostShuffleNorm: Bool

        @ModuleInfo(key: "norm") var norm: LayerNorm
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(config: MLXVLM.Qwen3VLConfiguration.VisionConfiguration, usePostShuffleNorm: Bool) {
            self.hiddenSize =
                config.hiddenSize * (config.spatialMergeSize * config.spatialMergeSize)
            self.usePostShuffleNorm = usePostShuffleNorm

            let normDim = usePostShuffleNorm ? hiddenSize : config.hiddenSize
            _norm.wrappedValue = LayerNorm(dimensions: normDim, eps: 1e-6)
            _linear1.wrappedValue = Linear(hiddenSize, hiddenSize)
            _linear2.wrappedValue = Linear(hiddenSize, config.outHiddenSize)
            _activation.wrappedValue = GELU()
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            var states = x
            if usePostShuffleNorm {
                states = states.reshaped(-1, hiddenSize)
            }
            states = norm(states)
            states = states.reshaped(-1, hiddenSize)
            states = linear1(states)
            states = activation(states)
            states = linear2(states)
            return states
        }
    }

    final class Attention: Module {
        let numHeads: Int
        let headDim: Int
        let scale: Float

        @ModuleInfo(key: "qkv") var qkv: Linear
        @ModuleInfo(key: "proj") var proj: Linear

        init(dim: Int, numHeads: Int) {
            self.numHeads = numHeads
            self.headDim = dim / numHeads
            self.scale = pow(Float(headDim), -0.5)

            _qkv.wrappedValue = Linear(dim, 3 * dim, bias: true)
            _proj.wrappedValue = Linear(dim, dim)
        }

        func callAsFunction(
            _ x: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            let sequenceLength = x.dim(0)

            var qkvStates = qkv(x)
            qkvStates = qkvStates.reshaped(sequenceLength, 3, numHeads, headDim)
            qkvStates = qkvStates.transposed(1, 0, 2, 3)

            let parts = split(qkvStates, parts: 3, axis: 0)
            var queries = parts[0][0, 0..., 0..., 0...]
            var keys = parts[1][0, 0..., 0..., 0...]
            var values = parts[2][0, 0..., 0..., 0...]

            queries = applyRotary(queries, freqs: rotaryPosEmb)
            keys = applyRotary(keys, freqs: rotaryPosEmb)

            queries = queries.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            keys = keys.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)
            values = values.reshaped(1, sequenceLength, numHeads, headDim).transposed(0, 2, 1, 3)

            let attended = Self.segmentedAttention(
                queries: queries, keys: keys, values: values,
                boundaries: cuSeqlens.asArray(Int.self), scale: scale
            )
            .transposed(0, 2, 1, 3).reshaped(sequenceLength, -1)

            return proj(attended)
        }

        // Each image / temporal group is independent. Never allocate a mask over the
        // concatenated document: memory is bounded by the largest segment.
        static func segmentedAttention(queries: MLXArray, keys: MLXArray, values: MLXArray, boundaries: [Int], scale: Float) -> MLXArray {
            precondition(boundaries.first == 0 && boundaries.last == queries.dim(2))
            var segments: [MLXArray] = []
            for index in 1..<boundaries.count {
                let range = boundaries[index - 1]..<boundaries[index]
                precondition(!range.isEmpty)
                let result = MLXFast.scaledDotProductAttention(
                    queries: queries[0..., 0..., range, 0...],
                    keys: keys[0..., 0..., range, 0...],
                    values: values[0..., 0..., range, 0...], scale: scale, mask: .none)
                eval(result)
                segments.append(result)
            }
            return concatenated(segments, axis: 2)
        }
    }

    final class MLP: Module, UnaryLayer {
        @ModuleInfo(key: "linear_fc1") var linear1: Linear
        @ModuleInfo(key: "linear_fc2") var linear2: Linear
        @ModuleInfo(key: "act") var activation: GELU

        init(dim: Int, hiddenDim: Int) {
            _linear1.wrappedValue = Linear(dim, hiddenDim, bias: true)
            _linear2.wrappedValue = Linear(hiddenDim, dim, bias: true)
            _activation.wrappedValue = GELU(approximation: .tanh)
        }

        func callAsFunction(_ x: MLXArray) -> MLXArray {
            linear2(activation(linear1(x)))
        }
    }

    final class VisionBlock: Module {
        @ModuleInfo(key: "norm1") var norm1: LayerNorm
        @ModuleInfo(key: "norm2") var norm2: LayerNorm
        @ModuleInfo(key: "attn") var attention: Attention
        @ModuleInfo(key: "mlp") var mlp: MLP

        init(_ config: MLXVLM.Qwen3VLConfiguration.VisionConfiguration) {
            _norm1.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _norm2.wrappedValue = LayerNorm(dimensions: config.hiddenSize, eps: 1e-6)
            _attention.wrappedValue = Attention(dim: config.hiddenSize, numHeads: config.numHeads)
            _mlp.wrappedValue = MLP(dim: config.hiddenSize, hiddenDim: config.intermediateSize)
        }

        func callAsFunction(
            _ hiddenStates: MLXArray,
            cuSeqlens: MLXArray,
            rotaryPosEmb: MLXArray
        ) -> MLXArray {
            var states = hiddenStates
            states =
                states + attention(norm1(states), cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryPosEmb)
            states = states + mlp(norm2(states))
            return states
        }
    }

    final class VisionModel: Module {

        let config: MLXVLM.Qwen3VLConfiguration.VisionConfiguration
        let spatialMergeSize: Int
        let numGridPerSide: Int

        @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
        @ModuleInfo(key: "rotary_pos_emb") var rotaryEmbedding: VisionRotaryEmbedding
        @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
        @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
        @ModuleInfo(key: "merger") var merger: PatchMerger
        @ModuleInfo(key: "deepstack_merger_list") var deepstackMergers: [PatchMerger]
        let deepstackVisualIndexes: [Int]

        init(_ config: MLXVLM.Qwen3VLConfiguration.VisionConfiguration) {
            self.config = config
            self.spatialMergeSize = config.spatialMergeSize
            self.numGridPerSide = Int(sqrt(Double(config.numPositionEmbeddings)))
            self.deepstackVisualIndexes = config.deepstackVisualIndexes

            _patchEmbed.wrappedValue = PatchEmbed(
                patchSize: config.patchSize,
                temporalPatchSize: config.temporalPatchSize,
                inChannels: config.inChannels,
                hiddenSize: config.hiddenSize)

            let headDim = config.hiddenSize / config.numHeads
            _rotaryEmbedding.wrappedValue = VisionRotaryEmbedding(dimension: headDim / 2)

            _posEmbed.wrappedValue = Embedding(
                embeddingCount: config.numPositionEmbeddings,
                dimensions: config.hiddenSize)

            _blocks.wrappedValue = (0..<config.depth).map { _ in VisionBlock(config) }
            _merger.wrappedValue = PatchMerger(config: config, usePostShuffleNorm: false)

            _deepstackMergers.wrappedValue = config.deepstackVisualIndexes.map { _ in
                PatchMerger(config: config, usePostShuffleNorm: true)
            }
        }

        private func rotaryPositionEmbedding(_ grids: [THW]) -> MLXArray {
            guard let maxHW = grids.map({ max($0.h, $0.w) }).max(), maxHW > 0 else {
                return MLXArray.zeros([1, 1], dtype: .float32)
            }

            let freqTable = rotaryEmbedding(sequenceLength: maxHW)
            let halfDim = freqTable.dim(-1)

            let merge = spatialMergeSize
            var allCoords: [MLXArray] = []
            let mergeScalar = MLXArray(Int32(merge))

            for grid in grids {
                let mergedH = grid.h / merge
                let mergedW = grid.w / merge

                guard mergedH > 0, mergedW > 0 else { continue }

                // Generate block and intra-block indices fully in MLX
                var blockRows = MLXArray(0..<mergedH).asType(.int32)
                blockRows = blockRows.reshaped([mergedH, 1, 1, 1])

                var blockCols = MLXArray(0..<mergedW).asType(.int32)
                blockCols = blockCols.reshaped([1, mergedW, 1, 1])

                let intra = MLXArray(0..<merge).asType(.int32)
                let intraRow = intra.reshaped([1, 1, merge, 1])
                let intraCol = intra.reshaped([1, 1, 1, merge])

                // Broadcast arithmetic mirrors the Python implementation
                var hIndex = blockRows * mergeScalar + intraRow
                var wIndex = blockCols * mergeScalar + intraCol

                hIndex = broadcast(hIndex, to: [mergedH, mergedW, merge, merge])
                wIndex = broadcast(wIndex, to: [mergedH, mergedW, merge, merge])

                // Flatten and stack coordinate pairs
                let hFlattened = hIndex.flattened()
                let wFlattened = wIndex.flattened()
                var coords = stacked([hFlattened, wFlattened], axis: -1)

                // Repeat for temporal frames
                if grid.t > 1 {
                    coords = tiled(coords, repetitions: [grid.t, 1])
                }

                allCoords.append(coords)
            }

            guard !allCoords.isEmpty else {
                return MLXArray.zeros([0, halfDim * 2], dtype: freqTable.dtype)
            }

            // Concatenate all coordinate pairs
            let allCoordsConcat = concatenated(allCoords, axis: 0) // (total_tokens, 2)

            // Extract h and w indices and lookup embeddings
            let hIndices = allCoordsConcat[0..., 0].asType(.int32)
            let wIndices = allCoordsConcat[0..., 1].asType(.int32)

            let hEmbeds = freqTable[hIndices, 0...]
            let wEmbeds = freqTable[wIndices, 0...]

            // Concatenate height and width embeddings
            return concatenated([hEmbeds, wEmbeds], axis: -1)
        }

        func positionalEmbeddings(_ grids: [THW]) -> MLXArray {
            let embeddingType = posEmbed(MLXArray([0])).dtype
            let hiddenSize = config.hiddenSize
            let maxIndex = numGridPerSide - 1

            // Step 1: Collect all indices and weights from all grids using MLX ops
            var cornerIndices: [[MLXArray]] = Array(repeating: [], count: 4)
            var cornerWeights: [[MLXArray]] = Array(repeating: [], count: 4)
            var gridSizes: [Int] = []

            for grid in grids {
                let h = grid.h
                let w = grid.w
                gridSizes.append(h * w)

                // Create linspace indices using broadcasting
                var hLinspace = MLXArray(0..<h).asType(.float32)
                hLinspace = hLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, h - 1)))

                var wLinspace = MLXArray(0..<w).asType(.float32)
                wLinspace = wLinspace * MLXArray(Float(maxIndex)) / MLXArray(Float(max(1, w - 1)))

                // Get floor/ceil and deltas
                let hFloor = hLinspace.asType(.int32)
                let hCeil = minimum(hFloor + 1, maxIndex)
                let dh = hLinspace - hFloor.asType(.float32)

                let wFloor = wLinspace.asType(.int32)
                let wCeil = minimum(wFloor + 1, maxIndex)
                let dw = wLinspace - wFloor.asType(.float32)

                // Broadcast to create meshgrid
                let hFloorExpanded = expandedDimensions(hFloor, axis: 1) // (h, 1)
                let hCeilExpanded = expandedDimensions(hCeil, axis: 1)
                let wFloorExpanded = expandedDimensions(wFloor, axis: 0) // (1, w)
                let wCeilExpanded = expandedDimensions(wCeil, axis: 0)

                let baseH = hFloorExpanded * numGridPerSide
                let baseHCeil = hCeilExpanded * numGridPerSide

                // Compute 4 corner indices
                cornerIndices[0].append((baseH + wFloorExpanded).flattened())
                cornerIndices[1].append((baseH + wCeilExpanded).flattened())
                cornerIndices[2].append((baseHCeil + wFloorExpanded).flattened())
                cornerIndices[3].append((baseHCeil + wCeilExpanded).flattened())

                // Compute bilinear weights
                let dhExpanded = expandedDimensions(dh, axis: 1)
                let dwExpanded = expandedDimensions(dw, axis: 0)

                cornerWeights[0].append(((1 - dhExpanded) * (1 - dwExpanded)).flattened())
                cornerWeights[1].append(((1 - dhExpanded) * dwExpanded).flattened())
                cornerWeights[2].append((dhExpanded * (1 - dwExpanded)).flattened())
                cornerWeights[3].append((dhExpanded * dwExpanded).flattened())
            }

            guard !cornerIndices[0].isEmpty else {
                return MLXArray.zeros([0, hiddenSize], dtype: embeddingType)
            }

            // Step 2: Batch embedding lookup
            let indicesTensors = cornerIndices.map { concatenated($0, axis: 0).asType(.int32) }
            let weightsTensors = cornerWeights.map {
                concatenated($0, axis: 0).asType(embeddingType)
            }

            let totalPatches = indicesTensors[0].dim(0)
            var patchPosEmbeds = MLXArray.zeros(
                [totalPatches, hiddenSize], dtype: embeddingType)

            for cornerIdx in 0..<4 {
                let cornerEmbeds = posEmbed(indicesTensors[cornerIdx])
                let weighted =
                    cornerEmbeds * expandedDimensions(weightsTensors[cornerIdx], axis: -1)
                patchPosEmbeds = patchPosEmbeds + weighted
            }

            // Step 3: Split by grid (like Python lines 344-349)
            var patchPosEmbedsSplit: [MLXArray] = []
            var offset = 0

            for size in gridSizes {
                let slice = patchPosEmbeds[offset..<(offset + size), 0...]
                patchPosEmbedsSplit.append(slice)
                offset += size
            }

            // Step 4: Process each grid (like Python lines 354-371)
            var resultEmbeds: [MLXArray] = []
            let merge = spatialMergeSize

            for (gridIdx, grid) in grids.enumerated() {
                let posEmbed = patchPosEmbedsSplit[gridIdx]
                let h = grid.h
                let w = grid.w
                let t = grid.t

                let featureDim = posEmbed.dim(-1)

                // Repeat for temporal dimension
                var temporalEmbeds = tiled(posEmbed, repetitions: [t, 1])

                // Reshape for merge pattern
                temporalEmbeds = temporalEmbeds.reshaped(t, h, w, featureDim)
                temporalEmbeds = temporalEmbeds.reshaped(
                    t,
                    h / merge,
                    merge,
                    w / merge,
                    merge,
                    featureDim
                )
                temporalEmbeds = temporalEmbeds.transposed(0, 1, 3, 2, 4, 5)
                temporalEmbeds = temporalEmbeds.reshaped(-1, featureDim)

                resultEmbeds.append(temporalEmbeds)
            }

            return concatenated(resultEmbeds, axis: 0)
        }

        private func cumulativeSequenceLengths(_ grids: [THW]) -> MLXArray {
            var seqLengths: [MLXArray] = []

            for grid in grids {
                let perFrame = grid.h * grid.w
                let repeated = tiled(MLXArray(perFrame), repetitions: [grid.t])
                seqLengths.append(repeated)
            }

            guard !seqLengths.isEmpty else {
                return MLXArray(0, dtype: .int32)
            }

            let concatSeqLengths = concatenated(seqLengths).asType(.int32)

            let cumSum = concatSeqLengths.cumsum()

            return padded(
                cumSum, widths: [IntOrPair((1, 0))], mode: .constant,
                value: MLXArray(0, dtype: cumSum.dtype))
        }

        func callAsFunction(_ pixelValues: MLXArray, gridTHW: [THW], cancellation: NativeCancellation) throws -> (MLXArray, [MLXArray]) {
            var hiddenStates = patchEmbed(pixelValues)

            let posEmbeds = positionalEmbeddings(gridTHW)
            hiddenStates = hiddenStates + posEmbeds

            let rotaryEmbeds = rotaryPositionEmbedding(gridTHW)
            let cuSeqlens = cumulativeSequenceLengths(gridTHW)

            var deepstackOutputs: [MLXArray] = []

            for (index, block) in blocks.enumerated() {
                try cancellation.check()
                hiddenStates = block(hiddenStates, cuSeqlens: cuSeqlens, rotaryPosEmb: rotaryEmbeds)
                eval(hiddenStates)
                if let dsIndex = deepstackVisualIndexes.firstIndex(of: index) {
                    let feature = deepstackMergers[dsIndex](hiddenStates)
                    deepstackOutputs.append(feature)
                }
            }

            hiddenStates = merger(hiddenStates)
            return (hiddenStates, deepstackOutputs)
        }

        func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
            var sanitized: [String: MLXArray] = [:]
            for (key, value) in weights {
                if key.contains("position_ids") {
                    continue
                } else if key.contains("patch_embed.proj.weight") {
                    if value.ndim == 5 && value.dim(-1) == config.inChannels {
                        sanitized[key] = value
                    } else {
                        sanitized[key] = value.transposed(0, 2, 3, 4, 1)
                    }
                } else {
                    sanitized[key] = value
                }
            }
            return sanitized
        }
    }
}

enum OvisPositions {
    // Qwen3-VL splits each video's grid into t independent [1,h,w] grids. The
    // timestamp and vision delimiters are text and advance the common offset.
    static func getRopeIndex(
        inputIds: MLXArray, imageGridTHW: [THW]?, videoGridTHW: [THW]?,
        spatialMergeSize: Int, imageTokenId: Int, videoTokenId: Int,
        visionStartTokenId: Int
    ) throws -> (MLXArray, MLXArray) {
        guard inputIds.dim(0) == 1 else { throw WorkerFailure(code: "invalid_request", message: "Ovis accepts one sequence per forward pass.") }
        let tokens = inputIds.asArray(Int.self)
        let videos = (videoGridTHW ?? []).flatMap { grid in (0..<grid.t).map { _ in THW(1, grid.h, grid.w) } }
        let images = imageGridTHW ?? []
        var axes = Array(repeating: [Int](), count: 3)
        var cursor = 0, offset = 0, imageIndex = 0, videoIndex = 0
        while cursor < tokens.count {
            let marker = tokens[cursor]
            if marker == imageTokenId || marker == videoTokenId {
                let grid: THW
                if marker == imageTokenId {
                    guard imageIndex < images.count else { throw WorkerFailure(code: "invalid_request", message: "Unexpected image token.") }
                    grid = images[imageIndex]; imageIndex += 1
                } else {
                    guard videoIndex < videos.count else { throw WorkerFailure(code: "invalid_request", message: "Unexpected video token.") }
                    grid = videos[videoIndex]; videoIndex += 1
                }
                let height = grid.h / spatialMergeSize, width = grid.w / spatialMergeSize
                let count = grid.t * height * width
                guard count > 0, cursor + count <= tokens.count, tokens[cursor..<(cursor + count)].allSatisfy({ $0 == marker }) else {
                    throw WorkerFailure(code: "invalid_request", message: "Ovis visual grid does not match its token block.")
                }
                for time in 0..<grid.t {
                    for row in 0..<height {
                        for column in 0..<width {
                            axes[0].append(offset + time); axes[1].append(offset + row); axes[2].append(offset + column)
                        }
                    }
                }
                offset += max(height, width)
                cursor += count
            } else {
                for axis in 0..<3 { axes[axis].append(offset) }
                offset += 1; cursor += 1
            }
        }
        guard imageIndex == images.count && videoIndex == videos.count else {
            throw WorkerFailure(code: "invalid_request", message: "Ovis visual grids were not consumed.")
        }
        return (MLXArray(axes.flatMap { $0 }).reshaped(3, 1, tokens.count).asType(.int32), MLXArray([offset - tokens.count]).reshaped(1, 1))
    }
}

// swiftlint:enable shorthand_operator
