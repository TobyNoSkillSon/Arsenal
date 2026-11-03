// SPDX-License-Identifier: MIT
// Vendored from Blaizzy/mlx-audio-swift @ 0dda61b29c308175e302dceef77c3bd84551e92b (Sources/MLXAudioSTS/Models/DeepFilterNet,
// MIT, licenses/mlx-audio-swift-MIT.txt). Arsenal changes: version 1, CPU feature loops and the Accelerate GRU removed;
// folded batch norm and transposed-conv weights keep the half dtype. See DeepFilterNetModel.swift.
// swiftlint:disable all
import Foundation
import MLX

// MARK: - ERB & Utility

extension DeepFilterNetModel {

    static func libdfFreqToErb(_ freqHz: Float) -> Float {
        9.265 * log1p(freqHz / (24.7 * 9.265))
    }

    static func libdfErbToFreq(_ erb: Float) -> Float {
        24.7 * 9.265 * (exp(erb / 9.265) - 1.0)
    }

    static func libdfErbBandWidths(
        sampleRate: Int,
        fftSize: Int,
        nbBands: Int,
        minNbFreqs: Int
    ) -> [Int] {
        guard sampleRate > 0, fftSize > 0, nbBands > 0 else { return [] }

        let nyq = sampleRate / 2
        let freqWidth = Float(sampleRate) / Float(fftSize)
        let erbLow = libdfFreqToErb(0)
        let erbHigh = libdfFreqToErb(Float(nyq))
        let step = (erbHigh - erbLow) / Float(nbBands)

        var widths = [Int](repeating: 0, count: nbBands)
        var prevFreq = 0
        var freqOver = 0
        let minBins = max(1, minNbFreqs)

        for i in 1...nbBands {
            let f = libdfErbToFreq(erbLow + Float(i) * step)
            let fb = Int((f / freqWidth).rounded())
            var nbFreqs = fb - prevFreq - freqOver
            if nbFreqs < minBins {
                freqOver = minBins - nbFreqs
                nbFreqs = minBins
            } else {
                freqOver = 0
            }
            widths[i - 1] = max(1, nbFreqs)
            prevFreq = fb
        }

        widths[nbBands - 1] += 1 // fft_size/2 + 1 bins
        let target = fftSize / 2 + 1
        let total = widths.reduce(0, +)
        if total > target {
            widths[nbBands - 1] -= (total - target)
        } else if total < target {
            widths[nbBands - 1] += (target - total)
        }
        return widths
    }

    static func vorbisWindow(size: Int) -> MLXArray {
        let half = max(1, size / 2)
        var window = [Float](repeating: 0, count: size)
        for i in 0..<size {
            let inner = sin(0.5 * Float.pi * (Float(i) + 0.5) / Float(half))
            window[i] = sin(0.5 * Float.pi * inner * inner)
        }
        return MLXArray(window)
    }

    static func linspace(start: Float, end: Float, count: Int) -> [Float] {
        guard count > 1 else { return [start] }
        let step = (end - start) / Float(count - 1)
        return (0..<count).map { start + Float($0) * step }
    }

    func w(_ key: String) throws -> MLXArray {
        guard let value = weights[key] else {
            throw DeepFilterNetError.missingWeightKey(key)
        }
        return value
    }
}

// MARK: - Weight Cache Builders

extension DeepFilterNetModel {

    static func buildBatchNormAffine(weights: [String: MLXArray]) -> ([String: MLXArray], [String: MLXArray]) {
        var scaleByPrefix = [String: MLXArray]()
        var biasByPrefix = [String: MLXArray]()

        for (key, mean) in weights where key.hasSuffix(".running_mean") {
            let prefix = String(key.dropLast(".running_mean".count))
            guard let gamma = weights["\(prefix).weight"],
                let beta = weights["\(prefix).bias"],
                let variance = weights["\(prefix).running_var"]
            else {
                continue
            }
            // Folded in float32, stored in the weights' half dtype so activations stay half.
            let scale = gamma.asType(.float32) / MLX.sqrt(variance.asType(.float32) + 1e-5)
            let shift = beta.asType(.float32) - mean.asType(.float32) * scale
            scaleByPrefix[prefix] = scale.reshaped([1, scale.shape[0], 1, 1]).asType(gamma.dtype)
            biasByPrefix[prefix] = shift.reshaped([1, shift.shape[0], 1, 1]).asType(gamma.dtype)
        }

        return (scaleByPrefix, biasByPrefix)
    }

    static func buildConv2dWeightCache(weights: [String: MLXArray]) -> [String: MLXArray] {
        var cache = [String: MLXArray]()
        for (key, weight) in weights where key.hasSuffix(".weight") && weight.ndim == 4 {
            cache[key] = weight.transposed(0, 2, 3, 1)
        }
        return cache
    }

    static func buildGRUTransposedWeightCache(weights: [String: MLXArray]) -> [String: MLXArray] {
        var cache = [String: MLXArray]()
        cache.reserveCapacity(24)
        for (key, weight) in weights where key.contains(".gru.weight_") && weight.ndim == 2 {
            cache[key] = weight.transposed()
        }
        return cache
    }

    static func buildGroupedTransposeWeights(
        weights: [String: MLXArray],
        groups: Int
    ) -> [String: [MLXArray]] {
        guard groups > 1 else { return [:] }
        var cache = [String: [MLXArray]]()
        for (key, weight) in weights where key.hasSuffix(".0.weight") && weight.ndim == 4 {
            guard weight.shape[0] % groups == 0 else { continue }
            let inPerGroup = weight.shape[0] / groups
            var grouped = [MLXArray]()
            grouped.reserveCapacity(groups)
            for g in 0..<groups {
                let inStart = g * inPerGroup
                let inEnd = inStart + inPerGroup
                let wg = weight[inStart..<inEnd, 0..., 0..., 0...]
                grouped.append(wg.transposed(1, 2, 3, 0))
            }
            cache[key] = grouped
        }
        return cache
    }

    static func buildDenseTransposeWeights(
        weights: [String: MLXArray],
        groups: Int
    ) -> [String: MLXArray] {
        guard groups > 1 else { return [:] }
        var cache = [String: MLXArray]()
        for (key, weight) in weights where key.hasSuffix(".0.weight") && weight.ndim == 4 {
            guard weight.shape[0] % groups == 0 else { continue }
            let inPerGroup = weight.shape[0] / groups
            let outPerGroup = weight.shape[1]
            let kT = weight.shape[2]
            let kF = weight.shape[3]
            let totalIn = inPerGroup * groups

            var outBlocks = [MLXArray]()
            outBlocks.reserveCapacity(groups)
            for g in 0..<groups {
                let inStart = g * inPerGroup
                let inEnd = inStart + inPerGroup
                let wg = weight[inStart..<inEnd, 0..., 0..., 0...].transposed(1, 2, 3, 0)
                let leftChannels = g * inPerGroup
                let rightChannels = totalIn - leftChannels - inPerGroup
                let left = MLXArray.zeros([outPerGroup, kT, kF, leftChannels], dtype: weight.dtype)
                let right = MLXArray.zeros([outPerGroup, kT, kF, rightChannels], dtype: weight.dtype)
                outBlocks.append(MLX.concatenated([left, wg, right], axis: 3))
            }
            cache[key] = MLX.concatenated(outBlocks, axis: 0)
        }
        return cache
    }
}
// swiftlint:enable all
