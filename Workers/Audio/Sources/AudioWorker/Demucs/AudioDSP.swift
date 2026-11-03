import Foundation

enum AudioDSP {

    static func remixChannels(
        channelMajor input: [Float],
        inputChannels: Int,
        targetChannels: Int,
        frames: Int
    ) -> [Float] {
        if inputChannels == targetChannels {
            return input
        }

        if targetChannels == 1 {
            var mono = [Float](repeating: 0, count: frames)
            for c in 0..<inputChannels {
                let base = c * frames
                for t in 0..<frames {
                    mono[t] += input[base + t]
                }
            }
            let scale = 1.0 / Float(inputChannels)
            for t in 0..<frames {
                mono[t] *= scale
            }
            return mono
        }

        if inputChannels == 1 && targetChannels > 1 {
            var output = [Float](repeating: 0, count: targetChannels * frames)
            for c in 0..<targetChannels {
                let outBase = c * frames
                for t in 0..<frames {
                    output[outBase + t] = input[t]
                }
            }
            return output
        }

        // Truncate or repeat the last channel to fit target count.
        var output = [Float](repeating: 0, count: targetChannels * frames)
        for c in 0..<targetChannels {
            let sourceChannel = min(c, inputChannels - 1)
            let srcBase = sourceChannel * frames
            let dstBase = c * frames
            for t in 0..<frames {
                output[dstBase + t] = input[srcBase + t]
            }
        }
        return output
    }

    static func resampleChannelMajor(
        _ input: [Float],
        channels: Int,
        inputSampleRate: Int,
        targetSampleRate: Int,
        frames: Int
    ) -> (samples: [Float], frames: Int) {
        guard inputSampleRate != targetSampleRate else {
            return (input, frames)
        }

        let ratio = Double(targetSampleRate) / Double(inputSampleRate)
        let outputFrames = max(1, Int((Double(frames) * ratio).rounded()))
        var output = [Float](repeating: 0, count: channels * outputFrames)

        for c in 0..<channels {
            let srcBase = c * frames
            let dstBase = c * outputFrames

            for i in 0..<outputFrames {
                let srcPos = Double(i) / ratio
                let left = min(max(Int(floor(srcPos)), 0), frames - 1)
                let right = min(left + 1, frames - 1)
                let frac = Float(srcPos - Double(left))

                let l = input[srcBase + left]
                let r = input[srcBase + right]
                output[dstBase + i] = l + (r - l) * frac
            }
        }

        return (output, outputFrames)
    }

    static func onePoleLowpass(
        _ input: [Float],
        cutoffHz: Float,
        sampleRate: Int
    ) -> [Float] {
        guard !input.isEmpty else { return [] }
        let dt = 1.0 / Float(sampleRate)
        let rc = 1.0 / (2.0 * Float.pi * max(cutoffHz, 1.0))
        let alpha = dt / (rc + dt)

        var output = [Float](repeating: 0, count: input.count)
        var y = input[0]
        output[0] = y
        for i in 1..<input.count {
            // swiftlint:disable:next shorthand_operator
            y = y + alpha * (input[i] - y)
            output[i] = y
        }
        return output
    }

    static func highpass(
        _ input: [Float],
        cutoffHz: Float,
        sampleRate: Int
    ) -> [Float] {
        let low = onePoleLowpass(input, cutoffHz: cutoffHz, sampleRate: sampleRate)
        var out = [Float](repeating: 0, count: input.count)
        for i in 0..<input.count {
            out[i] = input[i] - low[i]
        }
        return out
    }

    static func triangularWeights(length: Int) -> [Float] {
        if length <= 1 {
            return [1.0]
        }
        let center = Float(length - 1) / 2.0
        let denom = max(center, 1.0)
        var weights = [Float](repeating: 0, count: length)
        for i in 0..<length {
            let distance = abs(Float(i) - center) / denom
            weights[i] = max(1e-3, 1.0 - distance)
        }
        return weights
    }

    @inline(__always)
    static func clampAudio(_ value: Float) -> Float {
        min(1.0, max(-1.0, value))
    }
}
