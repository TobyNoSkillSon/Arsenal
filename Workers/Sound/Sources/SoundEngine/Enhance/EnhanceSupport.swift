// SPDX-License-Identifier: MIT
// Vireo enhancement engines: shared plumbing (channel split, resampling,
// output validation). No app names; plain Swift + AVFoundation.

import AVFoundation
import ArsenalWire
import Foundation
import VireoCore

/// A caller mistake (bad option, unusable input): invalid_request, which the service returns as HTTP 422. EngineError
/// is the engine's own failure and returns 500.
func invalidInput(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }

enum EnhanceAudio {
    /// De-interleave into one `[Float]` per channel.
    static func channels(of buffer: AudioClip) -> [[Float]] {
        let count = buffer.channels
        guard count > 1 else { return [buffer.samples] }
        let frames = buffer.frames
        var out = [[Float]](repeating: [Float](repeating: 0, count: frames), count: count)
        for f in 0..<frames {
            for c in 0..<count { out[c][f] = buffer.samples[f * count + c] }
        }
        return out
    }

    /// Interleave equal-length channels.
    static func interleave(_ channels: [[Float]]) -> [Float] {
        guard channels.count > 1 else { return channels.first ?? [] }
        let frames = channels.map(\.count).min() ?? 0
        var out = [Float](repeating: 0, count: frames * channels.count)
        for f in 0..<frames {
            for c in channels.indices { out[f * channels.count + c] = channels[c][f] }
        }
        return out
    }

    /// Band-limited sample-rate conversion (AVAudioConverter, mastering quality).
    /// Output length is exactly `round(count * to / from)`.
    static func resample(_ samples: [Float], from: Int, to: Int) throws -> [Float] {
        guard from != to, !samples.isEmpty else { return samples }
        guard from > 0, to > 0,
              let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(from), channels: 1, interleaved: false),
              let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(to), channels: 1, interleaved: false),
              let converter = AVAudioConverter(from: inFormat, to: outFormat),
              let input = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(samples.count))
        else { throw EngineError("cannot resample \(from) Hz → \(to) Hz") }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        input.frameLength = AVAudioFrameCount(samples.count)
        samples.withUnsafeBufferPointer { src in
            input.floatChannelData![0].update(from: src.baseAddress!, count: samples.count)
        }
        let expected = Int((Double(samples.count) * Double(to) / Double(from)).rounded())
        guard let output = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: AVAudioFrameCount(expected + 4096)) else {
            throw EngineError("cannot allocate resample buffer")
        }
        var consumed = false
        var error: NSError?
        var result = [Float]()
        result.reserveCapacity(expected)
        while true {
            output.frameLength = 0
            let status = converter.convert(to: output, error: &error) { _, outStatus in
                if consumed { outStatus.pointee = .endOfStream; return nil }
                consumed = true
                outStatus.pointee = .haveData
                return input
            }
            if let error { throw EngineError("resample failed: \(error.localizedDescription)") }
            let n = Int(output.frameLength)
            if n > 0 { result.append(contentsOf: UnsafeBufferPointer(start: output.floatChannelData![0], count: n)) }
            if status == .endOfStream || status == .error || (n == 0 && consumed) { break }
        }
        if result.count > expected { result.removeLast(result.count - expected) }
        if result.count < expected { result.append(contentsOf: repeatElement(0, count: expected - result.count)) }
        return result
    }

    /// Engine contract: never return non-finite samples.
    static func requireFinite(_ samples: [Float], model: String) throws {
        if let bad = samples.firstIndex(where: { !$0.isFinite }) {
            throw EngineError("\(model) produced a non-finite sample at frame \(bad)")
        }
    }
}
