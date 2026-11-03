// SPDX-License-Identifier: MIT
// Reads a whole audio file as mono float32 at a target rate. The maker averages channels (wav.mean(0))
// and then resamples; this does the same, and drains the converter so no tail samples are lost.
import AVFoundation
import Foundation
import MLX

enum AudioInput {
    static func mono(_ url: URL, sampleRate: Double) throws -> MLXArray {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch {
            throw RunnerError.invalid("\(url.lastPathComponent) is not a readable audio file.")
        }
        let source = file.processingFormat  // deinterleaved float32
        let frames = AVAudioFrameCount(file.length)
        guard frames > 0 else { throw RunnerError.invalid("\(url.lastPathComponent) holds no audio.") }
        guard let input = AVAudioPCMBuffer(pcmFormat: source, frameCapacity: frames) else {
            throw RunnerError.failed("Could not read \(url.lastPathComponent).")
        }
        // One read(into:) can stop short of the end (seen with 32-bit float WAVs): read until the file is drained.
        let channels = Int(source.channelCount)
        var mono = [Float]()
        mono.reserveCapacity(Int(frames))
        while file.framePosition < file.length {
            input.frameLength = 0
            try file.read(into: input)
            let got = Int(input.frameLength)
            if got == 0 { break }
            guard let data = input.floatChannelData else { throw RunnerError.failed("Could not read \(url.lastPathComponent).") }
            var block = [Float](repeating: 0, count: got)
            for c in 0..<channels { let p = data[c]; for i in 0..<got { block[i] += p[i] } }
            mono.append(contentsOf: block)
        }
        let n = mono.count
        guard n > 0 else { throw RunnerError.invalid("\(url.lastPathComponent) holds no audio.") }
        if channels > 1 { let s = 1 / Float(channels); for i in 0..<n { mono[i] *= s } }
        if source.sampleRate == sampleRate { return MLXArray(mono) }

        let inFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: source.sampleRate, channels: 1, interleaved: false)!
        let outFormat = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: sampleRate, channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: inFormat, to: outFormat),
            let monoBuffer = AVAudioPCMBuffer(pcmFormat: inFormat, frameCapacity: AVAudioFrameCount(n))
        else { throw RunnerError.failed("Could not convert \(url.lastPathComponent).") }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        monoBuffer.frameLength = AVAudioFrameCount(n)
        mono.withUnsafeBufferPointer { monoBuffer.floatChannelData![0].update(from: $0.baseAddress!, count: n) }
        let expected = Int((Double(n) * sampleRate / source.sampleRate).rounded(.up))
        var out = [Float]()
        out.reserveCapacity(expected + 4096)
        var fed = false
        while true {
            guard let chunk = AVAudioPCMBuffer(pcmFormat: outFormat, frameCapacity: 65_536) else { throw RunnerError.failed("Out of memory.") }
            var failure: NSError?
            let status = converter.convert(to: chunk, error: &failure) { _, inputStatus in
                if fed { inputStatus.pointee = .endOfStream; return nil }
                fed = true; inputStatus.pointee = .haveData; return monoBuffer
            }
            if let failure { throw RunnerError.failed(failure.localizedDescription) }
            out.append(contentsOf: UnsafeBufferPointer(start: chunk.floatChannelData![0], count: Int(chunk.frameLength)))
            if status == .endOfStream || status == .error || chunk.frameLength == 0 { break }
        }
        if out.count > expected { out.removeLast(out.count - expected) }
        guard !out.isEmpty else { throw RunnerError.invalid("\(url.lastPathComponent) is too short to resample.") }
        return MLXArray(out)
    }
}
