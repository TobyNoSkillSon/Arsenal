import AVFoundation
import Foundation
import IntegrationKit
import Testing
@testable import EmbedWorker

/// LCO-Embedding Omni's media front end against the makers' processor pixel values (Fixtures/lco_media_capture.py,
/// lco-media-reference.json): no weights, no MLX.
private struct MediaReference: Decodable {
    struct Entry: Decodable { let grid: [Int]; let count: Int; let samples: [Float] }
    let stride: Int
    let resize, alpha, video: Entry
    let videoResize: Entry
    enum CodingKeys: String, CodingKey { case stride, resize, alpha, video, videoResize = "video_resize" }
}

private func fixture(_ name: String, _ ext: String) throws -> URL {
    try #require(Bundle.module.url(forResource: name, withExtension: ext, subdirectory: "Fixtures"))
}

private func pixels(image name: String) throws -> ([Float], LCOGrid) {
    let image = try LCOImageIO.load(try fixture(name, "png"))
    let (h, w) = try LCOMedia.smartResize(height: image.height, width: image.width, minPixels: LCOMedia.imageMinPixels, maxPixels: LCOMedia.imageMaxPixels)
    return LCOPatches.make([LCOResample.bicubic(image, width: w, height: h)])
}

private func pixels(video name: String) throws -> ([Float], LCOGrid) {
    let url = try fixture(name, "mp4"), track = try LCOVideoIO.track(url, cancellation: NativeCancellation())
    let count = try #require(LCOMedia.frameCount(total: track.frames, fps: track.fps))
    let (h, w) = try LCOMedia.smartResize(height: track.height, width: track.width, minPixels: LCOMedia.videoMinPixels, maxPixels: LCOMedia.videoMaxPixels(frames: count))
    let frames = try LCOVideoIO.frames(
        url, track: track, indices: LCOMedia.sampleIndices(total: track.frames, count: count), width: w, height: h, cancellation: NativeCancellation())
    return LCOPatches.make(frames)
}

/// Largest difference in 8-bit levels (normalized units × std × 255) between Arsenal's values and the maker's samples.
private func levels(_ values: [Float], _ entry: MediaReference.Entry, stride: Int) -> (max: Float, mean: Float) {
    #expect(values.count == entry.count)
    var worst: Float = 0, total: Float = 0
    for (k, expected) in entry.samples.enumerated() {
        let i = k * stride, channel = (i % 1176) / 392
        let diff = abs(values[i] - expected) * LCOPatches.std[channel] * 255
        worst = max(worst, diff); total += diff
    }
    return (worst, total / Float(max(1, entry.samples.count)))
}

@Test func lcoImageResizeMatchesTheMakersProcessor() throws {
    let reference = try JSONDecoder().decode(MediaReference.self, from: Data(contentsOf: try fixture("lco-media-reference", "json")))
    let (values, grid) = try pixels(image: "lco-resize-1000x750")
    #expect([grid.t, grid.h, grid.w] == reference.resize.grid)
    let (worst, mean) = levels(values, reference.resize, stride: reference.stride)
    // PIL's bicubic in fixed point against Arsenal's in Double: at most one 8-bit level on any sample.
    #expect(worst <= 1.01, "max \(worst) levels, mean \(mean)")
}

/// Transparent pixels become white, as qwen_omni_utils.to_rgb pastes RGBA onto a white background.
@Test func lcoTransparentImageCompositesOntoWhite() throws {
    let reference = try JSONDecoder().decode(MediaReference.self, from: Data(contentsOf: try fixture("lco-media-reference", "json")))
    let (values, grid) = try pixels(image: "lco-alpha-336x252")
    #expect([grid.t, grid.h, grid.w] == reference.alpha.grid)
    let (worst, _) = levels(values, reference.alpha, stride: reference.stride)
    #expect(worst <= 1.01, "max \(worst) levels")
}

/// Untagged H.264 decodes with BT.601 limited range, as FFmpeg does; the drawn colours come back.
@Test func lcoVideoColoursMatchFFmpeg() throws {
    let reference = try JSONDecoder().decode(MediaReference.self, from: Data(contentsOf: try fixture("lco-media-reference", "json")))
    for (name, entry) in [("lco-video-392x308", reference.video), ("lco-video-640x360", reference.videoResize)] {
        let (values, grid) = try pixels(video: name)
        #expect([grid.t, grid.h, grid.w] == entry.grid, "\(name)")
        // FFmpeg's integer Y'CbCr tables against Arsenal's exact matrix: at most two 8-bit levels, about one on average.
        let (worst, mean) = levels(values, entry, stride: reference.stride)
        #expect(worst <= 2.01 && mean < 1.1, "\(name): max \(worst) levels, mean \(mean)")
    }
    #expect(LCOYCbCr.coefficients(nil) == (0.299, 0.114))
    #expect(LCOYCbCr.coefficients(String(kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2)) == (0.2126, 0.0722))
}

@Test func lcoRejectsControlTokensInText() {
    let specials = ["<|AUDIO|>", "<|IMAGE|>", "<|im_end|>", "<|im_start|>"]
    #expect(LCOMedia.controlToken(in: "a cat <|IMAGE|> here", specials: specials) == "<|IMAGE|>")
    #expect(LCOMedia.controlToken(in: "text<|im_end|>", specials: specials) == "<|im_end|>")
    #expect(LCOMedia.controlToken(in: "plain text with <b>tags</b> and |pipes|", specials: specials) == nil)
}

/// librosa mono=True is the mean of the channels.
@Test func lcoStereoAudioIsTheMeanOfItsChannels() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent("lco-stereo-\(UUID().uuidString).wav")
    defer { try? FileManager.default.removeItem(at: url) }
    let format = try #require(AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: 16000, channels: 2, interleaved: false))
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: 1600))
    buffer.frameLength = 1600
    for i in 0..<1600 {
        buffer.floatChannelData![0][i] = 0.5 * sin(Float(i) * 0.05); buffer.floatChannelData![1][i] = 0.1 * cos(Float(i) * 0.11)
    }
    do {
        let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    }
    let mono = try LCOAudioIO.mono(url, maximumSamples: 16000)
    #expect(mono.count == 1600)
    let worst = (0..<1600).map { abs(mono[$0] - (buffer.floatChannelData![0][$0] + buffer.floatChannelData![1][$0]) / 2) }.max() ?? 1
    #expect(worst < 1e-5, "max \(worst)")
}

/// A US-letter page renders at twice its point size, 1,224×1,584, and needs 2,508 tokens, so 13 fit in 32,768.
@Test func lcoPDFPagesAreCountedBeforeRendering() throws {
    let (w, h) = LCOAdapter.pageSize(CGRect(x: 0, y: 0, width: 612, height: 792))
    #expect((w, h) == (1224, 1584))
    let (rh, rw) = try LCOMedia.smartResize(height: h, width: w, minPixels: LCOMedia.imageMinPixels, maxPixels: LCOMedia.imageMaxPixels)
    let tokens = rh * rw / (28 * 28)
    #expect(tokens == 2508 && tokens * 13 <= 32768 && tokens * 14 > 32768, "\(tokens) tokens per page")
}
