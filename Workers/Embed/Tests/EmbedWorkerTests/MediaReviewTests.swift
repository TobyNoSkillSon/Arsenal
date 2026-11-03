import AVFoundation
import CoreGraphics
import Foundation
import PDFKit
import Testing
@testable import EmbedWorker

/// A 400×200 pt page whose left 200×200 pt is red, cropped to its left 300 pt and turned a quarter clockwise by /Rotate.
/// Shown upright it is 200 pt wide and 300 pt tall: the red square fills the top two thirds, white the bottom third.
func rotatedCroppedPDF(in directory: URL) throws -> URL {
    let source = directory.appendingPathComponent("flat.pdf"), url = directory.appendingPathComponent("rotated.pdf")
    var media = CGRect(x: 0, y: 0, width: 400, height: 200)
    let context = try #require(CGContext(source as CFURL, mediaBox: &media, nil))
    context.beginPDFPage(nil)
    context.setFillColor(CGColor(red: 1, green: 0, blue: 0, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: 200, height: 200))
    context.endPDFPage(); context.closePDF()
    // PDFKit is not safe to drive from parallel tests; one fixture is written at a time, to a new file.
    try pdfKitLock.withLock {
        let document = try #require(PDFDocument(url: source)), page = try #require(document.page(at: 0))
        page.rotation = 90
        page.setBounds(CGRect(x: 0, y: 0, width: 300, height: 200), for: .cropBox)
        #expect(document.write(to: url))
    }
    return url
}
private let pdfKitLock = NSLock()

/// The rotated page at 2× fills its 400×600 canvas: red reaches the top corners and the left and right edges down to
/// two thirds, white fills the bottom third. A page drawn at 1× in the middle of the canvas leaves the corners white.
/// Red is loose: the page's device red reaches sRGB through colour management (255, 38, 0 here).
func expectRotatedPageFillsTheCanvas(width: Int, height: Int, rgb: (Int, Int) -> (UInt8, UInt8, UInt8)) {
    #expect(width == 400 && height == 600, "\(width)×\(height)")
    guard width == 400, height == 600 else { return }
    let red: [(Int, Int)] = [(2, 2), (397, 2), (2, 395), (397, 395), (200, 200)]
    let white: [(Int, Int)] = [(2, 405), (397, 405), (2, 597), (397, 597), (200, 500)]
    for (x, y) in red {
        let (r, g, b) = rgb(x, y)
        #expect(r > 200 && g < 90 && b < 90, "(\(x), \(y)) is \(r),\(g),\(b), not red")
    }
    for (x, y) in white {
        let (r, g, b) = rgb(x, y)
        #expect(r > 230 && g > 230 && b > 230, "(\(x), \(y)) is \(r),\(g),\(b), not white")
    }
}

/// Review fix (L8, 2 Nov 2026): LCO drew PDF pages from the media box without /Rotate, so rotated scans were embedded
/// sideways with their crop margins.
@Test func lcoRotatedPDFPagesRenderUprightAndFillTheCanvas() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("lco-pdf-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = try rotatedCroppedPDF(in: directory)
    let document = try #require(CGPDFDocument(url as CFURL)), page = try #require(document.page(at: 1))
    let (w, h) = LCOAdapter.pageSize(page)
    let image = try LCOAdapter.render(page, width: w, height: h)
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
        guard
            let context = CGContext(
                data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
        else { return false }
        context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        return true
    }
    #expect(drawn)
    expectRotatedPageFillsTheCanvas(width: w, height: h) { x, y in
        let i = 4 * (y * w + x)
        return (rgba[i], rgba[i + 1], rgba[i + 2])
    }
}

/// Review fix (M4, 2 Nov 2026): Qwen3-Embedding pools the last id, its <|endoftext|>; truncation cut it off and pooled a
/// content token.
@Test func qwenEmbeddingTruncationKeepsThePooledEndToken() {
    let ids = Array(0..<40) + [151_643]
    let qwen = TextRetrievalAdapter.truncated(ids, limit: 16, keepingLast: true)
    #expect(qwen.count == 16 && qwen.last == 151_643 && Array(qwen.prefix(15)) == Array(0..<15))
    let other = TextRetrievalAdapter.truncated(ids, limit: 16, keepingLast: false)
    #expect(other == Array(0..<16))
    #expect(TextRetrievalAdapter.truncated(ids, limit: 41, keepingLast: true) == ids)
}

/// Review fix (L6, 2 Nov 2026): the frame list was built before the 32-frame cap, so a near-zero rate asked for billions
/// of positions and a denormal one trapped converting an infinite count.
@Test func eg2FrameSamplingCapsBeforeBuildingTheList() {
    for fps in [1e-10, 5e-324, Double.leastNormalMagnitude] {
        let indices = EG2Vision.sampleFrames(total: 100, fps: fps)
        #expect(indices.count == 32 && indices.allSatisfy { (0..<100).contains($0) }, "fps \(fps): \(indices.count) frames")
    }
    #expect(EG2Vision.sampleFrames(total: 100, fps: .infinity) == EG2Vision.sampleFrames(total: 100, fps: 0))
    #expect(EG2Vision.sampleFrames(total: 10, fps: 1e300) == [0])
    // Ordinary rates are unchanged: 300 frames at 30 fps is ten seconds, one frame a second.
    #expect(EG2Vision.sampleFrames(total: 300, fps: 30) == stride(from: 0, to: 300, by: 30).map { $0 })
}

/// Writes `frames` of 5.1 audio whose channel `c` carries `value(c, i)`.
private func sixChannelWAV(_ url: URL, rate: Double, frames: Int, value: (Int, Int) -> Float) throws {
    let layout = try #require(AVAudioChannelLayout(layoutTag: kAudioChannelLayoutTag_MPEG_5_1_A))
    let format = AVAudioFormat(standardFormatWithSampleRate: rate, channelLayout: layout)
    let file = try AVAudioFile(forWriting: url, settings: format.settings, commonFormat: .pcmFormatFloat32, interleaved: false)
    let buffer = try #require(AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(frames)))
    buffer.frameLength = AVAudioFrameCount(frames)
    for c in 0..<6 { for i in 0..<frames { buffer.floatChannelData![c][i] = value(c, i) } }
    try file.write(from: buffer)
}

/// Review fix (M5, 2 Nov 2026): LCO refused 3–8 channel audio (AVAudioFormat has no plain 6-channel PCM target); it now
/// takes the shared surround path, the mean of the channels as librosa's mono load gives.
@Test func lcoSurroundAudioIsTheMeanOfItsChannels() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("lco-surround-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    for rate in [16_000.0, 48_000.0] {
        let url = directory.appendingPathComponent("six-\(Int(rate)).wav")
        try sixChannelWAV(url, rate: rate, frames: Int(rate / 2)) { c, _ in Float(c + 1) / 20 }
        let mono = try LCOAudioIO.mono(url, maximumSamples: LCOMel.maxSamples)
        #expect(abs(mono.count - 8_000) <= 2, "\(rate) Hz gave \(mono.count) samples")
        let middle = mono[1000..<7000]
        #expect(middle.allSatisfy { abs($0 - 0.175) < 2e-3 }, "\(rate) Hz mean \(middle.first ?? .nan)")
    }
}

/// Review fix (L7, 2 Nov 2026): surround audio is read and resampled in 64k-frame blocks instead of a second full-length
/// multichannel buffer. A varying signal across several blocks still comes back as the exact per-sample mean, and a
/// resampled one keeps its length and level.
@Test func surroundAudioReadsInBlocksWithoutSeams() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent("surround-blocks-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let frames = 3 * SurroundAudio.block + 1234
    func value(_ c: Int, _ i: Int) -> Float { 0.1 * Float(c + 1) * sin(Float(i % 4096) * 0.003 * Float(c + 1)) }
    let native = directory.appendingPathComponent("native.wav")
    try sixChannelWAV(native, rate: 16_000, frames: frames, value: value)
    let mono = try SurroundAudio.mono(native, sampleRate: 16_000, maximumSamples: 16_000 * 600)
    #expect(mono.count == frames)
    let worst = (0..<min(frames, mono.count)).map { i in abs(mono[i] - (0..<6).map { value($0, i) }.reduce(0, +) / 6) }.max() ?? 1
    #expect(worst < 1e-6, "largest difference \(worst)")
    let high = directory.appendingPathComponent("high.wav")
    try sixChannelWAV(high, rate: 48_000, frames: frames) { c, _ in Float(c + 1) / 20 }
    let resampled = try SurroundAudio.mono(high, sampleRate: 16_000, maximumSamples: 16_000 * 600)
    #expect(abs(resampled.count - frames / 3) <= 2, "\(resampled.count) samples")
    #expect(resampled[1000..<(resampled.count - 1000)].allSatisfy { abs($0 - 0.175) < 2e-3 })
    #expect(throws: (any Error).self) { try SurroundAudio.mono(native, sampleRate: 16_000, maximumSamples: frames - 1) }
}
