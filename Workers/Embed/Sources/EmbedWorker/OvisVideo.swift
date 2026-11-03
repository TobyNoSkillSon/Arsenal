// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import IntegrationKit
import ArsenalWire
import MLXLMCommon

struct OvisVideoConfiguration {
    let json: [String: Any]
    let minPixels: Int
    let maxPixels: Int
    let patchSize: Int
    let mergeSize: Int
    let temporalSize: Int
    let minFrames: Int
    let maxFrames: Int
    let fps: Double

    init(_ json: [String: Any]) throws {
        guard let size = json["size"] as? [String: Int], let minimum = size["shortest_edge"], let maximum = size["longest_edge"],
            let patch = json["patch_size"] as? Int, let merge = json["merge_size"] as? Int, let temporal = json["temporal_patch_size"] as? Int,
            let minFrames = json["min_frames"] as? Int, let maxFrames = json["max_frames"] as? Int, let fps = json["fps"] as? Double,
            minimum > 0, maximum >= minimum, patch > 0, merge > 0, temporal > 0, minFrames > 0, maxFrames >= minFrames, fps > 0
        else { throw WorkerFailure(code: "invalid_config", message: "Ovis requires its pinned video processor settings.") }
        self.json = json; minPixels = minimum; maxPixels = maximum; patchSize = patch; mergeSize = merge; temporalSize = temporal
        self.minFrames = minFrames; self.maxFrames = maxFrames; self.fps = fps
    }

    // Transformers Qwen3VLVideoProcessor.smart_resize: limits cover all sampled
    // frames, including temporal padding when deciding whether to resize.
    func targetSize(frames: Int, height: Int, width: Int) throws -> (Int, Int) {
        let factor = patchSize * mergeSize
        guard frames > 0, height >= factor, width >= factor, Double(max(height, width)) / Double(min(height, width)) <= 200 else {
            throw WorkerFailure(code: "invalid_request", message: "Video dimensions are outside the Ovis processor limits.")
        }
        let divisor = Double(factor)
        var resizedHeight = Int((Double(height) / divisor).rounded(.toNearestOrEven)) * factor
        var resizedWidth = Int((Double(width) / divisor).rounded(.toNearestOrEven)) * factor
        let paddedFrames = ((frames + temporalSize - 1) / temporalSize) * temporalSize
        if paddedFrames * resizedHeight * resizedWidth > maxPixels {
            let beta = sqrt(Double(frames * height * width) / Double(maxPixels))
            resizedHeight = max(factor, Int(floor(Double(height) / beta / divisor)) * factor)
            resizedWidth = max(factor, Int(floor(Double(width) / beta / divisor)) * factor)
        } else if paddedFrames * resizedHeight * resizedWidth < minPixels {
            let beta = sqrt(Double(minPixels) / Double(frames * height * width))
            resizedHeight = Int(ceil(Double(height) * beta / divisor)) * factor
            resizedWidth = Int(ceil(Double(width) * beta / divisor)) * factor
        }
        return (resizedHeight, resizedWidth)
    }

    func sampleIndices(total: Int, sourceFPS: Double) -> [Int] {
        guard total > 0 else { return [] }
        let count = min(max(Int(Double(total) / sourceFPS * fps), minFrames), maxFrames, total)
        if count == 1 { return [0] }
        return (0..<count).map { Int((Double($0) * Double(total - 1) / Double(count - 1)).rounded(.toNearestOrEven)) }
    }
}

enum OvisVideoFraming {
    static func timestamps(_ times: [Double], temporalSize: Int) -> [Double] {
        stride(from: 0, to: times.count, by: temporalSize).map { (times[$0] + times[min($0 + temporalSize - 1, times.count - 1)]) / 2 }
    }

    static func placeholder(times: [Double], grid: THW, temporalSize: Int, mergeSize: Int) -> String {
        let stamps = timestamps(times, temporalSize: temporalSize)
        precondition(stamps.count == grid.t)
        let tokens = String(repeating: "<|video_pad|>", count: grid.h * grid.w / (mergeSize * mergeSize))
        return stamps.map { String(format: "<%.1f seconds>", locale: Locale(identifier: "en_US_POSIX"), $0) + "<|vision_start|>" + tokens + "<|vision_end|>" }.joined()
    }
}

struct OvisPromptExpansion {
    private(set) var text: String
    private var offset = 0
    init(_ text: String) { self.text = text }
    mutating func replace(_ original: String, with replacement: String) throws {
        let cursor = text.index(text.startIndex, offsetBy: offset)
        guard let range = text.range(of: original, range: cursor..<text.endIndex) else {
            throw WorkerFailure(code: "invalid_request", message: "Native template omitted a visual placeholder.")
        }
        let start = text.distance(from: text.startIndex, to: range.lowerBound)
        text.replaceSubrange(range, with: replacement)
        offset = start + replacement.count
    }
}

enum OvisVideoMetadata {
    static func read(url: URL, cancellation: NativeCancellation) async throws -> (Int, Double) {
        let asset = AVURLAsset(url: url)
        guard let track = try await asset.loadTracks(withMediaType: .video).first else {
            throw WorkerFailure(code: "invalid_request", message: "Video has no track.")
        }
        let fps = Double(try await track.load(.nominalFrameRate))
        let reader = try AVAssetReader(asset: asset)
        let output = AVAssetReaderTrackOutput(track: track, outputSettings: nil)
        output.alwaysCopiesSampleData = false
        reader.add(output)
        guard reader.startReading() else { throw WorkerFailure(code: "invalid_request", message: "Cannot inspect video frames.") }
        defer { reader.cancelReading() }
        var count = 0
        while let sample = output.copyNextSampleBuffer() {
            try cancellation.check()
            // Compressed readers also emit zero-sample edit/format markers. They
            // are not frames; counting buffers gives wrong uniform indices.
            count += CMSampleBufferGetNumSamples(sample)
        }
        guard reader.status != .failed else { throw reader.error ?? WorkerFailure(code: "invalid_request", message: "Cannot inspect video frames.") }
        return (count, fps > 0 ? fps : 24)
    }
}
