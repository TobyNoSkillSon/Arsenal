// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire
import AVFoundation

/// The `interpolate_video` contract shared by every interpolation model: two frames with optional `t` positions, or a
/// video with `factor` / `target_fps`. A model supplies `pair`, which prepares one frame pair (shared work such as
/// optical flow happens once there) and returns the frame at any 0 < t < 1; t = 0 and t = 1 return the source frames
/// without running the model, and a pair is prepared only when an in-between frame is needed.
enum InterpolationDriver {
    typealias Pair = (_ first: MLXArray, _ second: MLXArray) throws -> (Double) throws -> MLXArray

    /// Positions within 1e-4 of an endpoint are that endpoint: NTSC rates (30000/1001) put output times a few
    /// microseconds off the source frames, which would otherwise run the model for a frame identical to a source frame.
    static let endpointTolerance = 1e-4
    static func snap(_ t: Double) -> Double { t < endpointTolerance ? 0 : t > 1 - endpointTolerance ? 1 : t }

    static func run(_ input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void, pair: Pair) throws -> JSONObject {
        let root = try MediaBridge.output(input)
        if let frames = input["frames"]?.array {
            guard input["video"] == nil, input["factor"] == nil, input["target_fps"] == nil, frames.count == 2 else {
                throw MediaBridge.error("Supply exactly two frames and optional t positions, or a video with factor/target_fps.")
            }
            let a = try MediaBridge.tensor(NativeMedia.image(MediaBridge.path(frames[0])))
            let b = try MediaBridge.tensor(NativeMedia.image(MediaBridge.path(frames[1])))
            guard a.shape == b.shape else { throw MediaBridge.error("Frames must have equal sizes and t must be between zero and one.") }
            let times = input["t"]?.array ?? [.number(0.5)]
            guard !times.isEmpty, times.count <= 16, times.allSatisfy({ $0.number.map { $0.isFinite && (0...1).contains($0) } == true }) else {
                throw MediaBridge.error("t requires one to sixteen positions from zero to one.")
            }
            var prepared: ((Double) throws -> MLXArray)?
            var files: [JSONValue] = []
            for (index, time) in times.enumerated() {
                try cancellation.check()
                let t = snap(time.number!)
                let image: MLXArray
                if t == 0 {
                    image = a
                } else if t == 1 {
                    image = b
                } else {
                    if prepared == nil { prepared = try pair(a, b) }
                    image = try prepared!(t)
                }
                try cancellation.check()
                let url = root.appendingPathComponent("frame-\(index).png"); try MediaBridge.png(image, at: url)
                files.append(.object(try NativeMedia.file(url, root: root, mime: "image/png", id: "frame-\(index)")))
                try MediaBridge.progress(index + 1, total: times.count, label: "Interpolating", emit: emit)
            }
            return ["files": .array(files)]
        }
        guard input["t"] == nil, !(input["factor"] != nil && input["target_fps"] != nil) else {
            throw MediaBridge.error("Video accepts factor or target_fps, not t or both rate controls.")
        }
        let source = try MediaBridge.path(input["video"])
        let asset = AVURLAsset(url: source)
        let sourceFPS = try MediaBridge.wait { try await asset.loadTracks(withMediaType: .video).first?.load(.nominalFrameRate) ?? 0 }
        let factor = input["factor"]?.number ?? 2
        let fps = input["target_fps"]?.number ?? Double(sourceFPS) * factor
        guard fps.isFinite, (1...240).contains(fps), input["factor"] == nil || (factor.rounded() == factor && (2...16).contains(factor)) else {
            throw MediaBridge.error("Invalid video rate; target fps must be 1...240 or factor 2...16.")
        }
        let reader = try MediaBridge.wait { try await NativeVideoReader(url: source) }
        defer { reader.cancel() }
        guard let first = try reader.next(), let second = try reader.next() else { throw MediaBridge.error("Interpolation needs at least two video frames.") }
        var a = try MediaBridge.tensor(first), b = try MediaBridge.tensor(second)
        guard a.shape == b.shape else { throw MediaBridge.error("Video dimensions must stay fixed and timestamps must increase.") }
        var prepared: ((Double) throws -> MLXArray)?
        let origin = CMSampleBufferGetPresentationTimeStamp(first).seconds
        var ta = 0.0, tb = CMSampleBufferGetPresentationTimeStamp(second).seconds - origin
        guard tb > ta else { throw MediaBridge.error("Video timestamps must increase.") }
        let url = root.appendingPathComponent("interpolated.mp4"), expected = MediaBridge.frameEstimate(source, fps: fps)
        let writer = try NativeVideoWriter(url: url, width: a.dim(2), height: a.dim(1), transform: reader.transform)
        var complete = false
        defer { if !complete { writer.cancel(); try? FileManager.default.removeItem(at: url) } }
        var index = 0, lastInterval = tb
        while true {
            while Double(index) / fps < tb - 0.0000001 {
                try cancellation.check()
                let t = snap(Swift.max(0, Swift.min(1, (Double(index) / fps - ta) / (tb - ta))))
                let frame: MLXArray
                if t == 0 {
                    frame = a
                } else if t == 1 {
                    frame = b
                } else {
                    if prepared == nil { prepared = try pair(a, b) }
                    frame = try prepared!(t)
                }
                try cancellation.check()
                try writer.append(MediaBridge.buffer(frame), time: CMTime(seconds: Double(index) / fps, preferredTimescale: 60000), cancellation: cancellation)
                index += 1; try MediaBridge.progress(index, total: expected, label: "Interpolating", emit: emit)
            }
            guard let next = try reader.next() else { break }
            a = b; ta = tb; b = try MediaBridge.tensor(next); tb = CMSampleBufferGetPresentationTimeStamp(next).seconds - origin
            prepared = nil
            guard tb > ta, a.shape == b.shape else { throw MediaBridge.error("Video dimensions must stay fixed and timestamps must increase.") }
            lastInterval = tb - ta
        }
        while Double(index) / fps < tb + lastInterval - 0.0000001 {
            try cancellation.check(); try writer.append(MediaBridge.buffer(b), time: CMTime(seconds: Double(index) / fps, preferredTimescale: 60000), cancellation: cancellation)
            index += 1
        }
        try MediaBridge.wait { try await writer.finish() }; complete = true
        do {
            try MediaBridge.keepAudio(from: source, video: url, origin: origin, cancellation: cancellation)
        } catch {
            try? FileManager.default.removeItem(at: url); throw error
        }
        return ["files": [.object(try NativeMedia.file(url, root: root, mime: "video/mp4", id: "video"))]]
    }
}
