// SPDX-License-Identifier: MIT
// The media list as the worker reads it (Requests.media), the decode-time grid check (ElephantEngine.decodeMedia) and
// the operation schema's frame shape. Weight-free.
@testable import ElephantEngine
import Foundation
import Testing

@Suite struct MediaRequestTests {
    func parse(_ json: String) throws -> [MediaItem] { try Requests.media(try PyJSONParser.parse(Data(json.utf8)), maxItems: 32) }
    func code(_ json: String) -> String? {
        do { _ = try parse(json); return nil } catch let e as Requests.MediaError { return e.code } catch { return "other" }
    }

    @Test func framesArePathsOrPathObjectsWithDefaultFPS() throws {
        let m = try fixtureMedia().path
        let items = try parse(#"[{"type":"video","frames":["\#(m)/ball-0.png",{"path":"\#(m)/ball-1.png"}]}]"#)
        guard case .video(let v)? = items.first else { Issue.record("not a video"); return }
        #expect(v.frames.count == 2 && v.fps == 1.0 && v.width == 320 && v.height == 240)
    }

    @Test func countsComeFirstAndShapeErrorsAreInvalidRequest() throws {
        let m = try fixtureMedia().path
        // Counts are checked before any file: a one-frame list of missing files is a request error, not a file error.
        #expect(code(#"[{"type":"video","frames":["/no/such.png"]}]"#) == "invalid_request")
        let many = "[" + Array(repeating: "\"/no/such.png\"", count: 769).joined(separator: ",") + "]"
        #expect(code(#"[{"type":"video","frames":"# + many + "}]") == "invalid_request")
        #expect(code(#"[{"type":"video","frames":["\#(m)/ball-0.png","\#(m)/light-0.png"]}]"#) == "invalid_request") // sizes differ
        #expect(code(#"[{"type":"video","frames":["\#(m)/ball-0.png","\#(m)/ball-1.png"],"fps":0}]"#) == "invalid_request")
        #expect(code(#"[{"type":"audio","path":"\#(m)/ball-0.png"}]"#) == "invalid_request")
        // File problems are image errors.
        #expect(code(#"[{"type":"video","frames":["\#(m)/ball-0.png","/no/such.png"]}]"#) == "invalid_image")
        #expect(code(#"[{"type":"image","path":"relative.png"}]"#) == "invalid_image")
        #expect(code("[" + Array(repeating: #"{"type":"image","path":"\#(m)/tiny-50x40.png"}"#, count: 33).joined(separator: ",") + "]") == "invalid_request")
    }

    @Test func decodedPixelsMustMatchTheCountedGrid() throws {
        let m = try fixtureMedia()
        let images = try makeElephantProcessor(), videos = try makeVideoProcessor()
        let picture = try ImageInput.probe(m.appendingPathComponent("tiny-50x40.png"))
        let right = try ImageInput.tokens(height: picture.height, width: picture.width, config: images).grid
        _ = try ElephantEngine.decodeMedia([.image(picture)], [.image(right)], images: images, videos: videos)
        #expect(throws: PromptError.self) {
            try ElephantEngine.decodeMedia([.image(picture)], [.image(THW(1, right.h + 2, right.w))], images: images, videos: videos)
        }
        let clip = try VideoInput.probe((0..<4).map { m.appendingPathComponent("ball-\($0).png") }, fps: 2)
        let grid = try VideoInput.grid(clip, config: videos)
        #expect(throws: PromptError.self) {
            try ElephantEngine.decodeMedia([.video(clip)], [.video(THW(grid.t, grid.h, grid.w + 2), [[1], [2]])], images: images, videos: videos)
        }
        #expect(throws: PromptError.self) { try ElephantEngine.decodeMedia([.video(clip)], [.image(right)], images: images, videos: videos) }
    }

    @Test func schemaTakesFramesAsPathObjects() throws {
        // models/elephant-1/ops/decide.input.json, four folders above this test file.
        let ops = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath()
            .deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("ops/decide.input.json")
        let schema = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: ops)) as? [String: Any])
        let media = try #require((schema["properties"] as? [String: Any])?["media"] as? [String: Any])
        let kinds = try #require((media["items"] as? [String: Any])?["anyOf"] as? [[String: Any]])
        #expect(kinds.count == 2)
        let video = try #require(kinds.first { (($0["properties"] as? [String: Any])?["type"] as? [String: Any])?["const"] as? String == "video" })
        let frames = try #require((video["properties"] as? [String: Any])?["frames"] as? [String: Any])
        let item = try #require(frames["items"] as? [String: Any])
        #expect(item["type"] as? String == "object" && item["required"] as? [String] == ["path"] && item["x-arsenal-file"] != nil)
        #expect(item["anyOf"] == nil && frames["minItems"] as? Int == 2 && frames["maxItems"] as? Int == VideoInput.maxFrames)
    }
}
