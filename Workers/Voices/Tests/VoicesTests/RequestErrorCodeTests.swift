// SPDX-License-Identifier: MIT
import ArsenalWire
import Foundation
import Testing
@testable import VoicesWorker

/// A caller's mistake on any voices model is invalid_request (HTTP 422). The service has no mapping for invalid_input
/// or engine errors, so those came back as HTTP 500 (found on the VoxCPM2 real check, 2 Nov).
@Suite struct RequestErrorCodeTests {
    static func isInvalidRequest(_ error: Error) -> Bool { (error as? WorkerFailure)?.code == "invalid_request" }

    @Test func samplingControlsOutsideTheirRangeAreInvalidRequests() {
        for (model, request): (String, JSONObject) in [
            ("kokoro-82m", ["op": "speak", "text": "Hi", "temperature": 0.5]), ("fish-drama-3", ["op": "speak", "text": "Hi", "top_k": 0]),
            ("qwen3-tts-base", ["op": "speak", "text": "Hi", "max_tokens": 9000]), ("voxcpm2", ["op": "speak", "text": "Hi", "top_p": 0.5])
        ] {
            #expect(throws: (any Error).self, "\(model)") { try SpeechOperation.check(request, modelID: model) }
            #expect { try SpeechOperation.check(request, modelID: model) } throws: { Self.isInvalidRequest($0) }
        }
    }

    @Test func missingOrBlankFieldsAreInvalidRequests() {
        for request: JSONObject in [[:], ["text": " \n "], ["text": 3]] {
            #expect { try string(request, "text") } throws: { Self.isInvalidRequest($0) }
        }
    }

    @Test func unreadableOrOverlongReferenceClipsAreInvalidRequests() throws {
        let dir = FileManager.default.temporaryDirectory.appendingPathComponent("voices-ref-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: dir) }
        let text = dir.appendingPathComponent("not-audio.wav")
        try Data("this is not audio".utf8).write(to: text)
        #expect { try PCM.reference(text, rate: 16000) } throws: { Self.isInvalidRequest($0) }
        let long = dir.appendingPathComponent("long.wav"), bytes = 8000 * 2 * 31
        try (PCM.header(rate: 8000, bytes: UInt32(bytes)) + Data(count: bytes)).write(to: long)
        #expect { try PCM.reference(long, rate: 16000) } throws: { Self.isInvalidRequest($0) }
    }

    /// No request check in the voices worker uses the unmapped invalid_input code.
    @Test func noVoicesRequestCheckUsesAnUnmappedCode() throws {
        let sources = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("Sources/VoicesWorker")
        let files = try FileManager.default.contentsOfDirectory(at: sources, includingPropertiesForKeys: nil).filter { $0.pathExtension == "swift" }
        #expect(files.count >= 5)
        for file in files {
            let source = try String(contentsOf: file, encoding: .utf8)
            #expect(!source.contains("code: \"invalid_input\""), "\(file.lastPathComponent)")
        }
    }
}
