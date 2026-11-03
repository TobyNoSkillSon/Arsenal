// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

struct CapabilityFoundationTests {
    @Test func toolNamesMapOneToOneToOperations() {
        // Over MCP every capability is one arsenal_run operation; HTTP keeps one route per capability.
        #expect(MCPTools.all.contains { $0.name == MCPNames.tool("run") })
        let routes = DialectRegistry.standard.routes
        for capability in CapabilityContract.all { #expect(routes.contains { $0.path == "/arsenal/v1/" + capability.route && $0.operation == capability.operation }) }
    }
    @Test func spatialAndStructuredValidation() async throws {
        let router = APIRouter(service: StubModelsService(), support: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        let schema: JSONObject = ["type": "object", "required": ["name"], "properties": ["name": ["type": "string"]], "additionalProperties": false]
        let params: JSONObject = ["response_format": ["type": "json_schema", "json_schema": ["name": "fixture", "schema": .object(schema)]]]
        try await router.validateStructuredChat(["text": "{\"name\":\"a\"}"], params: params)
        do { try await router.validateStructuredChat(["text": "{\"wrong\":1}"], params: params); Issue.record("Schema-invalid model output was accepted") } catch let failure
            as HTTPFailure
        { #expect(failure.status == 502) }
        let chat = try await router.chatParams([
            "messages": [["role": "user", "content": [["type": "image_url", "image_url": ["url": "data:image/png;base64,YQ=="]]]]],
            "response_format": params["response_format"]!
        ])
        #expect(chat["response_format"] == params["response_format"])
    }
    @Test func captionsRequireRealTimestamps() throws {
        let segments: [JSONValue] = [["start": 1.234, "end": 2.5, "text": "Hello"]]
        #expect(try TranscriptCaptions.render(segments, vtt: false).contains("00:00:01,234 --> 00:00:02,500"))
        #expect(try TranscriptCaptions.render(segments, vtt: true).hasPrefix("WEBVTT"))
        #expect(throws: HTTPFailure.self) { try TranscriptCaptions.render([["start": 3, "end": 2, "text": "bad"]], vtt: false) }
    }
    @Test func selectorsAndInlineMediaFailBeforeWork() async throws {
        let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let router = APIRouter(service: StubModelsService(), support: directory)
        do { _ = try await router.capabilityModel(["model": 42], operation: "embed"); Issue.record("Invalid model selector was accepted") } catch let failure as HTTPFailure {
            #expect(failure.status == 422)
        }
        try FileManager.default.createSymbolicLink(at: directory.appendingPathComponent("inbox"), withDestinationURL: directory)
        do { _ = try await router.inlineMedia("data:image/png;base64,YQ==", category: "image"); Issue.record("Symlink inbox was accepted") } catch let failure as HTTPFailure {
            #expect(failure.status == 422)
        }
        #expect(try FileManager.default.contentsOfDirectory(atPath: directory.path) == ["inbox"])
    }
    @Test func audioProfilesRetainFileContractsAndRejectDiscardedStems() async throws {
        let router = APIRouter(service: StubModelsService(), support: FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString))
        #expect(try await router.singleAudioFilePath(["files": [["path": "/job/song.wav", "mime": "audio/wav"]]]) == "/job/song.wav")
        do { _ = try await router.singleAudioFilePath(["files": [["path": "a.wav"], ["path": "b.wav"]]]); Issue.record("Extra audio files were discarded") } catch let failure
            as HTTPFailure
        { #expect(failure.status == 502) }
    }
    @Test func terminalJobStatusIsStable() {
        for status in ["succeeded", "failed", "canceled"] {
            var job = OperationJob(id: UUID().uuidString, model: "m", operation: "op", createdAt: Date()); job.status = status
            #expect(job.terminal)
        }
    }
}
