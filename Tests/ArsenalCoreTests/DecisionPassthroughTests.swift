// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func v2DecisionRawPassthroughPreservesBytesAndLeavesValidationToWorker() async throws {
    let ops = ["decide", "judge"].map { op -> TOMLValue in
        .table(["name": .string(op), "title": .string(op), "input": .string("ops/\(op).input.json"), "output": .string("json"), "stream": .bool(false)])
    }
    let schema = Data(
        #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["state","questions"],"properties":{"state":{"type":"string"},"questions":{"type":"object"},"marker":{"type":"number"}}}"#
            .utf8)
    let extra: [String: TOMLValue] = [
        "schema": .integer(2), "selection": .table(["default": .string("native")]), "op": .array(ops),
        "setting": .array([
            .table([
                "id": .string("marker"), "title": .string("Marker"), "type": .string("number"), "default": .integer(7), "applies": .string("request"),
                "maps_to": .string("request.marker")
            ])
        ])
    ]
    let f = try await makeLocalCatalogFixture(kinds: ["decisions"], extraTable: extra, extraFiles: ["ops/decide.input.json": schema, "ops/judge.input.json": schema])
    defer { f.stop() }
    let id = "fixture-decisions"
    await #expect(throws: WorkerFailure.self) { _ = try await f.service.run(id: id, op: "decide", params: [:]) }
    try await f.service.get(id: id, precision: "bf16")
    let raw = "{\"questions\":{\"é\":{\"type\":\"noul\"},\"e\u{301}\":{\"type\":\"noul\"}},\"state\":\"first\"}"
    for op in ["decide", "judge"] {
        let result = try await f.service.run(id: id, op: op, params: ["body": .string(raw)])
        let forwarded = try #require(result["_test_body"]?.string)
        #expect(Data(forwarded.utf8) == Data(raw.utf8))
        #expect(result["_test_params"]?.object?.keys.sorted() == ["body"])
    }
    // The exception is precisely the dialect envelope, not arbitrary missing-schema requests.
    await #expect(throws: WorkerFailure.self) { _ = try await f.service.run(id: id, op: "decide", params: ["body": .string(raw), "marker": 1]) }
    await f.service.shutdown()
}

/// elephant-1 shape: a schema-3 decisions model whose raw HTTP body carries image paths. The bytes still pass through
/// untouched, but a path outside Arsenal's inbox and jobs folders is refused, as MCP and every other schema-3 input are.
@Test func v3DecisionRawBodyKeepsFilePathsInsideArsenalFolders() async throws {
    let ops = ["decide", "judge"].map { op -> TOMLValue in
        .table([
            "name": .string(op), "title": .string(op), "input": .string("ops/\(op).input.json"), "output": .string("json"),
            "output_schema": .string("ops/\(op).output.json"), "stream": .bool(false)
        ])
    }
    let schema = Data(
        #"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["state","questions"],"properties":{"state":{},"questions":{"type":"object"},"media":{"type":"array","items":{"anyOf":[{"type":"object","required":["type","path"],"additionalProperties":false,"properties":{"type":{"const":"image"},"path":{"type":"string"}},"x-arsenal-file":{"accept":["image/png","image/jpeg"]}}]}}}}"#
            .utf8)
    let f = try await makeLocalCatalogFixture(
        kinds: ["decisions"], extraTable: ["schema": .integer(3), "selection": .table(["default": .string("native")]), "op": .array(ops)],
        extraFiles: [
            "ops/decide.input.json": schema, "ops/judge.input.json": schema,
            "ops/decide.output.json": Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#.utf8),
            "ops/judge.output.json": Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#.utf8)
        ])
    defer { f.stop() }
    let id = "fixture-decisions"
    try await f.service.get(id: id, precision: "bf16")
    let outside = URL(fileURLWithPath: "/private/tmp/arsenal-outside-\(UUID().uuidString).png")
    try Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVQIHWP4z8DwHwAFgAI/ScLttAAAAABJRU5ErkJggg==")!.write(to: outside)
    defer { try? FileManager.default.removeItem(at: outside) }
    let inbox = f.paths.support.appendingPathComponent("inbox/pixel.png")
    try FileManager.default.createDirectory(at: inbox.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.copyItem(at: outside, to: inbox)
    func body(_ media: String) -> String { #"{"state":"<image:1>","questions":{"q":{"type":"noul"}},"media":"# + media + "}" }
    for op in ["decide", "judge"] {
        let inside = body(#"[{"type":"image","path":"\#(inbox.path)"}]"#)
        let result = try await f.service.run(id: id, op: op, params: ["body": .string(inside)])
        #expect(result["_test_body"]?.string == inside)
        // The worker reads a repeated key as Python does (the last one wins), so the check must too.
        for media in [
            #"[{"type":"image","path":"\#(outside.path)"}]"#, #"[{"type":"image","path":"\#(inbox.path)","path":"\#(outside.path)"}]"#,
            // An item that matches no schema branch would skip the file walk; the schema check refuses it first.
            #"[{"type":"image","path":"\#(outside.path)","z":0}]"#
        ] {
            do {
                _ = try await f.service.runRawResult(id: id, op: op, params: ["body": .string(body(media))])
                Issue.record("\(op) accepted a file outside inbox/jobs: \(media)")
            } catch let failure as WorkerFailure { #expect(failure.code == "invalid_request") }
        }
        // A raw body nested beside ordinary members is refused rather than forwarded unchecked.
        do {
            _ = try await f.service.run(
                id: id, op: op, params: ["state": "x", "questions": ["q": ["type": "noul"]], "body": .string(body(#"[{"type":"image","path":"\#(outside.path)"}]"#))])
            Issue.record("\(op) forwarded a nested raw body")
        } catch let failure as WorkerFailure { #expect(failure.code == "invalid_request") }
        let shadowed = #"{"media":[{"type":"image","path":"\#(outside.path)"}],"state":"<image:1>","questions":{"q":{"type":"noul"}},"media":[]}"#
        _ = try await f.service.run(id: id, op: op, params: ["body": .string(shadowed)])
    }
    await f.service.shutdown()
}
