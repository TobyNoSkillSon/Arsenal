// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

@Test func v22CatalogRangeValidation() throws {
    let query = try catalogQuery(["dtype": "mxfp4,oQ4e", "min_size_bytes": "100", "max_size_bytes": "200", "min_ram_mb": "1000", "max_ram_mb": "2000"])
    #expect(query.dtypes == ["mxfp4", "oQ4e"] && query.minSizeBytes == 100 && query.maxRAMMB == 2000)
    for bad in [
        ["min_size_bytes": "-1"], ["max_ram_mb": "nan"], ["min_ram_mb": "inf"], ["min_size_bytes": "200", "max_size_bytes": "100"], ["min_ram_mb": "2000", "max_ram_mb": "1000"]
    ] {
        #expect(throws: HTTPFailure.self) { _ = try catalogQuery(bad) }
    }
}

@Test func v22ControlGetPlanAndBodylessRefreshAndOpenAPI() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v22-core/api-" + UUID().uuidString)
    var row = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["bf16", "future"], downloadBytes: ["bf16": 123])
    row.formats = [.init(id: "future", bitsPerWeight: 3, source: .runnerNative, runners: [], storage: "exl3")]
    // The table counts only models with weights on this Mac: a downloaded one beside the not-downloaded voice.
    let kept = ModelRow(id: "kept", name: "Kept", kind: "voices", states: ["bf16": .init(state: .ready)])
    let service = StubModelsService(rows: [row, kept], supportDirectory: root)
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func send(_ method: String, _ path: String, _ body: JSONObject? = nil) async throws -> (JSONObject, Int) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/" + path)!)
        request.httpMethod = method; request.timeoutInterval = 10
        if let body { request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try WireJSON.decode(data), (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
    let plan = try await send("GET", "models/voice/get-plan?dtype=bf16")
    #expect(plan.1 == 200 && plan.0["dtype"] == "bf16" && plan.0["totalDownloadBytes"] == 123 && plan.0["requiresDownload"] == true)
    #expect(try await send("GET", "models/voice/get-plan?dtype=not-offered").1 == 400)
    #expect(try await send("GET", "models/absent/get-plan").1 == 404)
    #expect(try await send("GET", "models/voice/get-plan?dtype=future").1 == 422)
    for path in ["get", "load"] {
        let unsupported = try await send("POST", path, ["id": "voice", "dtype": "future"])
        #expect(unsupported.1 == 422 && unsupported.0["error"] == "No runner in this Arsenal yet." && unsupported.0["code"] == "unsupported_format")
    }
    let refreshed = try await send("POST", "refresh")
    #expect(refreshed.1 == 200 && refreshed.0["ok"] == true && refreshed.0["counts"]?["models"] == 1)
    let document = try await APIRouter(service: service, support: root).openAPI()
    #expect(document["paths"]?["/arsenal/v1/refresh"]?["post"]?["requestBody"] == nil)
    #expect(document["paths"]?["/arsenal/v1/models/{id}/get-plan"]?["get"]?["parameters"]?.array?.count == 2)
    await service.shutdown()
}

@Test func v22UnsupportedFormatCodeSurvivesUniversalAndControlErrorEnvelopes() throws {
    let error = WorkerFailure(code: "unsupported_format", message: "No runner in this Arsenal yet.")
    #expect(workerHTTPStatus(error) == 422)
    let failure = serviceHTTPFailure(error)
    #expect(failure.status == 422 && failure.code == "unsupported_format")
    let control = try WireJSON.decode(errorBody(failure, path: "/arsenal/v1/get", requestID: "fixture"))
    #expect(control["code"] == "unsupported_format" && control["error"] == .string(error.message))
    let provider = try WireJSON.decode(errorBody(failure, path: "/v1/audio/speech", requestID: "fixture"))
    #expect(provider["error"]?["code"] == "unsupported_format")
}

@Test func v22GetConsentRejectsAdditionalBytesAndReturnsRevisedPlan() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v22-fix-l/api-consent-" + UUID().uuidString)
    let row = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["bf16"], downloadBytes: ["bf16": 3_000_000_000])
    let service = StubModelsService(rows: [row], supportDirectory: root)
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    let plan = try await service.getPlan(id: "voice", dtype: "bf16")
    func post(_ approved: GetPlan?) async throws -> (JSONObject, Int) {
        var body: JSONObject = ["id": "voice", "dtype": "bf16"]
        if let approved { body["approved_plan"] = try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode(approved)) }
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/get")!)
        request.httpMethod = "POST"; request.timeoutInterval = 10; request.httpBody = try WireJSON.encode(body)
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try WireJSON.decode(data), (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
    let noConsent = try await post(nil)
    #expect(noConsent.1 == 409 && noConsent.0["error"]?["revised_plan"]?["totalDownloadBytes"] == 3_000_000_000)
    var stale = plan; stale.items[0].downloadBytes = 0
    #expect(try await post(stale).1 == 409)
    #expect(try await post(plan).1 == 202)
    await service.shutdown()
}
