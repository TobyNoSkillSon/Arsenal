// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

@Test func v21ControlRunnerRoundtripAndValidation() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-v21-service/api-" + UUID().uuidString)
    let row = ModelRow(id: "voice", name: "Voice", kind: "voices", states: ["bf16": .init(state: .ready)])
    let service = StubModelsService(
        rows: [row], metadata: ["voice": ["runner": [["name": "native", "language": "swift", "chips": ["*"]], ["name": "special", "language": "rust", "chips": ["*"]]]]],
        supportDirectory: root)
    try await service.load(id: "voice")
    let server = ArsenalAPIServer(service: service, supportDirectory: root, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func send(_ method: String, _ path: String, _ body: JSONObject? = nil) async throws -> (JSONObject, Int) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/arsenal/v1/" + path)!)
        request.httpMethod = method; request.timeoutInterval = 10
        if let body { request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try WireJSON.decode(data), (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
    #expect(try await send("GET", "status").0["version"]?.string == (Bundle.main.object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "development"))
    let path = "models/voice/runner"
    #expect(try await send("POST", path, ["runner": "special"]).1 == 200)
    let pending = try await send("GET", path)
    #expect(pending.0["runner"] == "special" && pending.0["resident"] == "native" && pending.0["needsReload"] == true)
    #expect(pending.0["runners"]?.array?.count == 2)
    #expect(try await send("POST", path, ["runner": .null]).1 == 200)
    #expect(try await send("GET", path).0["needsReload"] == false)
    for body: JSONObject in [[:], ["runner": 1], ["runner": false], ["runner": "not-declared"]] { #expect(try await send("POST", path, body).1 == 422) }
    let document = try await APIRouter(service: service, support: root).openAPI()
    #expect(document["paths"]?["/arsenal/v1/models/{id}/runner"]?["post"]?["requestBody"] != nil)
    #expect(document["paths"]?["/arsenal/v1/appearance"] == nil)
    #expect(try await send("GET", "appearance").1 == 404)
    #expect(try await send("POST", "appearance", ["appearance": "dark"]).1 == 404)
}

@Test func v21StatusVersionReadsBundleMetadata() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v21-fix-j/bundle-" + UUID().uuidString + ".bundle")
    let contents = root.appendingPathComponent("Contents")
    try FileManager.default.createDirectory(at: contents, withIntermediateDirectories: true)
    let info: [String: Any] = ["CFBundleIdentifier": "dev.arsenal.version-fixture", "CFBundleShortVersionString": "2.1.0", "CFBundlePackageType": "BNDL"]
    try PropertyListSerialization.data(fromPropertyList: info, format: .xml, options: 0).write(to: contents.appendingPathComponent("Info.plist"))
    let bundle = try #require(Bundle(url: root))
    #expect(APIRouter.releaseVersion(bundle: bundle) == "2.1.0")
}
