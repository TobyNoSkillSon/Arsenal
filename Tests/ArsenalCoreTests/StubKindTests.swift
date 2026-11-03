// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

private actor StubAudioFrames {
    var bytes = Data(), tokens = ""
    func receive(_ frame: WireFrame) {
        switch frame {
        case .binary(_, let data): bytes += data
        case .json(let event): tokens += event["text"]?.string ?? ""
        }
    }
}

@Test func stubAllKinds() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-api/stub-unit-" + UUID().uuidString)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    let service = StubModelsService()
    for entry in await service.catalog() { try await service.add(id: entry.id) }
    // Added without weights: in Get more models, not yet on the table; each Get puts its model there.
    #expect(await service.catalog().allSatisfy { $0.row.state == .notInstalled && !$0.added })
    #expect(await service.rows().isEmpty)
    for entry in await service.catalog() { try await service.get(id: entry.id, precision: "bf16") }
    let catalogCount = await service.catalog().count
    #expect(await service.rows().count == catalogCount)
    let voices = try await service.run(id: "stub-voices", op: "voices")
    #expect(voices["voices"]?.array?.first?["id"]?.string == "af_heart")
    let frames = StubAudioFrames()
    let audio = try await service.run(
        id: "stub-voices", op: "speak", params: ["voice": "af_heart", "text": "Hello", "out_dir": .string(root.path)], sink: { await frames.receive($0) })
    #expect(await frames.bytes.count == 9600)
    let wav = try Data(contentsOf: URL(fileURLWithPath: #require(audio["path"]?.string)))
    #expect(wav.count == 9644 && wav.starts(with: Data("RIFF".utf8)) && wav[8..<12] == Data("WAVE".utf8))
    let image = try await service.run(id: "stub-images", op: "generate", params: ["prompt": "kite", "out_dir": .string(root.path), "n": 2])
    #expect(image["images"]?.array?.count == 2)
    let path = try #require(image["images"]?.array?.first?["path"]?.string)
    #expect(try Data(contentsOf: URL(fileURLWithPath: path)).prefix(8) == Data([137, 80, 78, 71, 13, 10, 26, 10]))
    let questions: JSONValue = [
        "regression": ["type": "noul"], "component": ["type": "choice", "criteria": ["frontend": .null, "infra": .null, "payments": .null]],
        "severity": ["type": "score", "criteria": ["Cosmetic", "Workaround", "Blocked"]]
    ]
    let decision = try await service.run(id: "stub-decisions", op: "decide", params: ["state": "Checkout broken", "questions": questions])
    #expect(decision["answers"]?["regression"]?["noul"]?.number == 0.7696)
    #expect(decision["answers"]?["component"]?["choice"]?.string == "frontend")
    #expect(decision["answers"]?["severity"]?["score"]?.number == 1.0663)
    #expect(decision["answers"]?["severity"]?["legend"]?["2"]?.string == "Blocked")
    let batch = try await service.run(id: "stub-decisions", op: "judge", params: ["items": ["ok", ["image": "no"]], "questions": questions])
    #expect(batch["results"]?.array?[0]["answers"]?["regression"]?["type"] == nil)
    #expect(batch["results"]?.array?[0]["answers"]?["regression"]?["confidence"]?.number == 0.7696)
    #expect(batch["results"]?.array?[1]["ok"]?.bool == false)
    let chat = try await service.run(id: "stub-chat", op: "chat", params: [:], sink: { await frames.receive($0) })
    #expect(await frames.tokens == chat["text"]?.string)
    for row in await service.rows() { try await service.unload(id: row.id) }
    #expect(await service.memory().workersMB == 0)
}
