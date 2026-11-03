// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) struct SettingsToolRequestTests {
    @Test func modelsMenuKeepsTheFilePickerHook() async throws {
        let source = try SettingsPanel.reference(SettingsRenderDelegate.repo.appendingPathComponent("Resources/kinds/voices/design"))
        // On the table: its weights are on this Mac.
        let row = ModelRow(id: "file-setting", name: "Voice", kind: "voices", states: ["bf16": PrecisionState(state: .ready)])
        let metadata: JSONObject = ["setting": .array(source.settings.map { .object(SettingsRenderDelegate.declarationJSON($0)) })]
        let service = FixtureService(rows: [row], metadata: [row.id: metadata])
        let table = TableController(service: service)
        defer { table.stop() }
        await table.refresh()
        let menu = ModelsMenu(controller: table)
        _ = menu.modelItem()
        var selected: String? = "/tmp/fixture-reference.wav", calls = 0
        menu.presentOpenPanel = { panel in
            calls += 1
            #expect(!panel.canChooseDirectories && !panel.allowsMultipleSelection && panel.message == "Reference clip")
            return selected
        }
        let setting = try #require(table.settings[row.id]?.first { $0.id == "reference_path" })
        table.chooseFile(row, setting)
        try await waitUntil { (try? await service.settings(id: row.id).first { $0.id == setting.id }?.value.string) == selected }
        selected = nil
        table.chooseFile(row, setting)
        #expect(calls == 2)
        #expect((try await service.settings(id: row.id)).first { $0.id == setting.id }?.value == "/tmp/fixture-reference.wav")
    }

    /// Menu admission and both tool sessions use the same HTTP run route after loading model-folder settings.
    @Test(arguments: [false, true]) func toolSettingsAgreeWithAdmissionAndHTTP(resident: Bool) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-settings-tools-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let ids = ["qwen3.8-27b", "flux2-klein-4b"]
        var rows: [ModelRow] = [], metadata: [String: JSONObject] = [:]
        for id in ids {
            let source = try SettingsRenderDelegate.folder(SettingsRenderDelegate.repo.appendingPathComponent("models/\(id)"))
            var row = ModelRow(id: id, name: source.name, kind: source.kind, states: ["bf16": .init(state: .ready)])
            row.tools = source.tools
            rows.append(row)
            metadata[id] = ["setting": .array(source.settings.map { .object(SettingsRenderDelegate.declarationJSON($0)) })]
        }
        let fluxSchema = try WireJSON.decode(Data(contentsOf: SettingsRenderDelegate.repo.appendingPathComponent("models/flux2-klein-4b/ops/generate.input.json")))
        let generate = ModelOperation(name: "generate", title: "Generate", inputSchema: fluxSchema, output: "image", stream: false)
        let service = FixtureService(rows: rows, metadata: metadata, supportDirectory: root.appendingPathComponent("support"), operations: [ids[1]: [generate]])
        if resident { for id in ids { try await service.load(id: id, precision: "bf16", fast: false) } }
        try await service.setSettings(id: ids[0], values: ["max_tokens": 128])
        try await service.setSettings(id: ids[1], values: ["width": 1088, "height": 1024, "seed": 421_337])
        let api = APIHost(service: service, support: root.appendingPathComponent("support"), port: 0)
        await api.start()
        let center = ToolCenter(client: { api.port.map { ArsenalClient(port: $0) } }, playback: FixturePlayback())
        defer { center.closeAll(); api.stop() }
        let current = await service.rows()
        center.applyAdmissions(await center.requestAdmissions(rows: current, service: service))
        for row in current {
            let tool: ToolKind = row.kind == "chat" ? .chat : .image
            #expect(await service.runRefusal(id: row.id, op: row.kind == "chat" ? "chat" : "generate") == nil)
            #expect(center.blocked(row, tool: tool) == nil)
        }
        let port = try #require(api.port)
        let chat = ChatSession(model: V3RenderDelegate.qwen, client: { ArsenalClient(port: port) })
        let image = ImageSession(model: V3RenderDelegate.flux, client: { ArsenalClient(port: port) }, outputDirectory: root.appendingPathComponent("pictures"))
        defer { chat.close(); image.discard() }
        image.prepare()
        try await waitUntil { !image.loading }
        #expect(image.value("size").text == "1088x1024" && !image.canRun && image.value("seed").text == "421337")
        image.set("prompt", .text("A lighthouse"))
        #expect(image.canRun)
        image.start()
        try await waitUntil { !image.running }
        #expect(image.problem == nil && image.results.last?.seed == 421_337)
        let pinned = try #require(await service.runInputs.last)
        #expect(pinned["width"] == 1088 && pinned["height"] == 1024 && pinned["seed"] == 421_337 && pinned["resolution"] == nil)
        try await service.setSettings(id: ids[1], values: ["seed": .null])
        image.apply(settings: try await service.settings(id: ids[1]))
        #expect(image.value("seed").text.isEmpty)
        image.start()
        try await waitUntil { !image.running }
        #expect(image.problem == nil)
        let random = try #require(await service.runInputs.last?["seed"]?.number)
        #expect((0...Double(UInt32.max)).contains(random) && random == random.rounded())
        chat.draft = "Say hello"
        chat.send()
        try await waitUntil { !chat.generating }
        #expect(chat.messages.last?.error == nil && chat.messages.last?.text == "Hello from Arsenal.")
        let input = try #require(await service.runInputs.last)
        #expect(input["max_tokens"] == nil, "The model's live Answer limit applies instead of a tool override")
        #expect((try await service.settings(id: ids[0])).first { $0.id == "max_tokens" }?.value == 128)
        #expect(await service.runLog() == [ids[1] + " generate", ids[1] + " generate", ids[0] + " chat"])
        #expect(await service.rows().allSatisfy { $0.residentDType != nil })
        await service.shutdown()
    }
}
