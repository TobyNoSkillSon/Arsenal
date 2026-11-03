// SPDX-License-Identifier: MIT
import AppKit
import Foundation
@testable import Arsenal
import ArsenalCore
import Testing

@MainActor @Suite(.menuState) struct SettingsLayoutTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    func reference(_ kind: String, values: [String: JSONValue] = [:]) throws -> ([SettingState], SettingsPanel) {
        let (settings, panel) = try SettingsPanel.reference(repo.appendingPathComponent("Resources/kinds/\(kind)"))
        let states = settings.map { d in
            SettingState(declaration: SettingsRenderDelegate.declarationJSON(d), value: values[d.id] ?? d.defaultValue.map(SettingsPanel.json) ?? .null)
        }
        return (states, panel)
    }
    func ids(_ items: [PaneItem]) -> [String] { items.map(\.id) }

    /// The app's rows come first in a fixed order, a hairline, then the model's load settings: every section of its panel
    /// flat (the card has no More), with the panel's labels; request settings never appear.
    @Test func appRowsFirstThenTheLoadSettingsOnly() throws {
        let app = [
            SettingState(declaration: ["id": "__tools", "title": "Show Chat in menu", "type": "toggle", "applies": "app", "default": true], value: .bool(true)),
            SettingState(declaration: ["id": "__keep", "title": "Keep loaded", "type": "toggle", "applies": "app", "default": false], value: .bool(false))
        ]
        let (chat, chatPanel) = try reference("chat")
        let items = PaneLayout.items(app + chat, panel: chatPanel)
        #expect(ids(items) == ["__keep", "__tools", "divider-2", "context_limit"])
        let (images, imagesPanel) = try reference("images")
        #expect(ids(PaneLayout.items(app + images, panel: imagesPanel)) == ["__keep", "__tools"], "an image model's settings are all per request")
        let (decisions, decisionsPanel) = try reference("decisions")
        let kernels = PaneLayout.items(decisions, panel: decisionsPanel)
        #expect(ids(kernels) == ["stock_mlx"], "a load setting behind More shows in the card")
        if case .setting(let s, let style)? = kernels.first?.kind { #expect(s.title(style) == "Kernels" && s.control(style) == .segments) }
        // Height follows the items: padding, rows with gaps, dividers with margins.
        typealias M = SettingsPane.M
        let rows = CGFloat(items.filter { !$0.isDivider }.count), dividers = CGFloat(items.filter(\.isDivider).count)
        let gaps = rows - dividers - 1
        #expect(PaneLayout.height(items, footer: false) == M.top + M.bottom + rows * M.rowHeight + gaps * M.rowGap + dividers * (2 * M.dividerMargin + 1))
    }

    /// A preset row that sets request settings (Shape: width and height) is the image window's, not the card's; the
    /// match rule still names the preset whose values all hold.
    @Test func requestPresetsStayOutOfTheCard() throws {
        let (square, panel) = try reference("images")
        #expect(!PaneLayout.items(square, panel: panel).contains { $0.id.hasPrefix("presets-") })
        let shape = try #require(panel.sections.flatMap(\.rows).first { $0.label == "Shape" })
        #expect(PaneLayout.match(shape.presets, values: ["width": 1024, "height": 1024]).map { shape.presets[$0].title } == "1:1")
        #expect(PaneLayout.match(shape.presets, values: ["width": 1088, "height": 1024]) == nil)
    }

    /// Without a panel the card lists the load settings in declaration order.
    @Test func withoutAPanelTheLoadSettingsAreListed() throws {
        let (states, _) = try reference("chat")
        #expect(ids(PaneLayout.items(states, panel: nil)) == ["context_limit"])
    }

    /// One runner is not a choice (no row); two or more are a picker. A single-choice setting is plain text.
    @Test func oneRunnerHasNoRowAndOneChoiceIsText() {
        func runner(_ names: [String]) -> SettingState {
            SettingState(
                declaration: [
                    "id": .string(TableController.runnerKey), "title": "Runner", "type": "choice", "applies": "load", "default": .string(names[0]),
                    "choices": .array(names.map { .string($0) })
                ], value: .string(names[0]))
        }
        #expect(PaneLayout.items([runner(["native"])], panel: nil).isEmpty)
        #expect(ids(PaneLayout.items([runner(["native", "mlx-lm"])], panel: nil)) == [TableController.runnerKey])
        let one = SettingState(declaration: ["id": "context_limit", "type": "choice", "choices": [262144], "default": 262144, "applies": "load"], value: .number(262144))
        #expect(SettingsPane.plainValue(one) && !SettingsPane.plainValue(runner(["native", "mlx-lm"])))
    }

    /// While a Get or Load runs only load-time rows lock, and the panel says why under them.
    @Test func busyLocksOnlyLoadRowsAndSaysWhy() throws {
        let (states, panel) = try reference("chat")
        let keep = SettingState(declaration: ["id": "__keep", "title": "Keep loaded", "type": "toggle", "applies": "app", "default": false], value: .bool(false))
        let items = PaneLayout.items([keep] + states, panel: panel, lockReason: "Locked while loading")
        #expect(items.filter(PaneLayout.locked).map(\.id) == ["context_limit"])
        // The reason sits under the locked row, not at the foot.
        let notice = try #require(items.firstIndex { $0.kind == .notice("Locked while loading") })
        #expect(items[notice - 1].id == "context_limit")
        let (decisions, decisionsPanel) = try reference("decisions")
        // The card has no More: the load setting behind it shows, and the notice sits under it.
        #expect(ids(PaneLayout.items(decisions, panel: decisionsPanel, lockReason: "x")) == ["stock_mlx", "notice"])
        let row = ModelRow(id: "m", name: "M", kind: "chat", states: ["bf16": .init(state: .loading)])
        #expect(SettingsPane.lockReason(row) == "Locked while loading")
        #expect(SettingsPane.lockReason(ModelRow(id: "m", name: "M", kind: "chat", states: ["bf16": .init(state: .ready)])) == nil)
    }

    /// A setting's own row never repeats a preset row's label: video's and sound's "Length" presets sat above a raw
    /// "Length 5 s" row under More.
    @Test func noRowRepeatsAPresetRowsLabel() throws {
        for kind in SettingsPanel.referenceDirectories {
            let (settings, panel) = try SettingsPanel.reference(repo.appendingPathComponent("Resources/kinds/\(kind)"))
            let rows = panel.sections.flatMap(\.rows)
            let presets = Set(rows.filter(\.isPresets).compactMap(\.label))
            let titles = rows.compactMap { row in row.setting.map { id in row.label ?? settings.first { $0.id == id }?.title ?? id } }
            #expect(presets.isDisjoint(with: titles), "\(kind): \(presets.intersection(titles))")
        }
    }

    /// Restore defaults puts back the card's load settings that can change now, never Keep loaded and never a request
    /// setting (the apps' business): while a Get or Load runs nothing restores; when Arsenal quits nothing restores.
    @Test func restoreTouchesOnlyLoadSettingsThatAreEnabled() async throws {
        var row = ModelRow(id: "m", name: "M", kind: "chat", states: ["bf16": PrecisionState(state: .ready)])
        row.keepHot = true
        let metadata: JSONObject = [
            "setting": [
                ["id": "context", "title": "Context", "type": "choice", "default": 1024, "choices": [1024, 2048], "applies": "load"],
                ["id": "tone", "title": "Tone", "type": "choice", "default": "a", "choices": ["a", "b"], "applies": "request", "maps_to": "request.tone"]
            ]
        ]
        let service = FixtureService(rows: [row], metadata: [row.id: metadata], kinds: Fixtures.kinds)
        try await service.setSettings(id: row.id, values: ["context": 2048, "tone": "b"])
        try await service.setKeepHot(id: row.id, on: true)
        let c = TableController(service: service)
        await c.refresh()
        let ready = try #require(c.rows.first)
        #expect(c.restorable(ready).map(\.id) == ["context"])
        // Busy: the load-time setting is locked, so nothing restores.
        var busy = ready
        busy.states[busy.selectedPrecision] = PrecisionState(state: .loading)
        c.apply(rows: [busy], kinds: Fixtures.kinds)
        #expect(c.restorable(busy).isEmpty)
        c.apply(rows: [ready], kinds: Fixtures.kinds)
        c.restoreDefaults(ready)
        try await waitUntil { (try? await service.settings(id: row.id).first { $0.id == "context" }?.value) == 1024 }
        #expect((try await service.settings(id: row.id)).first { $0.id == "tone" }?.value == "b")
        let keepHot = await service.rows().first?.keepHot
        #expect(c.lastError == nil && keepHot == true)
        // Quitting: nothing restores.
        try await service.setSettings(id: row.id, values: ["context": 2048])
        await c.refresh()
        c.admitting = false
        #expect(c.restorable(try #require(c.rows.first)).isEmpty)
    }

    /// The runner picker comes after both switches, and every runner is named the same way.
    @Test func runnerPickerComesLastAndNamesAlike() {
        let row = ModelRow(id: "m", name: "M", kind: "chat")
        let runner = try? #require(
            TableController.runnerSetting(
                row, [CatalogRunner(name: "native", language: "swift", engine: "mlx", acceptedDefault: true), CatalogRunner(name: "mlx-lm", language: "swift", engine: "mlx")]))
        guard let runner else { return }
        #expect(runner.choices.map { runner.display($0) } == ["Native (Swift \u{00b7} MLX)", "mlx-lm (Swift \u{00b7} MLX)"])
        let keep = SettingState(declaration: ["id": "__keep", "type": "toggle", "applies": "app", "default": false], value: .bool(false))
        let tools = SettingState(declaration: ["id": "__tools", "type": "toggle", "applies": "app", "default": true], value: .bool(true))
        #expect(ids(PaneLayout.items([runner, tools, keep], panel: nil)) == ["__keep", "__tools", TableController.runnerKey])
    }

    /// One voice title loaded or not; language headings only when the voices span languages.
    @Test func voiceTitlesAndHeadings() {
        let us = VoiceChoice(id: "af_heart", name: "Heart", language: "en-US"), uk = VoiceChoice(id: "bf_emma", name: "Emma", language: "en-GB")
        #expect(VoiceChoice.title("af_heart", voices: []) == "Heart" && VoiceChoice.title("af_heart", voices: [us]) == "Heart")
        #expect(VoiceChoice.title("", voices: [us]) == "Default")
        #expect(VoiceChoice.rows([us, us]) == [.voice(us), .voice(us)])
        #expect(VoiceChoice.rows([us, uk]) == [.heading("American English"), .voice(us), .heading("British English"), .voice(uk)])
    }

    /// The card has no multi-line editor: a text setting a panel draws as an editor is a one-line field there.
    @Test func anEditorRowIsAOneLineFieldInTheCard() {
        let prompt = SettingState(declaration: ["id": "prompt", "title": "Prompt", "type": "text", "applies": "load"], value: "")
        #expect(prompt.control(PaneStyle(control: .editor)) == .text)
    }

    /// Opening a model's card leaves the real Models menu host exactly as it was: the card floats over the table.
    @Test func theMenuHostStaysWhenTheCardOpens() async throws {
        let folder = repo.appendingPathComponent("models/qwen3.8-27b")
        let (settings, _, _, _, _) = try SettingsRenderDelegate.folder(folder)
        let metadata: JSONObject = [
            "setting": .array(settings.map { .object(SettingsRenderDelegate.declarationJSON($0)) }),
            "settings_toml": .string(try String(contentsOf: folder.appendingPathComponent(SettingsPanel.fileName), encoding: .utf8))
        ]
        let row = ModelRow(id: "qwen3.8-27b", name: "Qwen3.8 27B", kind: "chat", states: ["bf16": .init(state: .ready)])
        let controller = TableController(service: FixtureService(rows: [row], metadata: [row.id: metadata]))
        await controller.refresh()
        #expect(controller.panels[row.id] != nil)
        let host = SettingsCardHost(controller)
        defer { host.close(); controller.stop() }
        await host.settle()
        let table = host.view.frame, natural = host.view.naturalSize
        await host.open(try #require(controller.rows.first))
        #expect(host.isOpen && host.height > 0)
        #expect(host.view.frame == table && host.view.naturalSize == natural)
        controller.settingsCard.close()
        await host.settle()
        #expect(host.view.frame == table && host.view.naturalSize == natural)
    }

    /// The Image window starts from the model's settings: a free size from width and height (joining the window's list
    /// when it lacks it), the steps, and a pinned seed; an unset seed stays empty (random each time).
    @Test func theImageWindowReadsTheModelsSettings() {
        let session = ImageSession(model: V3RenderDelegate.flux, client: { nil })
        let schema: JSONObject = [
            "properties": [
                "prompt": ["type": "string"], "width": ["minimum": 64, "maximum": 2048], "height": ["minimum": 64, "maximum": 2048],
                "steps": ["type": "integer", "minimum": 1, "maximum": 100], "seed": ["type": "integer"]
            ]
        ]
        session.preview(operations: [OperationDescriptor(name: "generate", title: "Generate", inputSchema: schema, output: "image", stream: false)])
        func state(_ id: String, _ value: JSONValue) -> SettingState { SettingState(declaration: ["id": .string(id)], value: value) }
        session.apply(settings: [state("width", 1344), state("height", 768), state("steps", 6), state("seed", .null)])
        #expect(session.value("size").text == "1344x768" && session.value("steps").text == "6" && session.value("seed").text.isEmpty)
        // A free size the list lacks is still offered and sent.
        session.apply(settings: [state("width", 1088), state("height", 1024), state("seed", 421_337)])
        #expect(session.value("size").text == "1088x1024" && session.value("seed").text == "421337")
        session.set("prompt", .text("A lighthouse"))
        #expect(session.canRun)
        session.apply(settings: [state("seed", .null)])
        #expect(session.value("seed").text.isEmpty, "Clearing a model's pinned seed restores random in an existing session")
    }

    /// Every icon a reference names is an SF Symbol on this macOS (a missing one would fall back to its title).
    @Test func referenceIconsExist() throws {
        for kind in SettingsPanel.referenceDirectories {
            let panel = try SettingsPanel.reference(repo.appendingPathComponent("Resources/kinds/\(kind)")).panel
            let icons = panel.sections.flatMap(\.rows).flatMap { $0.presets.compactMap(\.icon) + Array($0.icons.values) }
            for icon in icons { #expect(NSImage(systemSymbolName: icon, accessibilityDescription: nil) != nil, "\(kind): \(icon)") }
        }
    }
}
