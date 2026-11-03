// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct SettingsPanelTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    static let declarations: [SettingDeclaration] = [
        SettingDeclaration(id: "width", title: "Width", type: "number", defaultValue: .number(1024), min: 256, max: 2048, step: 64, applies: "request", mapsTo: "request.width"),
        SettingDeclaration(id: "height", title: "Height", type: "number", defaultValue: .number(1024), min: 256, max: 2048, step: 64, applies: "request", mapsTo: "request.height"),
        SettingDeclaration(
            id: "mode", title: "Mode", type: "choice", defaultValue: .string("a"), choices: [.string("a"), .string("b")], applies: "request", mapsTo: "request.mode"),
        SettingDeclaration(id: "fast", title: "Fast", type: "toggle", defaultValue: .bool(true), applies: "load")
    ]
    static let valid = """
        schema = 1
        hidden = ["fast"]
        [[section]]
        [[section.row]]
        label = "Shape"
        presets = [
          { title = "Square", icon = "square", values = { width = 1024, height = 1024 } },
          { title = "Wide", icon = "rectangle", values = { width = 1344, height = 768 } },
        ]
        [[section.row]]
        setting = "mode"
        labels = { a = "Alpha", b = "Beta" }
        show_when = { fast = true }
        [[section]]
        title = "More"
        collapsed = true
        [[section.row]]
        setting = "width"
        unit = "px"
        randomize = true
        [[section.row]]
        setting = "height"
        """

    @Test func aValidPanelReadsInOrder() throws {
        let panel = try SettingsPanel.parse(Self.valid, settings: Self.declarations)
        #expect(panel.sections.count == 2 && panel.sections.map(\.rows.count) == [2, 2] && panel.hidden == ["fast"])
        let presets = panel.sections[0].rows[0]
        #expect(presets.isPresets && presets.presets.map(\.title) == ["Square", "Wide"] && presets.presets[1].values == ["width": .number(1344), "height": .number(768)])
        #expect(panel.sections[0].rows[1].labels == ["a": "Alpha", "b": "Beta"] && panel.sections[0].rows[1].showWhen == ["fast": [.bool(true)]])
        #expect(panel.sections[1].collapsed && panel.sections[1].title == "More" && panel.sections[1].rows[0].randomize && panel.sections[1].rows[0].unit == "px")
        #expect(SettingsPanel(json: panel.json) == panel)
    }

    /// Each mistake an agent can make names where it is and what would fix it.
    @Test(arguments: [
        ("setting = \"cfg\"", "cfg is not a setting declared in model.toml"),
        ("setting = \"mode\"\ncontrol = \"slider\"", "control slider does not fit a choice setting"),
        ("setting = \"mode\"\nlabels = { c = \"Gamma\" }", "labels.c is not a value of mode"),
        ("setting = \"mode\"\ncolour = \"red\"", "unknown key colour"),
        ("label = \"Size\"\npresets = [{ title = \"A\", values = { width = 1000 } }, { title = \"B\", values = { width = 1024 } }]", "width does not accept 1000"),
        ("label = \"Size\"\npresets = [{ title = \"A\", values = { width = 512 } }, { title = \"B\", values = { height = 512 } }]", "every preset in a row sets the same settings"),
        ("setting = \"mode\"\nrandomize = true", "randomize needs a number or slider setting"),
        ("presets = [{ title = \"A\", values = { width = 512 } }, { title = \"B\", values = { width = 768 } }]", "a preset row needs a label")
    ])
    func mistakesAreNamed(row: String, expected: String) {
        let text = "schema = 1\nhidden = [\"width\", \"height\", \"mode\", \"fast\"]\n[[section]]\n[[section.row]]\n" + row + "\n"
        #expect(throws: ManifestError.self) { try SettingsPanel.parse(text, settings: Self.declarations) }
        do { _ = try SettingsPanel.parse(text, settings: Self.declarations) } catch {
            #expect(error.localizedDescription.contains(expected), "\(error.localizedDescription)")
            #expect(error.localizedDescription.contains("section 1, row 1"))
        }
    }

    @Test func everyDeclaredSettingHasARowOrIsHidden() {
        let text = Self.valid.replacingOccurrences(of: "hidden = [\"fast\"]\n", with: "")
        #expect(throws: ManifestError.self) { try SettingsPanel.parse(text, settings: Self.declarations) }
        do { _ = try SettingsPanel.parse(text, settings: Self.declarations) } catch {
            #expect(error.localizedDescription.contains("fast is declared in model.toml but has no row"))
        }
        // A preset may only set what also has a row, so a value no preset matches can be seen.
        let presetOnly = Self.valid.replacingOccurrences(of: "[[section.row]]\nsetting = \"height\"", with: "").replacingOccurrences(
            of: "hidden = [\"fast\"]", with: "hidden = [\"fast\", \"height\"]")
        #expect((try? SettingsPanel.parse(presetOnly, settings: Self.declarations)) != nil)
        let headed = Self.valid.replacingOccurrences(of: "[[section]]\n[[section.row]]\nlabel", with: "[[section]]\ntitle = \"Size\"\n[[section.row]]\nlabel")
        do { _ = try SettingsPanel.parse(headed, settings: Self.declarations); Issue.record("a heading was accepted") } catch {
            #expect(error.localizedDescription.contains("title is only for a collapsed section"))
        }
    }

    /// Each kind's reference is valid: its declarations, and a panel that arranges every one of them.
    @Test func everyKindReferenceIsValid() throws {
        for kind in SettingsPanel.referenceDirectories {
            let (settings, panel) = try SettingsPanel.reference(repo.appendingPathComponent("Resources/kinds/\(kind)"))
            #expect(!settings.isEmpty && !panel.sections.isEmpty, "\(kind)")
            let rows = Set(panel.sections.flatMap(\.rows).compactMap(\.setting))
            #expect(rows.union(panel.hidden) == Set(settings.map(\.id)), "\(kind)")
        }
        // The kinds whose models usually tune them get presets.
        for kind in ["images", "chat", "voices", "sound", "video"] {
            let panel = try SettingsPanel.reference(repo.appendingPathComponent("Resources/kinds/\(kind)")).panel
            #expect(panel.sections.flatMap(\.rows).contains { $0.isPresets }, "\(kind)")
        }
    }

    /// A request-time number setting may leave out its default (a seed: random until pinned); nothing else may. Clearing a pinned
    /// value returns to the runner's choice, and nothing is sent for it.
    @Test func aNumberWithoutADefaultIsUnsetUntilPinned() throws {
        let seed = SettingDeclaration(id: "seed", title: "Seed", type: "number", defaultValue: nil, min: 0, max: 4_294_967_295, step: 1, applies: "request", mapsTo: "request.seed")
        #expect(seed.problems().isEmpty)
        #expect(
            SettingDeclaration(id: "t", title: "T", type: "toggle", defaultValue: nil, applies: "request", mapsTo: "request.t").problems().contains {
                $0.contains("needs a default")
            })
        // A load setting always reaches the runner's config: it may not leave its default out (it would send null).
        let limit = SettingDeclaration(id: "limit", title: "Limit", type: "number", defaultValue: nil, min: 1, max: 10, applies: "load", mapsTo: "config.limit")
        #expect(limit.problems().contains { $0.contains("needs a default") })
        var settings = ModelSettings()
        settings.declarations["m"] = [try WireJSON.decode(JSONEncoder().encode(seed))]
        #expect(settings.states("m", loaded: false).first?.value == .null)
        #expect(settings.request("m", params: ["prompt": "x"])["seed"] == nil)
        try settings.set("m", key: "seed", value: .number(7))
        #expect(settings.request("m", params: ["prompt": "x"])["seed"] == .number(7))
        try settings.set("m", key: "seed", value: .null)
        #expect(settings.states("m", loaded: false).first?.value == .null && settings.request("m", params: [:])["seed"] == nil)
    }

    /// A folder's panel is checked with the rest of the folder: a panel naming an undeclared setting fails validation.
    @Test func folderValidationReadsThePanel() throws {
        let source = repo.appendingPathComponent("models/flux2-klein-4b")
        let folder = FileManager.default.temporaryDirectory.appendingPathComponent("panel-\(UUID().uuidString)/flux2-klein-4b")
        defer { try? FileManager.default.removeItem(at: folder.deletingLastPathComponent()) }
        try FileManager.default.createDirectory(at: folder.deletingLastPathComponent(), withIntermediateDirectories: true)
        try FileManager.default.copyItem(at: source, to: folder)
        let kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
        let manifest = try ModelManifest.load(folder, kinds: kinds, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        try "schema = 1\n[[section]]\n[[section.row]]\nsetting = \"guidance\"\n".write(to: folder.appendingPathComponent("settings.toml"), atomically: true, encoding: .utf8)
        let problems = manifest.problems(folder: folder, kinds: kinds, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
        #expect(problems.contains { $0.hasPrefix("flux2-klein-4b: settings.toml: section 1, row 1: guidance is not a setting declared in model.toml") }, "\(problems)")
    }
}
