// SPDX-License-Identifier: MIT
// `ARSENAL_QA="render-settings <dir>"`: each kind's reference settings panel (kinds/<kind>/settings.toml over the
// declarations in reference.toml) in the model's settings card as the gear opens it beside its row, at real size and 2×
// pixels, dark and light, on a stand-in of the menu material, plus the states that matter: values that match no preset,
// a pending load setting (Reload at the foot), a busy model, a loaded voice model, and a model without a panel (the
// generic list). `render-model-settings`: the models in the user's table from their own folders, in the states the brief
// names (not on this Mac, busy, pending, a voice model loaded and not). Each state also gets a dark|light sheet. No
// service work, no screen interaction: nothing is ordered on screen.
import AppKit
@testable import Arsenal
import ArsenalCore
import SwiftUI

@MainActor final class SettingsRenderDelegate: NSObject, NSApplicationDelegate {
    let directory: URL
    /// `render-model-settings`: the models in the user's table, from their own folders, instead of the kind references.
    let models: Bool
    /// `render-withdrawn`: these shots instead of the kind references or the user's table.
    var only: [Shot]?
    init(directory: URL, models: Bool = false) { self.directory = directory; self.models = models }
    /// The Models table's rows (the installed app's `table`, read once on 11 Oct 2026), and Kokoro for a voice model.
    static let tableModels = ["flux2-klein-4b", "qwen3.8-27b", "laya-multilingual", "laya-typed-decisions", "d1-3b", "kokoro-82m"]
    static func modelShots() -> [Shot] {
        tableModels.flatMap { id -> [Shot] in
            [Shot(name: id, dir: "", model: id)]
        } + [
            // Qwen as the user's table has it: not on Arsenal's store (Get).
            Shot(name: "qwen3.8-27b-not-on-mac", dir: "", state: .notInstalled, model: "qwen3.8-27b"),
            Shot(name: "qwen3.8-27b-busy", dir: "", state: .loading, model: "qwen3.8-27b"),
            Shot(name: "qwen3.8-27b-pending", dir: "", values: ["context_limit": .number(131_072)], pending: ["context_limit"], state: .loaded, model: "qwen3.8-27b"),
            // A Reload that failed: the window says why above the button.
            Shot(
                name: "qwen3.8-27b-reload-failed", dir: "", values: ["context_limit": .number(524_288)], pending: ["context_limit"], state: .loaded,
                model: "qwen3.8-27b", error: "Not enough free memory for a 512K context; the model stays loaded at 256K.",
                choices: ["context_limit": [32_768, 65_536, 131_072, 262_144, 524_288].map { .number($0) }]),
            Shot(name: "flux2-klein-4b-custom", dir: "", values: ["width": .number(1088), "steps": .number(20), "seed": .number(421_337)], model: "flux2-klein-4b"),
            Shot(name: "kokoro-82m-loaded", dir: "", values: ["voice": .string("af_heart")], state: .loaded, voices: true, model: "kokoro-82m")
        ]
    }
    /// A model folder's declarations, panel (nil without settings.toml), name, kind and tools.
    static func folder(_ folder: URL) throws -> (settings: [SettingDeclaration], panel: SettingsPanel?, name: String, kind: String, tools: [String]) {
        let manifest = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
        return (manifest.settings, try SettingsPanel.load(folder: folder, settings: manifest.settings), manifest.name, manifest.kind, manifest.tools ?? [])
    }
    nonisolated static var repo: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }

    struct Shot {
        var name: String
        /// The reference directory under kinds/ ("images", "images/resolution").
        var dir: String
        var panel = true
        var values: [String: JSONValue] = [:]
        var pending: Set<String> = []
        var state: ModelState = .ready
        var voices = false
        var runners = ["native"]
        /// The first field has keyboard focus (a field being edited).
        var focus = false
        /// A setting reduced to one choice (its default): a single-choice row.
        var single: String?
        /// A model folder (`models/<id>`) instead of a kind reference.
        var model: String?
        /// The model's last failed action, as the settings show it.
        var error: String?
        /// Old precisions on this Mac that the catalog no longer offers (the card's Remove footer).
        var withdrawn: [String: Int64] = [:]
        /// A setting's choices replaced (a Context offered as a real pop-up).
        var choices: [String: [JSONValue]] = [:]
        var kind: String { String(dir.split(separator: "/").first ?? "") }
    }
    static let titles = [
        "images": "Image model", "images/resolution": "FLUX.2 Klein 4B", "chat": "Qwen3.8 27B", "decisions": "Laya Multilingual", "voices": "Kokoro 82M",
        "voices/design": "Qwen3 TTS VoiceDesign", "sound": "Stable Audio 3 Small", "enhancement": "RE-USE", "video": "Video model"
    ]
    static let kokoro = ["af_heart", "af_bella", "am_adam", "am_michael", "bf_emma", "bm_george"]
    static func shots() -> [Shot] {
        SettingsPanel.referenceDirectories.map { Shot(name: $0.replacingOccurrences(of: "/", with: "-"), dir: $0) } + [
            Shot(name: "images-custom", dir: "images", values: ["width": .number(1088), "steps": .number(20), "seed": .number(421_337)]),
            Shot(name: "images-editing", dir: "images", values: ["seed": .number(421_337)], focus: true),
            Shot(name: "chat-busy", dir: "chat", values: ["temperature": .number(0.2), "top_p": .number(0.9)], state: .loading),
            Shot(
                name: "chat-pending", dir: "chat", values: ["context_limit": .number(262144), "temperature": .number(0.2), "top_p": .number(0.9)], pending: ["context_limit"],
                state: .loaded),
            Shot(name: "chat-runners", dir: "chat", runners: ["native", "mlx-lm"]),
            Shot(name: "chat-single", dir: "chat", single: "context_limit"),
            Shot(name: "voices-unloaded", dir: "voices", values: ["voice": .string("af_heart")]),
            Shot(name: "voices-loaded", dir: "voices", values: ["voice": .string("af_heart"), "speed": .number(1.2)], state: .loaded, voices: true),
            Shot(name: "chat-generic", dir: "chat", panel: false),
            Shot(name: "chat-refused", dir: "chat", error: "Thinking is not supported by this runner.")
        ]
    }

    /// `render-withdrawn`: the Remove footer alone, with one and with several leftovers, and beside a pending Reload.
    static func withdrawnShots() -> [Shot] {
        [
            Shot(name: "qwen3.8-27b-withdrawn-one", dir: "", model: "qwen3.8-27b", withdrawn: ["int8": 1_200_000_000]),
            Shot(name: "qwen3.8-27b-withdrawn-several", dir: "", model: "qwen3.8-27b", withdrawn: ["int4": 600_000_000, "int8": 1_200_000_000]),
            Shot(
                name: "qwen3.8-27b-withdrawn-reload", dir: "", values: ["context_limit": .number(131_072)], pending: ["context_limit"], state: .loaded,
                model: "qwen3.8-27b", withdrawn: ["int4": 600_000_000, "int8": 1_200_000_000])
        ]
    }

    static func declarationJSON(_ d: SettingDeclaration) -> JSONObject {
        (try? WireJSON.decode(JSONEncoder().encode(d))) ?? [:]
    }

    static func controller(_ shot: Shot) throws -> (TableController, ModelRow) {
        let source: (settings: [SettingDeclaration], panel: SettingsPanel?, name: String, kind: String, tools: [String])
        if let model = shot.model {
            source = try folder(repo.appendingPathComponent("models/\(model)"))
        } else {
            let reference = try SettingsPanel.reference(repo.appendingPathComponent("Resources/kinds/\(shot.dir)"))
            source = (
                reference.settings, reference.panel, titles[shot.dir] ?? shot.dir, shot.kind, ["chat": ["chat"], "images": ["image"], "voices": ["voice"]][shot.kind] ?? []
            )
        }
        let reference = (settings: source.settings, panel: source.panel)
        var row = ModelRow(id: shot.model ?? "ref-\(shot.kind)", name: source.name, kind: source.kind, states: ["bf16": .init(state: shot.state)])
        row.tools = source.tools
        row.toolsShown = true
        row.withdrawnTiers = shot.withdrawn
        if shot.state == .loaded { row.residentRunner = row.runner; row.needsReload = !shot.pending.isEmpty }
        // The card never grows past its table: a withdrawn shot's table holds other rows, as the owner's does, so the foot shows.
        let table = shot.withdrawn.isEmpty ? [row] : [row] + Fixtures.rows(.loaded).filter { $0.kind != "chat" }
        let c = TableController(service: FixtureService(rows: table), chip: "M5 Max")
        c.apply(rows: table, kinds: Fixtures.kinds)
        var states: [SettingState] = [
            SettingState(
                declaration: ["id": "__keep", "title": "Keep loaded", "type": "toggle", "applies": "app", "default": false, "help": "Stays in memory until you unload it."],
                value: .bool(false))
        ]
        let runners = shot.runners.enumerated().map { CatalogRunner(name: $1, language: $1 == "native" ? "swift" : "python", engine: "mlx", acceptedDefault: $0 == 0) }
        if let runner = TableController.runnerSetting(row, runners) { states.append(runner) }
        if let tools = TableController.toolsSetting(row) { states.append(tools) }
        states += reference.settings.map { d in
            var declaration = declarationJSON(d)
            if shot.single == d.id, let only = declaration["default"] { declaration["choices"] = .array([only]) }
            if let choices = shot.choices[d.id] { declaration["choices"] = .array(choices) }
            return SettingState(declaration: declaration, value: shot.values[d.id] ?? d.defaultValue.map(SettingsPanel.json) ?? .null, pending: shot.pending.contains(d.id))
        }
        // Its settings, panel and voices first (the menu sizes its columns to them when it opens; a loaded voice model
        // then has its voices). The menu is never popped up; the render draws its view off every screen.
        c.settings[row.id] = states
        if shot.panel, let panel = reference.panel { c.panels[row.id] = panel }
        if shot.voices {
            c.showVoices(
                row.id,
                ["voices": .array(kokoro.map { ["id": .string($0), "name": .string($0), "language": .string($0.hasPrefix("b") ? "en-GB" : "en-US")] })])
        }
        c.openSettings(row)
        // Opening starts without an old failure: the shot's failure is the one these settings saw.
        c.rowErrors[row.id] = shot.error
        return (c, row)
    }

    func applicationDidFinishLaunching(_ notification: Notification) {
        NSApp.setActivationPolicy(.accessory)
        ToolWindow.keepOffscreenFrames = true
        try? FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        Task { @MainActor in
            var log: [String] = []
            for shot in only ?? (models ? Self.modelShots() : Self.shots()) {
                var parts: [NSImage] = []
                for mode in [RenderAppearance.dark, .light] {
                    Appearance.shared.set(mode); NSApp.appearance = Appearance.shared.nsAppearance
                    let url = directory.appendingPathComponent("settings-\(shot.name)-\(mode.rawValue).png")
                    do {
                        let (c, row) = try Self.controller(shot)
                        let size = try await Self.render(c, row: row, to: url, focus: shot.focus)
                        if mode == .dark {
                            log.append("\(shot.name): \(Int(size.width))×\(Int(size.height)) pt; " + Self.describe(c, row: row))
                            // The Model hover's note (a native tooltip, so it is written here, not drawn).
                            log.append("\(shot.name) model hover: " + (ModelHelp.withdrawn(row) ?? "none"))
                        }
                        c.settingsCard.close()
                        if let image = NSImage(contentsOf: url) { parts.append(image) }
                    } catch { log.append("\(shot.name): \(error.localizedDescription)") }
                }
                Self.sheet(parts, to: directory.appendingPathComponent("sheet-\(shot.name).png"))
            }
            Appearance.shared.set(.dark)
            try? log.joined(separator: "\n").write(to: directory.appendingPathComponent("settings-renders.txt"), atomically: true, encoding: .utf8)
            QA.stopApp()
        }
    }

    /// What the settings list, for the log: each item's label and current value.
    static func describe(_ c: TableController, row: ModelRow) -> String {
        SettingsPane.items(c.settings[row.id] ?? [], panel: c.panels[row.id], lockReason: SettingsPane.lockReason(row), error: c.rowErrors[row.id])
            .map { item -> String in
                switch item.kind {
                case .divider: return "|"
                case .setting(let s, let style): return "\(s.title(style))=\(s.display(s.value, style))"
                case .presets(let label, let presets, let selected, _): return "\(label)=[\(selected.map { presets[$0].title } ?? "none")]"
                case .notice(let text): return "notice(\(text))"
                case .error(let text): return "error(\(text))"
                }
            }.joined(separator: " ")
    }

    /// The model's settings card as the gear opened it, in its table, on a stand-in of the menu material, at 2× pixels.
    static func render(_ c: TableController, row: ModelRow, to url: URL, focus: Bool = false) async throws -> NSSize {
        try await SettingsCardRender.render(c, row: row, to: url, focus: focus)
    }

    static func fields(in view: NSView) -> [MenuField] { (view as? MenuField).map { [$0] } ?? view.subviews.flatMap(fields) }

    /// The dark and light renders side by side at their own pixel size, on grey.
    static func sheet(_ images: [NSImage], to url: URL) {
        guard images.count == 2, let reps = images.map({ $0.representations.first as? NSBitmapImageRep }) as? [NSBitmapImageRep] else { return }
        let w = reps.map(\.pixelsWide).max() ?? 0, h = reps.map(\.pixelsHigh).max() ?? 0, gap = 24
        guard
            let out = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: 2 * w + 3 * gap, pixelsHigh: h + 2 * gap, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0)
        else { return }
        NSGraphicsContext.saveGraphicsState()
        NSGraphicsContext.current = NSGraphicsContext(bitmapImageRep: out)
        NSColor(white: 0.5, alpha: 1).setFill(); NSRect(x: 0, y: 0, width: out.pixelsWide, height: out.pixelsHigh).fill()
        for (i, rep) in reps.enumerated() {
            rep.draw(in: NSRect(x: gap + i * (w + gap), y: gap + h - rep.pixelsHigh, width: rep.pixelsWide, height: rep.pixelsHigh))
        }
        NSGraphicsContext.restoreGraphicsState()
        try? out.representation(using: .png, properties: [:])?.write(to: url)
    }
}
