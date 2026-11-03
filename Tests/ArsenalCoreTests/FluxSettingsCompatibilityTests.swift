// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// FLUX moved from one `resolution` setting to free width and height. Callers and saved settings from before keep their
/// meaning: a caller's `resolution` is not overridden by the new defaults, a saved resolution becomes width and height,
/// and `resolution` stays writable as an alias.
struct FluxSettingsCompatibilityTests {
    var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    func metadata() throws -> JSONObject {
        let manifest = try TOMLParser.decode(
            ModelManifest.self, from: String(contentsOf: repo.appendingPathComponent("models/flux2-klein-4b/model.toml"), encoding: .utf8))
        return try WireJSON.decode(JSONEncoder().encode(manifest))
    }
    func settings(saved: JSONObject = [:]) throws -> ModelSettings {
        var s = ModelSettings()
        if !saved.isEmpty { s.load(["flux": saved]) }
        s.declare("flux", metadata: try metadata())
        return s
    }
    /// The runner's own reading (FluxWorker Runner.swift): resolution sets both sides, then width and height override.
    static func size(_ params: JSONObject) -> (Int, Int) {
        var w = 1024, h = 1024
        if let r = params["resolution"]?.string, let side = Int(r.split(separator: "x")[0]) { w = side; h = side }
        if let n = params["width"]?.number { w = Int(n) }
        if let n = params["height"]?.number { h = Int(n) }
        return (w, h)
    }

    @Test func aCallersResolutionKeepsItsMeaning() throws {
        let s = try settings()
        // The two probes from review.
        #expect(Self.size(s.request("flux", params: ["resolution": "512x512"])) == (512, 512))
        #expect(Self.size(s.request("flux", params: ["resolution": "768x768", "width": 1152])) == (1152, 768))
        // Plain cases: nothing sent gets the defaults; one side sent keeps the other's default.
        #expect(Self.size(s.request("flux", params: [:])) == (1024, 1024))
        #expect(Self.size(s.request("flux", params: ["width": 640])) == (640, 1024))
        #expect(Self.size(s.request("flux", params: ["width": 1344, "height": 768])) == (1344, 768))
        // A size pinned in the panel applies to calls without a size, and steps aside for a caller's resolution.
        let pinned = try settings(saved: ["width": 1344, "height": 768])
        #expect(Self.size(pinned.request("flux", params: [:])) == (1344, 768))
        #expect(Self.size(pinned.request("flux", params: ["resolution": "512x512"])) == (512, 512))
    }

    @Test func aSavedResolutionMovesToWidthAndHeight() throws {
        let migrated = try settings(saved: ["resolution": "512x512", "steps": 6])
        #expect(migrated.values["flux"] == ["width": 512, "height": 512, "steps": 6])
        #expect(Self.size(migrated.request("flux", params: [:])) == (512, 512))
        // Per target: a saved width wins, the missing height comes from the resolution.
        let partly = try settings(saved: ["resolution": "512x512", "width": 1152])
        #expect(partly.values["flux"] == ["width": 1152, "height": 512])
        let kept = try settings(saved: ["resolution": "512x512", "width": 1344, "height": 768])
        #expect(kept.values["flux"] == ["width": 1344, "height": 768])
        // An unknown value is dropped rather than kept as a dead key.
        #expect(try settings(saved: ["resolution": "640x640"]).values["flux"] == [:])
    }

    @Test func resolutionStaysWritableAsAnAlias() async throws {
        var s = try settings()
        try s.set("flux", key: "resolution", value: "768x768")
        #expect(s.values["flux"] == ["width": 768, "height": 768])
        #expect(throws: WorkerFailure.self) { try s.set("flux", key: "resolution", value: "640x640") }
        // Through the service the HTTP and MCP settings routes call.
        let service = StubModelsService(rows: [ModelRow(id: "flux", name: "FLUX", kind: "images")], metadata: ["flux": try metadata()])
        try await service.setSetting(id: "flux", key: "resolution", value: "512x512")
        let states = try await service.settings(id: "flux")
        #expect(states.first { $0.id == "width" }?.value == 512 && states.first { $0.id == "height" }?.value == 512)
        #expect(!states.contains { $0.id == "resolution" })
    }

    @Test func aliasesAreValidated() {
        let width = SettingDeclaration(
            id: "width", title: "Width", type: "number", defaultValue: .number(1024), min: 64, max: 2048, step: 16, applies: "request", mapsTo: "request.width")
        #expect(SettingAlias(id: "resolution", values: ["512x512": ["width": .number(512)]]).problems(settings: [width]).isEmpty)
        #expect(SettingAlias(id: "resolution", values: ["500x500": ["width": .number(500)]]).problems(settings: [width]).contains { $0.contains("does not accept") })
        #expect(SettingAlias(id: "resolution", values: ["512x512": ["depth": .number(512)]]).problems(settings: [width]).contains { $0.contains("not a declared setting") })
        #expect(SettingAlias(id: "width", values: ["a": ["width": .number(512)]]).problems(settings: [width]).contains { $0.contains("also a declared setting") })
    }

    /// A real service in an isolated support dir whose image model carries FLUX's settings and resolution alias.
    func fixture(saved: JSONObject?) async throws -> LocalCatalogFixture {
        let fluxSettings = try metadata()
        func toml(_ value: JSONValue) -> TOMLValue {
            switch value {
            case .string(let s): return .string(s)
            case .number(let n): return n == n.rounded() ? .integer(Int64(n)) : .decimal(n)
            case .bool(let b): return .bool(b)
            case .array(let a): return .array(a.map(toml))
            case .object(let o): return .table(o.mapValues(toml))
            case .null: return .string("")
            }
        }
        var config = ArsenalConfiguration()
        if let saved { config.settings = ["fixture-images": saved] }
        return try await makeLocalCatalogFixture(
            kinds: ["images"], configuration: config,
            extraTable: ["setting": toml(fluxSettings["setting"]!), "setting_alias": toml(fluxSettings["setting_alias"]!)])
    }
    func size(_ service: RealModelsService) async throws -> (Double?, Double?) {
        let states = try await service.settings(id: "fixture-images")
        return (states.first { $0.id == "width" }?.value.number, states.first { $0.id == "height" }?.value.number)
    }

    /// A config write merges what another process saved meanwhile: a legacy resolution saved externally is normalised
    /// on that path too (toggling Keep loaded used to leave the live size at 1024×1024 until a Refresh).
    @Test func anExternallySavedResolutionIsNormalisedByAMergedWrite() async throws {
        let f = try await fixture(saved: nil); defer { f.stop() }
        #expect(try await size(f.service) == (1024, 1024))
        let file = f.paths.support.appendingPathComponent("config.json")
        var external = try ArsenalConfiguration.read(file)
        external.settings["fixture-images"] = ["resolution": "512x512"]
        try external.write(file)
        try await f.service.setKeepHot(id: "fixture-images", on: true)
        #expect(try await size(f.service) == (512, 512))
        await f.service.shutdown()
    }

    /// The reviewer's sequence on a real service in an isolated support dir: a saved resolution survives launch,
    /// Refresh, a partial save and a restart, and the legacy key is gone once a write persists.
    @Test func aSavedResolutionSurvivesRefreshAPartialSaveAndRestart() async throws {
        let f = try await fixture(saved: ["resolution": "512x512"])
        defer { f.stop() }
        func saved() throws -> JSONObject? {
            try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).settings["fixture-images"]
        }
        // Launch, then Refresh before any save: still 512×512.
        #expect(try await size(f.service) == (512, 512))
        try await f.service.refresh()
        #expect(try await size(f.service) == (512, 512))
        // Save only the width, restart: the height stays from the old resolution, and the legacy key is gone on disk.
        try await f.service.setSetting(id: "fixture-images", key: "width", value: 1152)
        #expect(try saved() == ["width": 1152, "height": 512])
        await f.service.shutdown()
        let restarted = try await f.reopen(threshold: 8_000_000_000)
        #expect(try await size(restarted) == (1152, 512))
        try await restarted.refresh()
        #expect(try await size(restarted) == (1152, 512))
        await restarted.shutdown()
    }

    /// FLUX's settings as builds before width and height declared them (main at 8e97ac5): one resolution choice.
    static let olderMetadata: JSONObject = [
        "setting": [
            [
                "applies": "request", "choices": ["512x512", "768x768", "1024x1024"], "default": "1024x1024", "id": "resolution", "help": "Image size in pixels.",
                "maps_to": "request.resolution", "title": "Resolution", "type": "choice"
            ],
            ["applies": "request", "default": 4, "id": "steps", "maps_to": "request.steps", "max": 100, "min": 1, "step": 1, "title": "Steps", "type": "number"]
        ]
    ]

    /// A recovery file written before aliases existed (its settings have no `aliases` key) still decodes, with its
    /// saved values intact.
    @Test func anOlderRecoveryFileStillDecodes() throws {
        var old = ModelSettings()
        old.declare("flux", metadata: Self.olderMetadata)
        try old.set("flux", key: "resolution", value: "512x512")
        let revision = SavedRevision(
            digest: String(repeating: "a", count: 64), settings: old, load: ["dtype": "bf16"], preference: ModelPreferences(precision: "bf16"),
            resident: ResidentModel(id: "flux", residency: .manual), runner: "native", menus: [])
        // The same file as an older build wrote it: the settings object without `aliases`.
        var file = try WireJSON.decode(JSONEncoder().encode(revision))
        var settings = try #require(file["settings"]?.object)
        settings["aliases"] = nil; file["settings"] = .object(settings)
        #expect(file["settings"]?["aliases"] == nil && file["settings"]?["values"] != nil)
        let decoded = try JSONDecoder().decode(SavedRevision.self, from: WireJSON.encode(file))
        #expect(decoded.settings.values["flux"] == ["resolution": "512x512"] && decoded.settings.aliases.isEmpty)
        #expect(decoded.settings.request("flux", params: [:])["resolution"] == "512x512")
    }

    /// A FLUX model loaded on the older declarations (resolution) gets the user's size after a Refresh brings the new
    /// ones (width and height): request settings come from the live state, and the retired resolution it still
    /// declares is added when a resolution value equals the live size.
    @Test func aResidentOnOlderDeclarationsKeepsTheSizeAfterRefresh() throws {
        var resident = ModelSettings()
        resident.declare("flux", metadata: Self.olderMetadata)
        resident.load(["flux": ["resolution": "512x512"]])
        var live = ModelSettings()
        live.load(["flux": ["resolution": "512x512"]])
        live.declare("flux", metadata: try metadata())
        #expect(live.values["flux"] == ["width": 512, "height": 512])
        var request = resident.residentRequest("flux", params: ["prompt": "x"], live: live)
        #expect(Self.size(request) == (512, 512) && request["resolution"] == "512x512")
        try live.set("flux", key: "width", value: 768); try live.set("flux", key: "height", value: 768)
        request = resident.residentRequest("flux", params: [:], live: live)
        #expect(Self.size(request) == (768, 768) && request["resolution"] == "768x768")
        // A size no resolution expresses: the live width and height carry it, no resolution is added.
        try live.set("flux", key: "width", value: 1344)
        request = resident.residentRequest("flux", params: [:], live: live)
        #expect(Self.size(request) == (1344, 768) && request["resolution"] == nil)
        // The caller's resolution still wins over everything.
        #expect(Self.size(resident.residentRequest("flux", params: ["resolution": "512x512"], live: live)) == (512, 512))
    }
}
