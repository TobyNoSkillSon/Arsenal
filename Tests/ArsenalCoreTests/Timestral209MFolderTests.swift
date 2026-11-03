// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The Timestral 209M kit folder, checked without running the model: its declarations and licence terms, its embedded
/// forecast kind (byte-identical to t0-gamma's, so both install side by side), its input schema's bounds against the
/// runner's, the host-side maths against the maker's own values, and its golden (fixture hash, counts first, then the
/// recorded port output against the maker reference within the stored tolerance).
struct Timestral209MFolderTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var folder: URL { repo.appendingPathComponent("models/timestral-209m") }
    /// The runner sources, bundled into the native worker since 2 Nov (Workers/Kit).
    private var sources: URL { repo.appendingPathComponent("Workers/Kit/Sources/KitTimestral209m") }

    private func json(_ path: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent(path))) as? [String: Any])
    }

    @Test func manifestDeclaresTheModelItsTermsAndItsForecastOperation() throws {
        let text = try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8)
        let manifest = try TOMLParser.decode(ModelManifest.self, from: text)
        #expect(manifest.id == "timestral-209m" && manifest.kind == "local.forecast" && manifest.kindFile == nil)
        #expect(manifest.params == "209.3M" && manifest.derive?["fp16"] == "cast:arsenal-kit-half-v1")
        #expect(manifest.weights.revision == "9874d252ab0aae16f7f7c999f98f98d6ad7d8365" && manifest.weights.files == ["model.safetensors", "config.json"])
        #expect(manifest.weights.sha256?["model.safetensors"] == "1456effb9927e3daa8c7234f10497e3424233869245a9c72b8ecebaf7ad2b7fa")
        #expect(manifest.weights.bytes?["model.safetensors"] == 837_271_648)
        #expect(manifest.operations.map(\.name) == ["forecast"] && manifest.runners.map(\.name) == ["kit"])
        #expect(manifest.runners.map(\.worker) == ["kit-timestral-209m"])
        #expect(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).contains { $0.id == "local.forecast" })
        // Revenue-capped licence: shown amber, as d1-3B.
        #expect(text.contains("commercial = false") && text.contains("US$20 million"))
        let licence = try String(contentsOf: folder.appendingPathComponent("licence/LICENSE"), encoding: .utf8)
        #expect(licence.contains("$20 million") && licence.contains("2026 - Mistral AI"))
    }

    /// Since 2 Nov the folder carries no kind tree: it uses the signed local.forecast (Resources/kinds), as every kit
    /// forecaster does, so the forecast folders can never disagree on it.
    @Test func folderUsesTheSignedForecastKind() throws {
        #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent("kinds").path))
        #expect(FileManager.default.fileExists(atPath: repo.appendingPathComponent("Resources/kinds/local.forecast/ops/forecast.input.json").path))
    }

    @Test func inputSchemaBoundsMatchTheRunner() throws {
        let properties = try #require(try json("ops/forecast.input.json")["properties"] as? [String: [String: Any]])
        #expect(
            Set(properties.keys) == [
                "values", "horizon", "quantile_levels", "co_targets", "past_covariates", "covariates", "categorical_past_covariates",
                "categorical_covariates", "symmetric_averaging", "clip_nonnegative"
            ])
        #expect(properties["values"]?["maxItems"] as? Int == 32768)
        #expect(properties["horizon"]?["maximum"] as? Int == 4096 && properties["horizon"]?["minimum"] as? Int == 1)
        for key in ["co_targets", "past_covariates", "covariates", "categorical_past_covariates", "categorical_covariates"] {
            #expect(properties[key]?["maxItems"] as? Int == 31, "\(key)")
        }
        #expect(properties["symmetric_averaging"]?["default"] as? Bool == true && properties["clip_nonnegative"]?["default"] as? Bool == true)
        let levels = try #require((properties["quantile_levels"]?["items"] as? [String: Any])?["enum"] as? [Double])
        #expect(levels.count == 21 && levels.first == 0.01 && levels.last == 0.99 && levels.contains(0.5))
        let request = try String(contentsOf: sources.appendingPathComponent("Request.swift"), encoding: .utf8)
        for line in ["static let maxContext = 32768", "static let maxHorizon = 4096", "static let maxRows = 32", "list.count <= 31"] {
            #expect(request.contains(line), "\(line)")
        }
    }

    /// Request.swift compiled on its own with Tests/KitModels/timestral-209m/main.swift (Foundation only: no MLX, no
    /// weights): the maker's host values, validation and the forecaster's block layout around a fake network.
    @Test func hostLogicMatchesTheMaker() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("timestral-host-\(UUID().uuidString)")
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        func run(_ tool: String, _ arguments: [String]) throws -> (Int32, String) {
            let process = Process()
            let pipe = Pipe()
            process.executableURL = URL(fileURLWithPath: tool)
            process.arguments = arguments
            process.standardOutput = pipe
            process.standardError = pipe
            try process.run()
            let output = pipe.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            return (process.terminationStatus, String(decoding: output, as: UTF8.self))
        }
        let binary = temporary.appendingPathComponent("check").path
        let files = [
            sources.appendingPathComponent("Request.swift").path,
            repo.appendingPathComponent("Tests/KitModels/timestral-209m/main.swift").path
        ]
        let compile = try run("/usr/bin/xcrun", ["swiftc", "-Onone"] + files + ["-o", binary])
        try #require(compile.0 == 0, "\(compile.1)")
        let result = try run(binary, [])
        #expect(result.0 == 0 && result.1.hasPrefix("ok 50"), "\(result.1)")
    }

    @Test func goldenHoldsTheFixtureHashAndAgreesWithTheMakerReference() throws {
        let golden = try json("gate/golden/golden.json")
        let fixture = try Data(contentsOf: folder.appendingPathComponent("gate/golden/fixtures.json"))
        #expect(golden["fixture_sha256"] as? String == FileDigest.sha256(fixture))
        let fixtures = try #require(JSONSerialization.jsonObject(with: fixture) as? [String: [String: Any]])
        let cases = try #require(golden["cases"] as? [String: [String: Any]])
        let tolerance = try #require(golden["tolerance"] as? [String: Any]).compactMapValues { ($0 as? NSNumber)?.doubleValue }
        #expect(Set(cases.keys) == Set(fixtures.keys) && cases.count == 6)
        for (name, entry) in cases {
            let request = try #require(fixtures[name])
            let horizon = try #require(request["horizon"] as? Int)
            let rows = 1 + ((request["co_targets"] as? [Any])?.count ?? 0)
            let kept = try #require(entry["steps_kept"] as? Int)
            let port = try #require(entry["port_median"] as? [[Double]]), maker = try #require(entry["maker_median"] as? [[Double]])
            let quantiles = try #require(entry["maker_quantiles"] as? [[[Double]]]), levels = try #require(entry["quantile_levels"] as? [Double])
            // Counts before values.
            #expect(entry["horizon"] as? Int == horizon && kept == min(horizon, 48), "\(name)")
            #expect(port.count == rows && maker.count == rows && quantiles.count == rows, "\(name)")
            #expect(port.allSatisfy { $0.count == kept } && maker.allSatisfy { $0.count == kept }, "\(name)")
            #expect(quantiles.allSatisfy { $0.count == kept && $0.allSatisfy { $0.count == levels.count } }, "\(name)")
            #expect(levels == (request["quantile_levels"] as? [Double] ?? levels) && (request["quantile_levels"] != nil || levels.count == 21), "\(name)")
            let scale = quantiles.flatMap { $0.flatMap { $0 } }.map(abs).max() ?? 1
            let allowed = (tolerance["max_abs_fraction_of_ref_max"] ?? 0) * scale + 1e-3
            for (a, b) in zip(port.joined(), maker.joined()) { #expect(abs(a - b) <= allowed, "\(name): \(a) vs \(b)") }
            let measured = try #require(entry["measured"] as? [String: Any]).compactMapValues { ($0 as? NSNumber)?.doubleValue }
            #expect((measured["cosine"] ?? 0) >= (tolerance["cosine_min"] ?? 1) && (measured["rel_l2"] ?? 1) <= (tolerance["rel_l2_max"] ?? 0), "\(name)")
            #expect((measured["gap_fraction"] ?? 1) <= (tolerance["max_abs_fraction_of_ref_max"] ?? 0), "\(name)")
            #expect((entry["component_worst_rel_l2"] as? Double ?? 1) <= (tolerance["component_rel_l2_max"] ?? 0), "\(name)")
        }
    }
}
