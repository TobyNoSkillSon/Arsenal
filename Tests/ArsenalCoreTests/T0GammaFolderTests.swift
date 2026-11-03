// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

/// The t0-gamma kit folder, checked without running the model: its declarations, its embedded kind (identical to
/// t0-alpha's, so both folders install side by side), its input schema's bounds against the runner's, and its golden
/// (fixture hash, counts first, then the recorded port output against the maker reference within the stored tolerance).
struct T0GammaFolderTests {
    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
    private var folder: URL { repo.appendingPathComponent("models/t0-gamma") }
    /// The runner sources, bundled into the native worker since 2 Nov (Workers/Kit).
    private var sources: URL { repo.appendingPathComponent("Workers/Kit/Sources/KitT0Gamma") }

    private func json(_ path: String) throws -> [String: Any] {
        try #require(JSONSerialization.jsonObject(with: Data(contentsOf: folder.appendingPathComponent(path))) as? [String: Any])
    }

    @Test func manifestDeclaresTheModelAndItsOwnForecastOperation() throws {
        let manifest = try TOMLParser.decode(ModelManifest.self, from: String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
        #expect(manifest.id == "t0-gamma" && manifest.kind == "local.forecast" && manifest.kindFile == nil)
        #expect(manifest.params == "255.6M" && manifest.derive?["fp16"] == "cast:arsenal-kit-half-v1")
        #expect(manifest.weights.revision == "e08aee083778f1f32da4974c0a34276776a35c52" && manifest.weights.files == ["model.safetensors", "config.json"])
        #expect(manifest.weights.sha256?["model.safetensors"] == "1603e255629be22ea45518b502f16c55d06821c7090a1a851673e779e9c245a6")
        #expect(manifest.weights.bytes?["model.safetensors"] == 1_022_492_908)
        let ops = manifest.operations
        #expect(ops.map(\.name) == ["forecast"])
        #expect(ops.first?.input == "ops/forecast.input.json" && ops.first?.outputSchema == "ops/forecast.output.json")
        #expect(manifest.runners.map(\.name) == ["kit"])
        #expect(manifest.runners.map(\.worker) == ["kit-t0-gamma"])
        #expect(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).contains { $0.id == "local.forecast" })
    }

    /// Since 2 Nov the folder carries no kind tree: it uses the signed local.forecast (Resources/kinds), as every kit
    /// forecaster does, so the forecast folders can never disagree on it.
    @Test func folderUsesTheSignedForecastKind() throws {
        #expect(!FileManager.default.fileExists(atPath: folder.appendingPathComponent("kinds").path))
        #expect(FileManager.default.fileExists(atPath: repo.appendingPathComponent("Resources/kinds/local.forecast/ops/forecast.input.json").path))
    }

    @Test func inputSchemaBoundsMatchTheRunner() throws {
        let properties = try #require(try json("ops/forecast.input.json")["properties"] as? [String: [String: Any]])
        #expect(Set(properties.keys) == ["values", "horizon", "quantile_levels", "covariates", "past_covariates", "symmetric_averaging", "clip_nonnegative", "snap_integers"])
        // Request.swift: ForecastRequest.maxContext 8192 and maxCovariates 31, T0Config.maxHorizon 1024, Levels.maxRequested 99.
        #expect(properties["values"]?["maxItems"] as? Int == 8192)
        #expect(properties["horizon"]?["maximum"] as? Int == 1024 && properties["horizon"]?["minimum"] as? Int == 1)
        #expect(properties["covariates"]?["maxItems"] as? Int == 31 && properties["past_covariates"]?["maxItems"] as? Int == 31)
        #expect(properties["quantile_levels"]?["maxItems"] as? Int == 99)
        let request = try String(contentsOf: sources.appendingPathComponent("Request.swift"), encoding: .utf8)
        for line in ["static let maxContext = 8192", "static let maxCovariates = 31", "let maxHorizon = 1024", "static let maxRequested = 99"] {
            #expect(request.contains(line), "\(line)")
        }
    }

    /// Review fixes of 2 Nov: half-to-even snapping, float32 history tests, the median from the 0.5 level, and
    /// validation that agrees with the schema. Request.swift is compiled on its own with Tests/KitModels/t0-gamma/main.swift
    /// (Foundation only: no MLX, no weights) and run.
    @Test func requestLogicFollowsTheMakerAndTheSchema() throws {
        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("t0-gamma-request-\(UUID().uuidString)")
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
            repo.appendingPathComponent("Tests/KitModels/t0-gamma/main.swift").path
        ]
        let compile = try run("/usr/bin/xcrun", ["swiftc", "-Onone"] + files + ["-o", binary])
        try #require(compile.0 == 0, "\(compile.1)")
        let result = try run(binary, [])
        #expect(result.0 == 0 && result.1.hasPrefix("ok 36"), "\(result.1)")
    }

    /// Release review (2 Nov): the pass's output zeroes only NaN, as the maker's sanitize does; an overflow is no longer
    /// turned into a confident forecast of 0, and Wire refuses it as non-finite instead.
    @Test func forecastZeroesOnlyNaN() throws {
        let model = try String(contentsOf: sources.appendingPathComponent("Model.swift"), encoding: .utf8)
        #expect(model.contains("query: rollout).map { $0.isNaN ? 0 : $0 }"))
        #expect(!model.contains("isFinite ? $0 : 0"))
    }

    @Test func goldenHoldsTheFixtureHashAndAgreesWithTheMakerReference() throws {
        let golden = try json("gate/golden/golden.json")
        let fixture = try Data(contentsOf: folder.appendingPathComponent("gate/golden/fixtures.json"))
        #expect(golden["fixture_sha256"] as? String == FileDigest.sha256(fixture))
        let fixtures = try #require(JSONSerialization.jsonObject(with: fixture) as? [String: [String: Any]])
        let cases = try #require(golden["cases"] as? [String: [String: Any]])
        let tolerance = try #require(golden["tolerance"] as? [String: Any]).compactMapValues { ($0 as? NSNumber)?.doubleValue }
        #expect(Set(cases.keys) == Set(fixtures.keys) && cases.count == 5)
        for (name, entry) in cases {
            let request = try #require(fixtures[name])
            let horizon = try #require(request["horizon"] as? Int)
            let rows = 1 + ((request["past_covariates"] as? [Any])?.count ?? 0)
            let port = try #require(entry["port_median"] as? [[Double]]), maker = try #require(entry["maker_median"] as? [[Double]])
            let quantiles = try #require(entry["maker_quantiles"] as? [[[Double]]]), levels = try #require(entry["quantile_levels"] as? [Double])
            // Counts before values.
            #expect(port.count == rows && maker.count == rows && quantiles.count == rows, "\(name)")
            #expect(port.allSatisfy { $0.count == horizon } && maker.allSatisfy { $0.count == horizon }, "\(name)")
            #expect(quantiles.allSatisfy { $0.count == horizon && $0.allSatisfy { $0.count == levels.count } }, "\(name)")
            #expect(levels == (request["quantile_levels"] as? [Double] ?? levels) && (request["quantile_levels"] != nil || levels.count == 21), "\(name)")
            let scale = quantiles.flatMap { $0.flatMap { $0 } }.map(abs).max() ?? 1
            let slack = (entry["snap_integers"] as? Bool == true) ? tolerance["snap_integers_extra_abs"] ?? 0 : 0
            let allowed = (tolerance["max_abs_fraction_of_ref_max"] ?? 0) * scale + slack + 1e-4
            for (a, b) in zip(port.joined(), maker.joined()) { #expect(abs(a - b) <= allowed, "\(name): \(a) vs \(b)") }
            let measured = try #require(entry["measured"] as? [String: Any]).compactMapValues { ($0 as? NSNumber)?.doubleValue }
            #expect((measured["cosine"] ?? 0) >= (tolerance["cosine_min"] ?? 1) && (measured["rel_l2"] ?? 1) <= (tolerance["rel_l2_max"] ?? 0), "\(name)")
        }
        let components = try #require(golden["component_worst_rel_l2"] as? [String: Any]).compactMapValues { ($0 as? NSNumber)?.doubleValue }
        #expect(components.count == 7 && components.values.allSatisfy { $0 <= tolerance["component_rel_l2_max"] ?? 0 })
    }
}
