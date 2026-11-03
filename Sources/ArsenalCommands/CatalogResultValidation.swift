// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalBench

/// Dispatch on the document, never its filename. Unversioned intake records predate schema-2 table stats.
enum CatalogResultValidation {
    static func validate(data: Data, path: String, folder: URL, manifest: ModelManifest) throws {
        let root = try WireJSON.decode(data)
        try finite(.object(root))
        guard manifest.id == folder.lastPathComponent, root["model"]?.string == manifest.id else { throw invalid("result model identity mismatch") }
        switch root["schema"] {
        case .number(1):
            let result = try ResultValidation.validate(data: data)
            guard result.kind == manifest.kind else { throw invalid("result kind identity mismatch") }
        case .number(2):
            try table(data: data, root: root, folder: folder, manifest: manifest)
        case .string(QuickCheckEvidence.schema):
            try quickCheck(root, manifest: manifest)
        case nil:
            try intake(root, path: path, folder: folder, manifest: manifest)
        default: throw invalid("unknown result schema")
        }
    }

    private static func table(data: Data, root: JSONObject, folder: URL, manifest: ModelManifest) throws {
        try fields(root, required: ["schema", "model", "mode", "stats"])
        let result = try PrecisionStatsResult.decode(data)
        guard result.stats.kind.rawValue == manifest.kind else { throw invalid("stat kind identity mismatch") }
        let stats = try object(root, "stats")
        try fields(stats, required: ["kind", "dtype", "runner", "availability", "metrics"], optional: ["qualityCheckPassed", "qualityOnlyUnavailable", "qualityIdentity"])
        if stats["qualityIdentity"] != nil { try digest(stats, "qualityIdentity", count: 64) }
        var validatedEvidence = Set<String>()
        let cells = try object(stats, "metrics").sorted { $0.key < $1.key } + [("availability", stats["availability"]!)]
        for (name, value) in cells {
            guard let cell = value.object else { throw invalid("stat must be an object") }
            try fields(cell, required: ["state"], optional: ["value", "reason", "basis", "provenance"])
            if let raw = cell["provenance"], raw != .null {
                guard let provenance = raw.object else { throw invalid("stat provenance must be an object") }
                try fields(provenance, required: ["chip", "date", "method"], optional: ["gpuCores", "baseline", "source", "revision", "run", "defaultWorkload"])
                // Persisted table figures must resolve to the run that supplied this precision's evidence.
                let run = try text(provenance, "run"), source = try text(provenance, "source")
                let sourcePath = String(source.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)[0]).trimmingCharacters(in: .whitespaces)
                // SafePath validates even absent sources. Only the current bound intake may be archived.
                let sourceURL = try SafePath.resolve(sourcePath, inside: folder)
                if !FileManager.default.fileExists(atPath: sourceURL.path) {
                    try binding(data: data, result: result, source: sourcePath, run: run, folder: folder, manifest: manifest)
                    continue
                }
                let evidence = try read(sourcePath, folder: folder)
                if evidence["schema"]?.string == QuickCheckEvidence.schema {
                    try quickCheckCell(name: name, cell: cell, run: run, evidence: evidence, stats: stats, set: result.stats, manifest: manifest)
                    if validatedEvidence.insert(sourcePath).inserted { try quickCheck(evidence, manifest: manifest) }
                    continue
                }
                guard evidence["model"]?.string == manifest.id, let evidenceRun = evidence["run"]?.string,
                    run == evidenceRun + "-" + result.stats.dtype,
                    evidence["rows"]?.array?.contains(where: { $0["precision"]?.string == result.stats.dtype }) == true
                else { throw invalid("stat run/evidence identity mismatch") }
                if let verdict = stats["qualityCheckPassed"] {
                    guard let row = evidence["rows"]?.array?.first(where: { $0["precision"]?.string == result.stats.dtype }), verdict == row["qualified"] else {
                        throw invalid("table quality verdict differs from its exact intake row")
                    }
                }
                if validatedEvidence.insert(sourcePath).inserted { try intake(evidence, path: sourcePath, folder: folder, manifest: manifest) }
            } else if ["measured", "quickCheck"].contains(cell["state"]?.string ?? "") {
                throw invalid("recorded stat needs run evidence")
            }
        }
    }

    /// A quick-check record: the measuring script's rows for this model's precisions, at the folder's revision.
    private static func quickCheck(_ root: JSONObject, manifest: ModelManifest) throws {
        try fields(root, required: ["schema", "model", "run", "revision", "chip", "gpu_cores", "method", "rows"], optional: ["source", "note"])
        guard root["model"]?.string == manifest.id, root["revision"]?.string == manifest.weights.revision, validRun(try text(root, "run")),
            let cores = root["gpu_cores"]?.number, cores > 0, cores.rounded() == cores, let rows = root["rows"]?.array, !rows.isEmpty
        else { throw invalid("invalid quick-check identity") }
        _ = try text(root, "chip"); _ = try text(root, "method")
        var seen = Set<String>()
        for entry in rows {
            let row = try requireObject(entry), precision = try text(row, "precision")
            guard manifest.precisions.contains(precision), seen.insert(precision).inserted, ISO8601DateFormatter().date(from: try text(row, "date")) != nil,
                let peak = row["peak_footprint_mb"]?.number, peak > 0, row["error"] == nil
            else { throw invalid("invalid quick-check row") }
            let ops = try object(row, "ops")
            for op in ops.values {
                let op = try requireObject(op)
                try nonnegative(op, "seconds_warm")
                guard op["status"]?.number != nil else { throw invalid("quick-check operation needs its status") }
            }
            guard ops.values.contains(where: { $0["status"]?.number == 200 }) else { throw invalid("quick-check row has no completed request") }
        }
    }

    /// A table cell bound to a quick-check record is a quick check of peak memory or request time, equal to its row.
    private static func quickCheckCell(
        name: String, cell: JSONObject, run: String, evidence: JSONObject, stats: JSONObject, set: PrecisionStatSet, manifest: ModelManifest
    ) throws {
        let dtype = set.dtype
        // The kind's own speed figure when it is one request's time: secondsPerRequest, or secondsPerImage for an image
        // model whose operation is not generation (an upscaler, background removal).
        let requestTime = [StatMetric.secondsPerRequest, .secondsPerImage].contains(set.speed.metric) ? set.speed.metric.rawValue : nil
        guard evidence["model"]?.string == manifest.id, let evidenceRun = evidence["run"]?.string, run == evidenceRun + "-" + dtype,
            let row = evidence["rows"]?.array?.first(where: { $0["precision"]?.string == dtype })?.object
        else { throw invalid("stat run/quick-check identity mismatch") }
        guard cell["state"]?.string == StatState.quickCheck.rawValue, stats["qualityCheckPassed"] == nil else {
            throw invalid("a quick-check record supplies quick checks only, with no quality verdict")
        }
        let value = cell["value"]?.number
        func equal(_ recorded: Double?) -> Bool {
            guard let recorded, let value else { return false }
            return abs(value - recorded) <= 1e-9 * max(1, abs(recorded))
        }
        switch name {
        case "availability": guard value == nil else { throw invalid("availability carries no value") }
        case StatMetric.peakMemoryGB.rawValue:
            guard equal(row["peak_footprint_mb"]?.number.map { QuickCheckEvidence.gigabytes(mebibytes: $0) }) else { throw invalid("peak memory differs from its quick-check row") }
        case let metric where metric == requestTime:
            let completed = row["ops"]?.object?.values.filter { $0["status"]?.number == 200 } ?? []
            guard completed.contains(where: { equal($0["seconds_warm"]?.number) }) else { throw invalid("request time differs from its quick-check row") }
        default: throw invalid("a quick-check record supplies only peak memory and the kind's request time")
        }
    }

    /// Compact bindings retain the complete frozen numerical qualification. A fresh clone
    /// validates those verdicts and exact table hashes; raw captures are optional local evidence.
    private static func binding(data: Data, result: PrecisionStatsResult, source: String, run: String, folder: URL, manifest: ModelManifest) throws {
        let directory = folder.appendingPathComponent("gate/evidence")
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.lastPathComponent.hasPrefix("quality-binding-") && $0.pathExtension == "json" }.sorted { $0.path < $1.path }
        guard let file = files.last else { throw invalid("missing current quality binding") }
        let root = try WireJSON.decode(Data(contentsOf: file))
        guard root["schema"] == 3, root["qualification_policy"]?.string == IntakeQualification.policyID,
            let captureRun = root["run"]?.string, validRun(captureRun), run == captureRun + "-" + result.stats.dtype,
            source == "results/intake/" + captureRun + ".json",
            let outcomes = root["outcomes"]?.array, coversDeclaredTiers(outcomes.map { $0["dtype"]?.string }, manifest: manifest),
            let outcome = outcomes.first(where: { $0["dtype"]?.string == result.stats.dtype }),
            outcome["result_sha256"]?.string == FileDigest.sha256(data),
            outcome["quality_identity"]?.string == result.stats.qualityIdentity,
            outcome["passed"]?.bool == result.stats.qualityCheckPassed,
            outcome["recipe"]?.string == (manifest.derive?[result.stats.dtype] ?? "vendor:" + result.stats.dtype)
        else { throw invalid("table differs from its current quality binding") }
        for outcome in outcomes {
            guard let dtype = outcome["dtype"]?.string, let q = outcome["qualification"]?.object else { throw invalid("missing bound qualification") }
            let row: JSONObject = [
                "precision": .string(dtype), "qualified": outcome["passed"] ?? .null, "qualification": .object(q),
                "selection_gate": q["selection_gate"] ?? .null, "gate": q["heldout_gate"] ?? q["selection_gate"] ?? .null
            ]
            _ = try IntakeQualification.validatedRow(row, model: manifest)
            for key in ["capture_result_sha256", "quality_identity", "result_sha256", "bounds_sha256"] {
                guard let value = outcome[key]?.string, ModelManifest.isHex(value, count: 64) else { throw invalid("invalid bound evidence digest") }
            }
        }
    }

    private static func intake(_ root: JSONObject, path: String, folder: URL, manifest: ModelManifest) throws {
        if manifest.id == "birefnet" { try backgroundIntake(root, path: path, folder: folder, manifest: manifest); return }
        try fields(
            root, required: ["model", "revision", "date", "chip", "gpu_cores", "worker_sha256", "policy", "rows", "fp16_selected", "network_bytes", "note"],
            optional: [
                "kind", "run", "capture_run", "fp16_component_passed", "reference_audio_sha256", "resource_receipt", "implementation_identity", "load_identity", "selected_half",
                "worker_build", "recipes", "reconciliation"
            ])
        guard root["model"]?.string == manifest.id, root["fp16_selected"]?.bool != nil,
            let cores = root["gpu_cores"]?.number, cores > 0, cores.rounded() == cores,
            let network = root["network_bytes"]?.number, network >= 0,
            let rows = root["rows"]?.array, !rows.isEmpty
        else { throw invalid("invalid intake identity or required values") }
        let date = try text(root, "date")
        guard ISO8601DateFormatter().date(from: date) != nil else { throw invalid("invalid intake date") }
        _ = try text(root, "chip"); _ = try text(root, "note")
        try digest(root, "revision", count: 40); try digest(root, "worker_sha256", count: 64)
        if root["reference_audio_sha256"] != nil { try digest(root, "reference_audio_sha256", count: 64) }
        if root["implementation_identity"] != nil || root["load_identity"] != nil {
            try digest(root, "implementation_identity", count: 64)
            let hasFolderRecipe =
                FileManager.default.fileExists(atPath: folder.appendingPathComponent("kernels/manifest.toml").path)
                || manifest.settings.contains { $0.applies == "load" }
            if root["load_identity"] != .string("") || hasFolderRecipe { try digest(root, "load_identity", count: 64) }
        }
        if root["fp16_component_passed"] != nil && root["fp16_component_passed"]?.bool == nil { throw invalid("invalid component pass flag") }
        let policy = try object(root, "policy"), fixture = try object(policy, "fixture")
        // Historical intake has no kind field: its typed workload declares images or sound.
        let kind =
            fixture["width"]?.number != nil && fixture["height"]?.number != nil
            ? "images"
            : fixture["duration"]?.number != nil
                ? "sound"
                : fixture["questions"]?.object != nil && fixture["state"] != nil
                    ? "decisions"
                    : fixture["text"]?.string != nil ? "voices" : fixture["audio"]?["path"]?.string != nil ? "enhancement" : nil
        guard kind == manifest.kind, root["kind"] == nil || root["kind"]?.string == manifest.kind,
            policy["independent_reference_qualified"]?.bool != nil
        else { throw invalid("intake kind/workload identity mismatch") }
        if kind == "decisions" {
            guard manifest.kind == "decisions", let questions = fixture["questions"]?.object, !questions.isEmpty else { throw invalid("invalid decision workload") }
        } else if kind == "voices" {
            _ = try text(fixture, "text"); try nonnegative(fixture, "seed")
            guard fixture["stream"]?.bool == true else { throw invalid("voice intake must use the default streamed workload") }
        } else if kind == "enhancement" {
            _ = try text(try object(fixture, "audio"), "path")
            guard fixture["stream"]?.bool == false else { throw invalid("enhancement intake requires usable-file delivery") }
        } else {
            _ = try text(fixture, "prompt")
            try nonnegative(fixture, "steps"); try nonnegative(fixture, "seed")
            for key in kind == "images" ? ["width", "height"] : ["duration"] { try nonnegative(fixture, key) }
        }
        let run = root["run"]?.string
        if root["run"] != nil {
            guard let run, validRun(run), path == "results/intake/" + run + ".json", policy["run"]?.string == run else {
                throw invalid("intake run identity mismatch")
            }
            let evidencePath = "gate/evidence/intake/" + run
            _ = try text(policy, "reference_scope"); _ = try text(policy, "timing_order")
            if policy["gate"] != nil {
                _ = try text(policy, "gate")
            } else {
                _ = try text(policy, "format_gate"); _ = try text(policy, "quantized_tier_gate")
            }
            let receipt = try read(evidencePath + "/receipt.json", folder: folder)
            guard receipt["run"]?.string == run, receipt["results"]?.string == path else { throw invalid("intake receipt identity mismatch") }
            _ = try existing(try text(receipt, "results"), folder: folder)
            if let capture = root["capture_run"] {
                guard let captureRun = capture.string, validRun(captureRun), receipt["capture_run"] == capture else { throw invalid("invalid intake capture run") }
                let captured = try read("results/intake/" + captureRun + ".json", folder: folder)
                guard captured["run"] == capture, captured["model"] == root["model"] else { throw invalid("capture identity mismatch") }
            } else {
                guard receipt["evidence"]?.string == evidencePath else { throw invalid("intake evidence link mismatch") }
                _ = try existing(try text(receipt, "evidence"), folder: folder, directory: true)
            }
        } else {
            // First-generation screenings link each row to its retained snapshot receipt by capture date.
            _ = try text(policy, "source")
            for key in ["image_min_psnr_db", "audio_max_spectral_relative_l2", "audio_max_waveform_relative_l2"] { try nonnegative(policy, key) }
        }
        if root["recipes"] != nil {
            guard kind == "voices", let run else { throw invalid("component recipes need a voice intake run") }
            try voiceRecipes(root, folder: folder, run: run)
        }
        let current = try currentIntake(path, folder: folder)
        var precisions = Set<String>()
        for value in rows {
            guard let row = value.object else { throw invalid("intake row must be an object") }
            if row["error"] != nil {
                try fields(row, required: ["error"]); _ = try text(row, "error"); continue
            }
            try fields(
                row, required: ["precision", "qualified", "state", "peak_gb", "load"],
                optional: [
                    "cold_seconds", "fixture_sha256", "sanity_reply", "warm", "paired_baseline_warm", "baseline_drift_fraction", "gate", "capture_run",
                    "machine_in_use", "quality_metric", "reference", "relative_gate_passed", "reply", "rtf", "seconds", "waveform_relative_l2", "quality_reply",
                    "offered", "precisionQualityWarnings", "end_to_end", "warm_rtf", "paired_baseline_rtf", "capture_load", "serving_worker", "selection_gate", "qualification"
                ])
            let precision = try text(row, "precision")
            guard ["bf16", "fp16", "int8", "int4"].contains(precision), precisions.insert(precision).inserted,
                row["qualified"]?.bool != nil, row["state"]?.string == "quickCheck"
            else { throw invalid("invalid intake row identity/state") }
            if row["offered"] != nil || row["precisionQualityWarnings"] != nil {
                guard row["offered"]?.bool == (precision == "int4" || row["qualified"]?.bool == true),
                    row["precisionQualityWarnings"]?.object
                        == (precision == "int4" && row["qualified"]?.bool != true ? ["int4": .string(PrecisionStatSet.int4QualityWarning)] : [:])
                else { throw invalid("intake offering/quality warning mismatch") }
            }
            if kind == "voices" {
                let ratio = try object(row, "warm_rtf"), diagnostic = try object(row, "end_to_end")
                guard ratio["unit"]?.string == "compute s/audio s", let samples = ratio["samples"]?.array, !samples.isEmpty,
                    samples.allSatisfy({ ($0.number ?? -1) > 0 }), (ratio["median"]?.number ?? 0) > 0,
                    diagnostic["diagnostic_only"]?.bool == true, diagnostic["affects_precision_gate"]?.bool == false,
                    (diagnostic["candidate_samples"]?.number ?? 0) > 0
                else { throw invalid("invalid voice timing/sanity evidence") }
            }
            if row["serving_worker"] != nil || row["capture_load"] != nil { try servingWorker(row) }
            try nonnegative(row, "peak_gb")
            let load = try object(row, "load")
            guard load["ok"]?.bool == true else { throw invalid("intake load must succeed") }
            try nonnegative(load, "load_s")
            if let run {
                try nonnegative(row, "cold_seconds"); try digest(row, "fixture_sha256", count: 64)
                _ = try object(row, "sanity_reply")
                try warm(try object(row, "warm"))
                if row["paired_baseline_warm"] != nil { try warm(try object(row, "paired_baseline_warm")) }
                let gate = try object(row, "gate")
                let components = try object(gate, "components")
                guard gate["component_gate_passed"]?.bool != nil, !components.isEmpty else { throw invalid("invalid intake component gate") }
                for component in components.values {
                    guard let entry = component.object, entry["finite"]?.bool != nil, entry["passed"]?.bool != nil, entry["activation_passed"]?.bool != nil else {
                        throw invalid("invalid intake component")
                    }
                    try nonnegative(entry, "relative_l2"); try nonnegative(entry, "clamp_count")
                    _ = try object(entry, "bounds")
                }
                try digest(gate, "bounds_sha256", count: 64)
                let base = "gate/evidence/intake/" + run + "/"
                if current || row["qualification"] != nil || row["selection_gate"] != nil || policy["selection_component_keys"] != nil {
                    try qualification(row, folder: folder, base: base)
                } else {
                    let candidates = ["bounds.json", "quantized-bounds.json"].compactMap { try? existing(base + $0, folder: folder) }
                    guard try candidates.contains(where: { try FileDigest.sha256($0) == gate["bounds_sha256"]?.string }) else { throw invalid("intake bounds evidence mismatch") }
                }
            } else {
                try nonnegative(row, "seconds"); try nonnegative(row, "quality_metric"); try nonnegative(row, "waveform_relative_l2")
                guard row["relative_gate_passed"]?.bool != nil, row["machine_in_use"]?.bool != nil else { throw invalid("invalid screening gate") }
                _ = try text(row, "reference"); _ = try object(row, "reply")
                let snapshotPath = "gate/evidence/intake-" + String(date.prefix(10)) + "-" + precision + "-snapshot.json"
                if FileManager.default.fileExists(atPath: folder.appendingPathComponent(snapshotPath).path) {
                    let snapshot = try read(snapshotPath, folder: folder)
                    guard snapshot["model"]?.string == manifest.id, snapshot["precision"]?.string == precision else { throw invalid("screening snapshot identity mismatch") }
                }
            }
        }
        try finite(.object(root))
    }

    /// Background removal has source images and masks rather than a generation prompt/step count.
    private static func backgroundIntake(_ root: JSONObject, path: String, folder: URL, manifest: ModelManifest) throws {
        try fields(
            root,
            required: [
                "model", "revision", "date", "chip", "gpu_cores", "worker_sha256", "policy", "rows", "run", "network_bytes", "resource_receipt", "implementation_identity",
                "load_identity"
            ], optional: ["capture_run", "note"])
        let run = try text(root, "run"), policy = try object(root, "policy")
        guard validRun(run), path == "results/intake/" + run + ".json", policy["run"]?.string == run,
            root["revision"]?.string == manifest.weights.revision, manifest.kind == "images", manifest.basePrecision == "fp16",
            root["network_bytes"] == 0, policy["searched"] == false, policy["machine_in_use"] == true,
            ISO8601DateFormatter().date(from: try text(root, "date")) != nil,
            let cores = root["gpu_cores"]?.number, cores > 0, cores.rounded() == cores,
            let rows = root["rows"]?.array, rows.count == 3, coversDeclaredTiers(rows.map { $0["precision"]?.string }, manifest: manifest)
        else { throw invalid("invalid background intake identity") }
        _ = try text(root, "chip"); _ = try text(policy, "reference_scope"); _ = try text(policy, "timing_order")
        for key in ["worker_sha256", "implementation_identity"] { try digest(root, key, count: 64) }
        if root["load_identity"] != .string("") { try digest(root, "load_identity", count: 64) }
        let expected = Set(
            ["patch-0", "patch-1", "squeeze", "decoder"]
                + (0..<2).flatMap { c in
                    (0..<4).flatMap { ["stage-\(c)-\($0)/out", "stage-\(c)-\($0)/down"] }
                })
        guard Set(policy["component_keys"]?.array?.compactMap(\.string) ?? []) == expected else { throw invalid("background component coverage differs") }
        let diagnostic = try object(policy, "mask_diagnostic")
        guard diagnostic["bound"] == 0.02, diagnostic["affects_precision_gate"] == false, diagnostic["each_image_must_pass"] == true,
            let fixtures = diagnostic["fixtures"]?.array, fixtures.count == 3,
            Set(fixtures.compactMap { $0["name"]?.string }) == Set((0..<3).map { "birefnet-fixed-\($0).png" })
        else { throw invalid("invalid frozen mask diagnostic") }
        for fixture in fixtures { try digest(try requireObject(fixture), "sha256", count: 64) }
        let resources = try object(root, "resource_receipt")
        guard resources["stopped"] == false, resources["initial_swap_bytes"] == resources["peak_swap_bytes"],
            let elapsed = resources["elapsed_s"]?.number, elapsed > 0, elapsed < 590,
            let budget = resources["worker_budget_bytes"]?.number, budget > 0,
            let peak = resources["worker_peak_bytes"]?.number, peak > 0, peak <= budget
        else { throw invalid("background intake exceeded its resource guard") }
        let base = "gate/evidence/intake/" + run + "/"
        for value in rows {
            let row = try requireObject(value), dtype = try text(row, "precision")
            try fields(
                row,
                required: [
                    "precision", "qualified", "state", "gate", "selection_gate", "qualification", "load", "capture_load", "serving_worker", "peak_gb", "warm",
                    "first_request_seconds"
                ], optional: ["paired_baseline_warm", "baseline_drift_fraction", "end_to_end", "offered", "precisionQualityWarnings", "fixture_sha256", "capture_run"])
            guard row["state"] == "quickCheck", row["qualified"]?.bool != nil,
                Set(try object(try object(row, "gate"), "components").keys) == expected,
                let gb = row["peak_gb"]?.number, gb > 0
            else { throw invalid("invalid background intake row") }
            try nonnegative(row, "first_request_seconds"); try warm(try object(row, "warm"))
            let worker = try object(row, "serving_worker")
            guard let pid = worker["pid"]?.number, pid > 0, pid.rounded() == pid, worker["measurement_components_requests"] == 0 else {
                throw invalid("background serving peak received captures")
            }
            for key in ["load", "capture_load"] {
                let load = try object(row, key)
                guard load["ok"] == true, load["dtype"]?.string == dtype, load["requested_load"]?["dtype"]?.string == dtype else {
                    throw invalid("background load precision differs")
                }
                try nonnegative(load, "cold_load_seconds")
            }
            try qualification(row, folder: folder, base: base)
            if dtype != "fp16" {
                guard row["offered"] == .bool(dtype == "int4" || row["qualified"] == true),
                    row["precisionQualityWarnings"]?.object == (dtype == "int4" && row["qualified"] != true ? ["int4": .string(PrecisionStatSet.int4QualityWarning)] : [:]),
                    let diagnostics = row["end_to_end"]?.array, diagnostics.count == 3
                else { throw invalid("background offering/mask coverage differs") }
                for value in diagnostics {
                    let item = try requireObject(value)
                    guard let mae = item["mean_absolute_error"]?.number, mae >= 0, mae <= 1,
                        item["bound"] == 0.02, item["passed"] == .bool(mae <= 0.02), item["diagnostic_only"] == true,
                        item["affects_precision_gate"] == false, item["shape"] == [384, 512]
                    else { throw invalid("invalid background mask diagnostic") }
                }
                for entry in try object(try object(row, "gate"), "components").values {
                    guard entry["bounds"]?["relative_l2_max"] == 0.05, entry["bounds"]?["cosine_min"] == 0.999 else {
                        throw invalid("background quantized bounds differ")
                    }
                }
                try warm(try object(row, "paired_baseline_warm"))
            }
        }
    }

    private static func currentIntake(_ path: String, folder: URL) throws -> Bool {
        let results = folder.appendingPathComponent("results")
        let files = try FileManager.default.contentsOfDirectory(at: results, includingPropertiesForKeys: nil)
        for file in files where file.pathExtension == "json" {
            let table = try WireJSON.decode(Data(contentsOf: file))
            guard table["schema"] == 2, let stats = table["stats"]?.object else { continue }
            let cells = Array(stats["metrics"]?.object?.values ?? [:].values) + [stats["availability"] ?? .null]
            if cells.contains(where: { cell in
                guard let source = cell["provenance"]?["source"]?.string else { return false }
                return String(source.split(separator: ";", maxSplits: 1, omittingEmptySubsequences: false)[0]).trimmingCharacters(in: .whitespaces) == path
            }) {
                return true
            }
        }
        return false
    }

    private static func qualification(_ row: JSONObject, folder: URL, base: String) throws {
        let q = try object(row, "qualification"), plan = try object(q, "fixtures")
        let fixed = q["searched"] == false && plan["searched"] == false && plan["heldout"] == nil
        let manifest = try ModelManifest.load(folder)
        guard !fixed || manifest.kind != "voices" else {
            throw invalid("searched-capable voice recipes require distinct held-out qualification")
        }
        try fields(q, required: Set(["policy", "searched", "fixtures", "selection_gate", "ordinary_load", "passed"] + (fixed ? [] : ["heldout_gate"])))
        try fields(plan, required: Set(["policy", "selection"] + (fixed ? ["searched"] : ["heldout"])))
        guard q["searched"]?.bool != nil, q["passed"] == row["qualified"], (fixed ? q["selection_gate"] : q["heldout_gate"]) == row["gate"],
            q["selection_gate"] == row["selection_gate"],
            try read(
                base
                    + (fixed
                        ? "qualification-plan-" + (["int8", "int4"].contains(row["precision"]?.string ?? "") ? "quantized" : (row["precision"]?.string ?? "")) + ".json"
                        : "qualification-plan.json"),
                folder: folder) == plan,
            try IntakeQualification.complete(
                plan: plan, selection: object(q, "selection_gate"), heldout: fixed ? nil : object(q, "heldout_gate"),
                searched: q["searched"]?.bool == true, ordinaryLoad: object(q, "ordinary_load")) == q
        else { throw invalid("incomplete or mismatched held-out qualification") }
        guard try IntakeQualification.validatedRow(row, model: manifest) == q else { throw invalid("invalid completion qualification") }
        let load = try object(q, "ordinary_load")
        guard load["ok"]?.bool == true, load["requested_load"]?["dtype"] == row["precision"] else { throw invalid("qualification must load the exact tier ordinarily") }
        let inventory = try read(base + "fixture-manifest.json", folder: folder)
        guard let files = inventory["files"]?.array else { throw invalid("missing captured file inventory") }
        func recorded(_ path: String, hash: JSONValue?) throws {
            guard let hash = hash?.string, hash.count == 64,
                files.filter({ $0["path"]?.string == path && $0["sha256"]?.string == hash }).count == 1
            else { throw invalid("qualification file identity differs from captured inventory") }
            let url = try SafePath.resolve(base + path, inside: folder)
            // Raw tensor evidence stays local; public checkouts retain its captured hash inventory.
            if FileManager.default.fileExists(atPath: url.path) {
                guard try FileDigest.sha256(url) == hash else { throw invalid("qualification file digest mismatch") }
            } else if !path.hasSuffix(".safetensors") {
                throw invalid("missing qualification JSON evidence")
            }
        }
        for (arm, directory, fallback) in [("selection", "fixture", "quantized-bounds.json"), ("heldout", "heldout-fixture", "heldout-quantized-bounds.json")] {
            if fixed && arm == "heldout" { continue }
            let id = try object(plan, arm), gate = try object(q, arm + "_gate")
            try fields(
                id, required: Set(["inputs_sha256", "bounds_sha256", "component_keys", "component_count"] + (fixed ? [] : ["plan_sha256"])),
                optional: ["bounds", "bounds_file", "bounds_policy"])
            for key in ["inputs_sha256", "bounds_sha256"] + (fixed ? [] : ["plan_sha256"]) { try digest(id, key, count: 64) }
            for key in ["reference_outputs_sha256", "candidate_outputs_sha256"] { try digest(gate, key, count: 64) }
            let name = id["bounds_file"]?.string ?? fallback
            guard !name.contains("/"), !name.hasPrefix(".") else { throw invalid("unsafe qualification bounds path") }
            let bounds = try read(base + name, folder: folder)
            try recorded(directory + "/inputs.safetensors", hash: id["inputs_sha256"])
            if !fixed { try recorded(directory + "/fixture-plan.json", hash: id["plan_sha256"]) }
            try recorded(name, hash: id["bounds_sha256"])
            if let embedded = id["bounds"] { guard embedded == .object(bounds) else { throw invalid("embedded frozen bounds mismatch") } }
            _ = try IntakeQualification.validateGate(gate, bounds: bounds)
            if let format = gate["format_qualification"]?.object {
                let file = "format-qualification/format-gate.json", boundFile = "format-qualification/bounds.json"
                try recorded(file, hash: gate["format_qualification_sha256"])
                try recorded(boundFile, hash: format["bounds_sha256"])
                guard try read(base + file, folder: folder) == format else { throw invalid("nested format qualification differs from retained gate") }
                _ = try IntakeQualification.validateGate(format, bounds: read(base + boundFile, folder: folder))
            }
            // Tie output identities to the retained inventory, including when raw arrays stay local.
            for key in ["reference_outputs_sha256", "candidate_outputs_sha256"] {
                guard files.contains(where: { $0["path"]?.string?.hasSuffix("/outputs.safetensors") == true && $0["sha256"] == gate[key] }) else {
                    throw invalid("qualification output identity is absent from captured inventory")
                }
            }
        }
    }

    private static func voiceRecipes(_ root: JSONObject, folder: URL, run: String) throws {
        let half = try text(root, "selected_half"), recipes = try object(root, "recipes")
        guard ["bf16", "fp16"].contains(half) else { throw invalid("invalid voice reference half") }
        try fields(recipes, required: [half, "int8", "int4"])
        for dtype in [half, "int8", "int4"] {
            let recipe = try text(recipes, dtype)
            let snapshot = try read("gate/evidence/intake/" + run + "/" + dtype + "-snapshot.json", folder: folder)
            guard snapshot["model"] == root["model"], snapshot["precision"]?.string == dtype,
                snapshot["derived"]?["recipe"]?.string == recipe
            else { throw invalid("voice recipe/snapshot identity mismatch") }
        }
    }

    private static func servingWorker(_ row: JSONObject) throws {
        let worker = try object(row, "serving_worker"), captureLoad = try object(row, "capture_load")
        try fields(worker, required: ["pid", "measurement_components_requests"])
        guard let pid = worker["pid"]?.number, pid > 0, pid.rounded() == pid,
            worker["measurement_components_requests"]?.number == 0, captureLoad["ok"]?.bool == true
        else { throw invalid("serving peak needs a fresh worker with zero component captures") }
        try nonnegative(captureLoad, "load_s")
    }

    private static func warm(_ root: JSONObject) throws {
        guard let samples = root["samples_s"]?.array, !samples.isEmpty, samples.allSatisfy({ ($0.number ?? -1) >= 0 }) else { throw invalid("invalid warm samples") }
        try nonnegative(root, "median_s")
    }
    private static func fields(_ root: JSONObject, required: Set<String>, optional: Set<String> = []) throws {
        guard required.isSubset(of: Set(root.keys)), Set(root.keys).subtracting(required).isSubset(of: optional) else { throw invalid("result has missing or unknown fields") }
    }
    private static func text(_ root: JSONObject, _ key: String) throws -> String {
        guard let value = root[key]?.string, !value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw invalid("missing/invalid \(key)") }; return value
    }
    private static func requireObject(_ value: JSONValue) throws -> JSONObject {
        guard let object = value.object else { throw invalid("result entry must be an object") }; return object
    }
    private static func object(_ root: JSONObject, _ key: String) throws -> JSONObject {
        guard let value = root[key]?.object else { throw invalid("missing/invalid \(key) object") }; return value
    }
    private static func nonnegative(_ root: JSONObject, _ key: String) throws {
        guard let value = root[key]?.number, value.isFinite, value >= 0 else { throw invalid("missing/invalid \(key) number") }
    }
    private static func digest(_ root: JSONObject, _ key: String, count: Int) throws {
        let value = try text(root, key)
        guard value.count == count, value.allSatisfy({ "0123456789abcdef".contains($0) }) else { throw invalid("invalid \(key) digest") }
    }
    private static func existing(_ path: String, folder: URL, directory: Bool = false) throws -> URL {
        let url = try SafePath.resolve(path, inside: folder)
        let values = try url.resourceValues(forKeys: [.isRegularFileKey, .isDirectoryKey])
        guard directory ? values.isDirectory == true : values.isRegularFile == true else { throw invalid("missing evidence \(path)") }; return url
    }
    private static func read(_ path: String, folder: URL) throws -> JSONObject { try WireJSON.decode(Data(contentsOf: existing(path, folder: folder))) }
    private static func finite(_ value: JSONValue) throws {
        switch value {
        case .number(let n): if !n.isFinite { throw invalid("result numbers must be finite") }
        case .object(let root): for value in root.values { try finite(value) }
        case .array(let values): for value in values { try finite(value) }
        default: break
        }
    }
    /// Intake evidence names each precision once and covers every precision the folder declares. It may also keep the rows of
    /// a quantized tier the folder has since withdrawn (AGENTS.md, Quality gate: a tier that is not worth keeping is removed):
    /// those rows stay validated as recorded, and the 16-bit figures bound to the same run stay valid.
    static func coversDeclaredTiers(_ precisions: [String?], manifest: ModelManifest) -> Bool {
        let recorded = precisions.compactMap { $0 }, tiers = Set(recorded)
        return recorded.count == precisions.count && tiers.count == recorded.count && tiers.isSuperset(of: manifest.precisions)
            && tiers.subtracting(manifest.precisions).isSubset(of: ["int8", "int4"])
    }
    private static func invalid(_ message: String) -> ManifestError { .invalid(message) }
    private static func validRun(_ run: String) -> Bool { run.range(of: "^[A-Za-z0-9][A-Za-z0-9.-]*$", options: .regularExpression) != nil }
}
