// SPDX-License-Identifier: MIT
import Foundation

extension ModelIntakeCommand {
    static func voiceSanity(reference: JSONObject, candidate: JSONObject) throws -> JSONObject {
        let a = try MeasurementMedia.output(kind: .voices, reply: reference, fixtureID: "voice-default")
        let b = try MeasurementMedia.output(kind: .voices, reply: candidate, fixtureID: "voice-default")
        guard let left = a.samples, let right = b.samples, a.sampleRate == b.sampleRate, let rate = a.sampleRate,
            left.count >= 256, right.count >= 256
        else { throw ManifestError.invalid("Voice sanity needs nonempty PCM at matching rates") }
        let common = min(left.count, right.count)
        return [
            "diagnostic_only": true, "affects_precision_gate": false, "sample_rate_hz": .number(Double(rate)),
            "reference_samples": .number(Double(left.count)), "candidate_samples": .number(Double(right.count)), "common_prefix_samples": .number(Double(common)),
            "sample_counts_match": .bool(left.count == right.count), "sample_identity": .bool(left == right),
            "rms": .number(sqrt(right.reduce(0) { $0 + $1 * $1 } / Double(right.count))), "peak_abs": .number(right.map(abs).max()!),
            "common_prefix_spectral_relative_l2": .number(try StatsQuality.spectralDistance(reference: Array(left.prefix(common)), candidate: Array(right.prefix(common))))
        ]
    }
    static func writeVoiceStats(rows: [JSONValue], model: ModelManifest, folder: URL, artifact: BundledWorkerArtifact, run: String, result: URL) throws {
        try IntakeQualification.validateCompletion(rows: rows, model: model, evidence: folder.appendingPathComponent("gate/evidence/intake/" + run))
        try writeCaptureManifest(folder.appendingPathComponent("gate/evidence/intake/" + run))
        guard let runner = model.runners.first else { throw ManifestError.invalid("Missing voice runner") }
        let implementation = artifact.qualityImplementation(model: model, runner: runner, directory: folder)
        let captured = try WireJSON.decode(Data(contentsOf: result))
        let loadIdentity = try ModelLoadConfiguration.identity(model: model, runner: runner, folder: folder)
        guard captured["run"]?.string == run, captured["model"]?.string == model.id,
            captured["rows"]?.array?.filter({ $0["format_trial"]?.bool != true }) == rows, captured["implementation_identity"]?.string == implementation,
            captured["load_identity"]?.string == loadIdentity,
            captured["worker_sha256"]?.string == (try FileDigest.sha256(artifact.helpers.appendingPathComponent("arsenal-worker-native")))
        else { throw ManifestError.invalid("Voice receipt does not match exact run/build/load recipe") }
        let date = captured["date"]?.string ?? ISO8601DateFormatter().string(from: Date())
        let resultHash = try FileDigest.sha256(result)
        var outcomes: [JSONValue] = []
        for value in rows {
            guard let row = value.object, let dtype = row["precision"]?.string, row["warm"]?["median_s"]?.number != nil,
                let peak = row["peak_gb"]?.number, let gate = row["gate"]?.object,
                let qualification = row["qualification"]?.object, let selection = qualification["selection_gate"]?.object,
                let plan = qualification["fixtures"]?.object,
                let checked = try? IntakeQualification.complete(
                    plan: plan, selection: selection, heldout: gate,
                    searched: qualification["searched"]?.bool == true, ordinaryLoad: qualification["ordinary_load"]?.object), checked == qualification
            else { throw ManifestError.invalid("Incomplete voice intake row") }
            _ = try IntakeQualification.validatedRow(row, model: model)
            let source = "results/intake/" + result.lastPathComponent + "; bounds SHA256 " + (gate["bounds_sha256"]?.string ?? "missing")
            let provenance = StatProvenance(
                chip: HostChip.current, date: date,
                method:
                    "Production Load/speak on default speech workload, seed 0, streamed output; balanced ABBA BAAB warm windows; fresh workers exclude capture-only buffers from lifetime footprint. Busy Mac; quick check. Worker build: "
                    + (captured["worker_build"]?.string ?? "unspecified") + ".",
                baseline: "Selected " + model.basePrecision + " cast, same production path", source: source,
                revision: model.weights.revision, run: run + "-" + dtype, gpuCores: HostChip.gpuCores, defaultWorkload: true)
            var stats = PrecisionStatSet(kind: .voices, dtype: dtype, runner: runner.name)
            stats.qualityCheckPassed = qualification["passed"]?.bool
            stats.qualityIdentity = PrecisionEvidence.qualityIdentity(model: model, dtype: dtype, runner: runner.name, implementation: implementation, loadIdentity: loadIdentity)
            guard stats.qualityIdentity != nil else { throw ManifestError.invalid("Missing built worker source-closure receipt") }
            stats.availability = StatValue(
                state: dtype == "int8" && stats.qualityCheckPassed == false ? .unavailable : .quickCheck,
                reason: dtype == "int8" && stats.qualityCheckPassed == false ? "Did not pass the quality check" : nil, provenance: provenance)
            guard let rtf = row["warm_rtf"]?["median"]?.number else { throw ManifestError.invalid("Missing voice RTF") }
            stats.set(.realTimeFactor, StatValue(state: .quickCheck, value: rtf, provenance: provenance))
            if let first = row["sanity_reply"]?["first_audio_s"]?.number { stats.set(.firstAudioSeconds, StatValue(state: .quickCheck, value: first, provenance: provenance)) }
            if let distance = row["end_to_end"]?["common_prefix_spectral_relative_l2"]?.number {
                var diagnostic = provenance
                diagnostic.method =
                    "Unaligned common-prefix Hann256/hop128 magnitude-spectral distance on the one cold WAV per tier; diagnostic only, unequal lengths and sampling divergence are reported separately. "
                    + provenance.method
                stats.set(.spectralDistance, StatValue(state: .quickCheck, value: distance, provenance: diagnostic))
            }
            stats.set(.peakMemoryGB, StatValue(state: .quickCheck, value: peak, provenance: provenance))
            let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
            let mode = try ModelLoadConfiguration.defaultFast(model: model, runner: runner, folder: folder) ? "fast" : "exact"
            let data = try encoder.encode(PrecisionStatsResult(model: model.id, mode: mode, stats: stats))
            _ = try PrecisionStatsResult.decode(data)
            let tablePath = "results/intake-m5-max-" + dtype + "-native-" + mode + ".json"
            try data.write(to: folder.appendingPathComponent(tablePath), options: .atomic)
            let boundsPath = "gate/evidence/intake/" + run + "/" + "heldout-quantized-bounds.json"
            guard gate["bounds_sha256"]?.string == (try FileDigest.sha256(folder.appendingPathComponent(boundsPath))) else {
                throw ManifestError.invalid("Voice binding bounds differ from exact run")
            }
            outcomes.append(
                .object([
                    "result": .string(tablePath), "result_sha256": .string(FileDigest.sha256(data)), "dtype": .string(dtype),
                    "passed": .bool(stats.qualityCheckPassed == true), "quality_identity": .string(stats.qualityIdentity!),
                    "capture_result": .string("results/intake/" + result.lastPathComponent), "capture_result_sha256": .string(resultHash),
                    "bounds_sha256": gate["bounds_sha256"]!, "qualification": .object(qualification), "recipe": .string(model.derive?[dtype] ?? "vendor:" + dtype)
                ]))
        }
        let binding: JSONObject = [
            "schema": 3, "qualification_policy": .string(IntakeQualification.policyID), "family": .string(runner.worker ?? model.kind), "run": .string(run),
            "implementation_identity": .string(implementation!), "load_identity": .string(loadIdentity),
            "worker_sha256": captured["worker_sha256"]!, "outcomes": .array(outcomes),
            "basis":
                "Written by intake completion from frozen distinct selection/held-out inputs, complete gates, final stored recipe through ordinary Load, exact worker and stats. Schema 2 voice completion bindings are historical selection-set passes."
        ]
        let bindingData = try WireJSON.encode(binding)
        try bindingData.write(to: folder.appendingPathComponent("gate/evidence/intake/" + run + "/quality-binding.json"), options: .withoutOverwriting)
        try bindingData.write(to: folder.appendingPathComponent("gate/evidence/quality-binding-" + String(date.prefix(10)) + ".json"), options: .atomic)
    }
}
