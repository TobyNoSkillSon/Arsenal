// SPDX-License-Identifier: MIT
import Foundation

/// Measures the actual service; no family loader, quantization, cache or gate is duplicated here.
public actor MeasurementAdapter: PrecisionMeasurementRunner {
    public nonisolated let fixture: Bool
    private let service: RealModelsService
    private let model: ModelManifest
    private let directory: URL
    private var loadPeak: UInt64?
    public init(service: RealModelsService, model: ModelManifest, directory: URL, fixture: Bool = false) {
        self.service = service; self.model = model; self.directory = directory; self.fixture = fixture
    }
    public func prepare(dtype: String, sourceDtype: String) async throws -> MeasurementPreparation {
        try await service.unload(id: model.id)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let current = try await service.manifest(model.id)
        guard current.weights.revision == model.weights.revision, sourceDtype == current.basePrecision else {
            throw ManifestError.invalid("Served revision/source differs from pending measurement")
        }
        let selectedRow = try await row()
        if let reason = ServingMeasurementPolicy.refusal(model: model.id, fast: selectedRow.fast) { throw ManifestError.invalid(reason) }
        // This is precisely users' Get, including recipes/vendor formats and the selected converter.
        let plan = try await service.getPlan(id: model.id, dtype: dtype)
        guard plan.totalDownloadBytes == 0 else { throw ManifestError.invalid("Measurement never downloads; import the pinned source/assets before draining") }
        try await service.measurementGet(id: model.id, precision: dtype, approvedPlan: plan)
        let peak = ServedFootprintPeak(); peak.start(); defer { peak.stop() }
        try await service.measurementLoad(id: model.id, precision: dtype)
        loadPeak = peak.value
        let loaded = try await row()
        let context = loaded.precisionStats[dtype]?.value(.maximumContextTokens).value.flatMap(Int.init(exactly:))
        return MeasurementPreparation(sourceDtype: sourceDtype, dtype: dtype, inMemory: false, loadPeakBytes: loadPeak, maximumContextTokens: context, servedPath: true)
    }
    private func row() async throws -> ModelRow {
        guard let row = await service.rows().first(where: { $0.id == model.id }) else { throw ManifestError.invalid("Served model is absent") }
        return row
    }
    public func quality(kind: ModelStatKind) async throws -> MeasurementOutput {
        if kind == .chat {
            let reply = try await service.supervisor.request(
                id: model.id, op: "measurement_logits",
                params: [
                    "prefix": .string(MeasurementWorkload.prefix), "continuation": .string(MeasurementWorkload.continuation),
                    "output_dir": .string(directory.path)
                ], timeout: 300)
            guard let path = reply["output_file"]?.string else { throw ManifestError.invalid("Loaded chat worker has no teacher-forced logits response") }
            let file = try SafePath.resolve(path, inside: directory)
            defer { try? FileManager.default.removeItem(at: file) }
            guard try FileDigest.bytes(file) <= 256 * 1024 * 1024 else { throw ManifestError.invalid("Logits exceed bounded output budget") }
            let logits = try JSONDecoder().decode([[Double]].self, from: Data(contentsOf: file))
            return MeasurementOutput(fixtureID: MeasurementWorkload.identity(kind: kind), logits: logits)
        }
        let result = try await operation(kind: kind, batched: kind == .decisions, count: 4)
        defer { MeasurementMedia.cleanup(result.reply, jobs: service.paths.directory("jobs")) }
        return try MeasurementMedia.output(kind: kind, reply: result.reply, fixtureID: MeasurementWorkload.identity(kind: kind))
    }
    public func warmup(kind: ModelStatKind, steps: Int) async throws {
        for _ in 0..<steps {
            let result = try await operation(kind: kind, batched: false, count: 1)
            MeasurementMedia.cleanup(result.reply, jobs: service.paths.directory("jobs"))
        }
    }
    public func window(kind: ModelStatKind, steps: Int, batched: Bool) async throws -> MeasurementWindow {
        let peak = ServedFootprintPeak(); peak.start(); defer { peak.stop() }
        var elapsed = 0.0, units = 0.0, energyUnits = 0.0, first: Double?
        // Chat is one ordinary bounded generation; media is one ordinary usable output; decisions repeat singles or judge a batch.
        let count = kind == .decisions && !batched ? steps : 1
        for _ in 0..<count {
            let result = try await operation(kind: kind, batched: batched, count: batched ? steps : 1)
            defer { MeasurementMedia.cleanup(result.reply, jobs: service.paths.directory("jobs")) }
            elapsed += result.seconds; first = first ?? result.first
            switch kind {
            case .decisions: units += Double(try MeasurementMedia.answers(result.reply, batched: batched).count)
            case .chat:
                guard let tokens = result.reply["generated_tokens"]?.number, tokens > 1, let firstToken = result.decodeFirst else {
                    throw ManifestError.invalid("Ordinary chat request produced fewer than two tokens or no usable output")
                }
                // Includes wire delivery and final usable output; exclude prefill from sustained decode.
                elapsed -= firstToken; units += tokens - 1; energyUnits += tokens
            case .images: units += 1
            case .voices, .sound, .enhancement:
                guard let seconds = result.reply["duration_s"]?.number, seconds > 0 else { throw ManifestError.invalid("Ordinary audio operation returned no duration") }
                units += seconds
            case .audio, .embeddings, .perception, .restoration, .geometry: throw ModelStatKind.noWorkload(kind)
            }
        }
        return MeasurementWindow(
            elapsedSeconds: elapsed, completedUnits: units, firstOutputSeconds: first,
            peakBytes: max(loadPeak ?? 0, peak.value ?? 0), batchSize: batched ? steps : 1, energyUnits: kind == .chat ? energyUnits : nil)
    }
    private func operation(kind: ModelStatKind, batched: Bool, count: Int) async throws -> (reply: JSONObject, seconds: Double, first: Double?, decodeFirst: Double?) {
        let input = try MeasurementWorkload.request(kind: kind, model: model.id, batched: batched, count: count, directory: directory)
        let observation = MeasurementDelivery()
        let began = ProcessInfo.processInfo.systemUptime
        let result = try await service.run(id: model.id, op: input.op, params: input.params, sink: { frame in await observation.receive(frame) })
        let seconds = ProcessInfo.processInfo.systemUptime - began
        let first = await observation.first.map { $0 - began }
        // Delete ordinary output artifacts after capturing quality; timing ends before stat decoding/cleanup.
        return (result, seconds, first, await observation.decodeFirst.map { $0 - began })
    }
    public func gate(kind: ModelStatKind, dtype: String, metrics: [String: Double]) -> MeasurementGate {
        let exact: Bool
        switch kind {
        case .chat: exact = metrics[StatMetric.nextTokenAgreementPercent.rawValue] == 100
        case .decisions: exact = metrics[StatMetric.agreementPercent.rawValue] == 100
        case .images: exact = metrics[StatMetric.psnrDB.rawValue] == 120
        case .voices, .sound, .enhancement: exact = metrics["sampleIdentity"] == 1
        case .audio, .embeddings, .perception, .restoration, .geometry: exact = false
        }
        return MeasurementGate(
            passed: exact, policy: "Reference argmax preservation; images/audio require output identity until a noise gate is declared",
            unavailableReason: exact ? nil : "Own-16-bit output differs; a noise-tolerance gate has not been declared for this fixture.")
    }
    public func unload() async { try? await service.unload(id: model.id) }
    public func shutdown() async { await service.shutdown() }
}

private actor MeasurementDelivery {
    var first: Double?
    var decodeFirst: Double?
    func receive(_ frame: WireFrame) {
        let usable: Bool
        switch frame {
        case .binary(_, let bytes): usable = !bytes.isEmpty
        case .json(let body):
            if body["phase"]?.string == "decode", (body["generated_tokens"]?.number ?? 0) >= 1, decodeFirst == nil {
                decodeFirst = ProcessInfo.processInfo.systemUptime
            }
            usable = ["token", "reasoning"].contains(body["event"]?.string ?? "") && body["text"]?.string?.isEmpty == false
        }
        if usable && first == nil { first = ProcessInfo.processInfo.systemUptime }
    }
}
