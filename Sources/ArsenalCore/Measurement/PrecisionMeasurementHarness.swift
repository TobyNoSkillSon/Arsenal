// SPDX-License-Identifier: MIT
import Foundation

public struct MeasurementPreparation: Codable, Sendable {
    public var sourceDtype: String
    public var dtype: String
    public var servedPath: Bool
    public var inMemory: Bool
    public var downloadedBytes: Int64
    public var loadPeakBytes: UInt64?
    public var maximumContextTokens: Int?
    public init(
        sourceDtype: String, dtype: String, inMemory: Bool = true, downloadedBytes: Int64 = 0, loadPeakBytes: UInt64? = nil, maximumContextTokens: Int? = nil,
        servedPath: Bool = false
    ) {
        self.sourceDtype = sourceDtype; self.dtype = dtype; self.inMemory = inMemory; self.downloadedBytes = downloadedBytes; self.loadPeakBytes = loadPeakBytes
        self.servedPath = servedPath; self.maximumContextTokens = maximumContextTokens
    }
}

/// A fixed fixture output, drained/evaluated before it is returned. Chat uses teacher-forced logits,
/// never a free-running generation. Images are 1024² RGB; audio retains sample rate and count.
public struct MeasurementOutput: Codable, Sendable {
    public var decisions: [Int]?
    public var logits: [[Double]]?
    public var samples: [Double]?
    public var width: Int?
    public var height: Int?
    public var sampleRate: Int?
    public var fixtureID: String
    public init(fixtureID: String, decisions: [Int]? = nil, logits: [[Double]]? = nil, samples: [Double]? = nil, width: Int? = nil, height: Int? = nil, sampleRate: Int? = nil) {
        self.fixtureID = fixtureID; self.decisions = decisions; self.logits = logits; self.samples = samples
        self.width = width; self.height = height; self.sampleRate = sampleRate
    }
}

public struct MeasurementWindow: Codable, Sendable {
    public var elapsedSeconds: Double
    public var completedUnits: Double
    public var firstOutputSeconds: Double?
    public var peakBytes: UInt64?
    public var batchSize: Int
    public var energyUnits: Double?
    public init(elapsedSeconds: Double, completedUnits: Double, firstOutputSeconds: Double? = nil, peakBytes: UInt64? = nil, batchSize: Int = 1, energyUnits: Double? = nil) {
        self.elapsedSeconds = elapsedSeconds; self.completedUnits = completedUnits; self.firstOutputSeconds = firstOutputSeconds
        self.peakBytes = peakBytes; self.batchSize = batchSize; self.energyUnits = energyUnits
    }
}

public struct MeasurementGate: Codable, Sendable {
    public var passed: Bool
    public var policy: String
    public var unavailableReason: String?
    public init(passed: Bool, policy: String, unavailableReason: String? = nil) {
        self.passed = passed; self.policy = policy; self.unavailableReason = unavailableReason
    }
}

/// Measurement surrounds ordinary Get/Load/operations; chat adds only teacher-forced logits.
/// Get derives each snapshot from its original 16-bit source, exactly as for users.
/// window returns only after synchronizing GPU leaves and recurrent state. Stat-only host conversions are outside timing; default usable-output materialization stays included.
public protocol PrecisionMeasurementRunner: Sendable {
    var fixture: Bool { get }
    func prepare(dtype: String, sourceDtype: String) async throws -> MeasurementPreparation
    func quality(kind: ModelStatKind) async throws -> MeasurementOutput
    func warmup(kind: ModelStatKind, steps: Int) async throws
    func window(kind: ModelStatKind, steps: Int, batched: Bool) async throws -> MeasurementWindow
    func gate(kind: ModelStatKind, dtype: String, metrics: [String: Double]) async throws -> MeasurementGate
    func unload() async
}

public struct MeasurementProbe: Sendable {
    public var begin: @Sendable () async throws -> Void
    public var end: @Sendable () async throws -> CPUGPUEnergy
    public var batteryCapacityJoules: Double?
    public init(capacityJoules: Double?, begin: @escaping @Sendable () async throws -> Void, end: @escaping @Sendable () async throws -> CPUGPUEnergy) {
        batteryCapacityJoules = capacityJoules; self.begin = begin; self.end = end
    }
    public static var unavailable: Self {
        Self(
            capacityJoules: nil, begin: { throw ManifestError.invalid("CPU/GPU energy counters unavailable") },
            end: { throw ManifestError.invalid("CPU/GPU energy counters unavailable") })
    }
}

public struct MeasurementMethod: Sendable {
    public var chip: String
    public var gpuCores: Int?
    public var date: String
    public var controlled: Bool
    public var busyMachine: Bool
    public var source: String
    public init(chip: String, date: String = ISO8601DateFormatter().string(from: Date()), controlled: Bool = false, busyMachine: Bool = true, source: String, gpuCores: Int? = nil)
    {
        self.chip = chip; self.gpuCores = gpuCores; self.date = date; self.controlled = controlled; self.busyMachine = busyMachine; self.source = source
    }
}

public enum PrecisionMeasurementHarness {
    private static let loadMemoryMethod = "Served worker physical footprint sampled ordinary Load peak; conversion is a separate Get process"
    private static let runningMemoryMethod = "Served worker physical footprint sampled ordinary Load and operation peak"
    private static let combinedMemoryMethod = "Served worker physical footprint sampled ordinary Load and single/batch operations"

    /// No approval: append all three runs and return, before touching the adapter or counters.
    /// Approval is supplied only by the agent's explicit go command, never saved as a recurring permission.
    public static func run(
        model: String, kind: ModelStatKind, sourceDtype: String, revision: String, runnerID: String, mode: String,
        runner: any PrecisionMeasurementRunner, probe: MeasurementProbe, method: MeasurementMethod, queue: PendingBenchmarkQueue, approved: Bool = false
    ) async throws -> [PrecisionStatsResult] {
        guard ["fp16", "bf16"].contains(sourceDtype) else { throw ManifestError.invalid("Measurement source must be the model's original 16-bit dtype") }
        guard !kind.isOperation else { throw ModelStatKind.noWorkload(kind) }
        if !runner.fixture && !approved {
            try queue.enqueue(
                [sourceDtype, "int8", "int4"].map {
                    PendingBenchmark(model: model, kind: kind, dtype: $0, sourceDtype: sourceDtype, revision: revision, runner: runnerID, mode: mode)
                })
            return []
        }
        let runID = UUID().uuidString
        var results: [PrecisionStatsResult] = []
        var reference: MeasurementOutput?
        var baseStats: [PrecisionStatSet] = []
        var bracketFailure: String?
        for dtype in [sourceDtype, "int8", sourceDtype, "int4", sourceDtype] {
            let bracket = dtype == sourceDtype && !results.isEmpty
            var stats = PrecisionStatSet(kind: kind, dtype: dtype, runner: runnerID)
            do {
                let preparation = try await runner.prepare(dtype: dtype, sourceDtype: sourceDtype)
                guard preparation.sourceDtype == sourceDtype, preparation.dtype == dtype, runner.fixture ? preparation.inMemory : preparation.servedPath,
                    preparation.downloadedBytes == 0
                else {
                    throw ManifestError.invalid("Runner must Load the ordinary served precision derived by Get from own 16-bit without downloads")
                }
                let output = bracket ? reference! : try await runner.quality(kind: kind)
                if reference == nil { reference = output }
                guard let reference, output.fixtureID == reference.fixtureID else { throw ManifestError.invalid("Candidate fixture differs from 16-bit fixture") }
                let quality: [(StatMetric, Double)]
                var comparisonError: String?
                do { quality = try qualityMetrics(kind: kind, reference: reference, candidate: output) } catch {
                    if dtype == sourceDtype { throw error }
                    quality = []; comparisonError = error.localizedDescription
                }
                let state: StatState = runner.fixture ? .quickCheck : method.controlled ? .measured : .quickCheck
                let baseline = "own " + sourceDtype
                func provenance(_ description: String) -> StatProvenance {
                    StatProvenance(
                        chip: method.chip, date: method.date,
                        method: description + (runner.fixture ? "; CPU fixture, not model evidence" : method.busyMachine ? "; Mac in use" : ""),
                        baseline: baseline, source: method.source + ":" + output.fixtureID, revision: revision, run: runID, gpuCores: method.gpuCores)
                }
                for (metric, value) in quality { stats.set(metric, StatValue(state: state, value: value, provenance: provenance(qualityMethod(kind)))) }
                let gate: MeasurementGate
                if let comparisonError {
                    for metric in kind.metrics where [.agreementPercent, .nextTokenAgreementPercent, .nextTokenKL, .psnrDB, .spectralDistance].contains(metric) {
                        stats.set(metric, .unavailable(comparisonError))
                    }
                    gate = MeasurementGate(passed: false, policy: "Quality comparison requires matching counts, shapes and finite output", unavailableReason: comparisonError)
                } else {
                    gate = try await runner.gate(kind: kind, dtype: dtype, metrics: gateMetrics(kind: kind, quality: quality, reference: reference, candidate: output))
                }
                guard !gate.policy.isEmpty, gate.passed || gate.unavailableReason?.isEmpty == false else {
                    throw ManifestError.invalid("Runner must name its quality gate and failure reason")
                }
                stats.availability = gate.passed ? StatValue(state: state) : .unavailable(gate.unavailableReason!)
                stats.availability.provenance = provenance("Runner quality gate: " + gate.policy)
                if let bytes = preparation.loadPeakBytes, kind == .chat {
                    stats.set(.loadPeakMemoryGB, StatValue(state: state, value: Double(bytes) / 1e9, provenance: provenance(loadMemoryMethod)))
                }
                if kind == .chat, let context = preparation.maximumContextTokens, context > 0 {
                    stats.set(
                        .maximumContextTokens, StatValue(state: .declared, value: Double(context), provenance: provenance("Model config maximum context; no KV preallocation")))
                }
                try await runner.warmup(kind: kind, steps: 3)
                var energyError: String?
                do { try await probe.begin() } catch { energyError = error.localizedDescription }
                let window: MeasurementWindow
                do { window = try await runner.window(kind: kind, steps: kind == .images ? 1 : 32, batched: false) } catch {
                    if energyError == nil { _ = try? await probe.end() }; throw error
                }
                let energy: CPUGPUEnergy?
                if energyError == nil {
                    do { energy = try await probe.end() } catch { energyError = error.localizedDescription; energy = nil }
                } else {
                    energy = nil
                }
                try validate(window)
                let speedMetric = speed(kind)
                let speedValue = [.chat].contains(kind) ? window.completedUnits / window.elapsedSeconds : window.elapsedSeconds / window.completedUnits
                stats.set(
                    speedMetric,
                    StatValue(
                        state: state, value: kind == .decisions ? speedValue * 1000 : speedValue,
                        provenance: provenance(
                            "3 ordinary warm requests; short ordinary operation window; own-16-bit brackets interleaved around candidates; usable output delivered before outside stop; batch 1"
                        )))
                if let first = window.firstOutputSeconds, [.chat, .voices].contains(kind) {
                    stats.set(
                        kind == .chat ? .firstTokenSeconds : .firstAudioSeconds,
                        StatValue(state: state, value: first, provenance: provenance("Warm request to first usable output; excludes empty/role frames")))
                }
                if let bytes = window.peakBytes {
                    stats.set(.peakMemoryGB, StatValue(state: state, value: Double(bytes) / 1e9, provenance: provenance(runningMemoryMethod)))
                }
                if kind == .decisions {
                    do {
                        let batch = try await runner.window(kind: kind, steps: 32, batched: true); try validate(batch)
                        guard batch.batchSize > 1 else { throw ManifestError.invalid("Runner supports sequential decisions only") }
                        stats.set(
                            .batchedDecisionsPerSecond,
                            StatValue(
                                state: state, value: batch.completedUnits / batch.elapsedSeconds,
                                provenance: provenance("32 decisions; batch size \(batch.batchSize); GPU drained before stop")))
                        if let bytes = batch.peakBytes {
                            let peak = max(Double(bytes) / 1e9, stats.value(.peakMemoryGB).value ?? 0)
                            stats.set(.peakMemoryGB, StatValue(state: state, value: peak, provenance: provenance(combinedMemoryMethod)))
                        }
                    } catch { stats.set(.batchedDecisionsPerSecond, .unavailable(error.localizedDescription)) }
                }
                let (energyMetric, denominator) = try energyUnit(kind)
                if let energy, let capacity = probe.batteryCapacityJoules {
                    let percent = try BatteryEnergy.percent(
                        joules: energy.joules, capacityJoules: capacity, completedUnits: window.energyUnits ?? window.completedUnits, statedUnits: denominator)
                    stats.set(
                        energyMetric,
                        StatValue(
                            state: state, value: percent,
                            provenance: provenance(
                                "Rootless IOReport CPU+GPU joules / battery full-charge capacity; normalized to stated unit; two significant digits; approximately ±5%; includes background CPU/GPU"
                            )))
                } else {
                    stats.set(energyMetric, .unavailable(energyError ?? "Battery capacity unavailable"))
                }
                await runner.unload()
            } catch {
                if dtype == sourceDtype && !bracket { reference = nil }
                await runner.unload()
                let identity = stats.availability.provenance
                stats.availability = .unavailable(error.localizedDescription); stats.availability.provenance = identity
                for metric in kind.metrics where stats.value(metric).state == .notMeasured { stats.set(metric, .unavailable(error.localizedDescription)) }
            }
            if stats.availability.provenance == nil {
                stats.availability.provenance = StatProvenance(
                    chip: method.chip, date: method.date, method: "Measurement attempt; " + (runner.fixture ? "CPU fixture" : "Mac in use"),
                    baseline: "own " + sourceDtype, source: method.source, revision: revision, run: runID, gpuCores: method.gpuCores)
            }
            try stats.validate()
            if dtype == sourceDtype { baseStats.append(stats) }
            if bracket && stats.availability.state == .unavailable { bracketFailure = stats.availability.reason }
            if !bracket { results.append(PrecisionStatsResult(model: model, mode: mode, stats: stats)) }
            if reference == nil { break } // A failed 16-bit reference cannot generate valid lower-precision comparisons.
        }
        if let index = results.firstIndex(where: { $0.stats.dtype == sourceDtype }) {
            for metric in kind.metrics {
                let values = baseStats.compactMap { $0.value(metric).value }
                guard values.count == 3 else { continue }
                var value = results[index].stats.value(metric)
                value.value = [.peakMemoryGB, .loadPeakMemoryGB].contains(metric) ? values.max() : values.reduce(0, +) / Double(values.count)
                results[index].stats.set(metric, value)
            }
        }
        if let bracketFailure {
            for index in results.indices {
                let provenance = results[index].stats.availability.provenance
                results[index].stats.availability = .unavailable("Own-16-bit timing bracket failed: " + bracketFailure)
                results[index].stats.availability.provenance = provenance
            }
        }
        return results
    }
    private static func gateMetrics(kind: ModelStatKind, quality: [(StatMetric, Double)], reference: MeasurementOutput, candidate: MeasurementOutput) -> [String: Double] {
        var metrics = Dictionary(uniqueKeysWithValues: quality.map { ($0.0.rawValue, $0.1) })
        if [.voices, .sound, .enhancement].contains(kind) {
            metrics["sampleIdentity"] = StatsQuality.audioIdentical(reference: reference, candidate: candidate) ? 1 : 0
        }
        return metrics
    }
    public static func qualityMetrics(kind: ModelStatKind, reference: MeasurementOutput, candidate: MeasurementOutput) throws -> [(StatMetric, Double)] {
        guard reference.fixtureID == candidate.fixtureID else { throw ManifestError.invalid("Quality fixture identities differ") }
        switch kind {
        case .decisions:
            guard let a = reference.decisions, let b = candidate.decisions else { throw ManifestError.invalid("Decision answers unavailable") }
            return [(.agreementPercent, try StatsQuality.agreement(reference: a, candidate: b))]
        case .chat:
            guard let a = reference.logits, let b = candidate.logits else { throw ManifestError.invalid("Teacher-forced next-token logits unavailable") }
            let score = try StatsQuality.chat(reference: a, candidate: b)
            return [(.nextTokenAgreementPercent, score.agreement), (.nextTokenKL, score.kl)]
        case .images:
            guard reference.width == 1024, reference.height == 1024, candidate.width == reference.width, candidate.height == reference.height,
                let a = reference.samples, let b = candidate.samples, a.count == 1024 * 1024 * 3
            else { throw ManifestError.invalid("Quality image must be 1024² RGB at default steps and fixed seed") }
            return [(.psnrDB, try StatsQuality.psnr(reference: a, candidate: b))]
        case .voices, .sound, .enhancement:
            guard let rate = reference.sampleRate, rate > 0, candidate.sampleRate == rate, let a = reference.samples, let b = candidate.samples else {
                throw ManifestError.invalid("Quality audio sample rates differ or samples are unavailable")
            }
            return [(.spectralDistance, try StatsQuality.spectralDistance(reference: a, candidate: b))]
        case .audio, .embeddings, .perception, .restoration, .geometry: throw ModelStatKind.noWorkload(kind)
        }
    }
    private static func validate(_ window: MeasurementWindow) throws {
        guard window.elapsedSeconds.isFinite, window.elapsedSeconds > 0, window.completedUnits.isFinite, window.completedUnits > 0,
            window.firstOutputSeconds.map({ $0.isFinite && $0 >= 0 }) ?? true,
            window.energyUnits.map({ $0.isFinite && $0 > 0 }) ?? true
        else { throw ManifestError.invalid("Invalid measurement window") }
    }
    private static func speed(_ kind: ModelStatKind) -> StatMetric {
        switch kind {
        case .decisions: return .millisecondsPerDecision
        case .chat: return .tokensPerSecond
        case .images: return .secondsPerImage
        case .voices, .sound, .enhancement: return .realTimeFactor
        case .audio, .embeddings, .perception, .restoration, .geometry: return .secondsPerRequest
        }
    }
    private static func qualityMethod(_ kind: ModelStatKind) -> String {
        switch kind {
        case .decisions: return "Fixed decision argmax answers vs own 16-bit; exact item count/order"
        case .chat: return "Fixed short teacher-forced continuation; next-token argmax agreement and mean KL in nats vs own 16-bit; no free-running generation"
        case .images: return "Fixed seed; 1024² RGB [0,1]; default steps; PSNR vs own 16-bit; identical pixels cap at 120 dB"
        case .voices, .sound, .enhancement: return "Fixed clip/seed; same sample rate and sample count; Hann 256/hop 128 magnitude spectra relative L2 vs own 16-bit"
        case .audio, .embeddings, .perception, .restoration, .geometry: return "No served quality workload"
        }
    }
    public static func energyUnit(_ kind: ModelStatKind) throws -> (StatMetric, Double) {
        switch kind {
        case .decisions: return (.batteryPercentPer1000Decisions, 1000)
        case .chat: return (.batteryPercentPer100000Tokens, 100000)
        case .images: return (.batteryPercentPer100Images, 100)
        case .voices: return (.batteryPercentPerHourSpeech, 3600)
        case .sound, .enhancement: return (.batteryPercentPerHourAudio, 3600)
        case .audio, .embeddings, .perception, .restoration, .geometry: throw ModelStatKind.noWorkload(kind)
        }
    }
}
