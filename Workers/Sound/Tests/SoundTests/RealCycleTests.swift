// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalWire
import SoundEngine
import Testing
@testable import SoundWorker

private func repository() -> URL {
    URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
}

@Test func modelFoldersValidate() throws {
    let repo = repository(), kinds = Set(try KindSpec.load(directory: repo.appendingPathComponent("Resources/kinds")).map(\.id))
    for id in ["stable-audio-3-medium", "stable-audio-3-small-sfx", "re-use", "mossformer2-se-48k", "unipase", "deepfilternet3"] {
        let folder =
            [repo.appendingPathComponent("models/" + id), repo.appendingPathComponent("retired-models/" + id)].first {
                FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path)
            } ?? repo.appendingPathComponent("models/" + id)
        let manifest = try ModelManifest.load(folder, kinds: kinds)
        #expect(manifest.problems(folder: folder, kinds: kinds).isEmpty)
        var parser = TOMLParser(try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8))
        #expect(
            try ModelSchemaValidation.validate(document: parser.parse(), schemaData: Data(contentsOf: repo.appendingPathComponent("Resources/schemas/model.schema.json"))).isEmpty)
        for op in manifest.operations {
            let data = try OperationSchema.read(path: op.input, folder: folder, kindsDirectory: repo.appendingPathComponent("Resources/kinds"))
            try ModelSchemaValidation.checkSchema(schemaData: data)
        }
    }
}

// Opt in for exactly one model and only under mlock gpulock; normal tests never initialise a real model.
@Test(.heavy) func realGetLoadRunUnload() async throws {
    let env = ProcessInfo.processInfo.environment
    guard let id = env["ARSENAL_SOUND_REAL_MODEL"], let binary = env["ARSENAL_SOUND_BINARY"] else { return }
    // Isolated support only: the sound temp tree, or a lab profile directory named profile-<model>.
    guard let support = env["ARSENAL_SUPPORT_DIR"],
        support.hasPrefix("/tmp/arsenal-mb-v2-sound/") || support.hasPrefix("/private/tmp/arsenal-mb-v2-sound/")
            || URL(fileURLWithPath: support).pathComponents.contains("profile-" + id)
    else {
        throw WorkerFailure(code: "test_environment", message: "Real sound checks require an isolated temporary support directory")
    }
    let repo = repository(), folder = repo.appendingPathComponent("models/" + id)
    let model = try ModelManifest.load(folder, kinds: ["sound", "enhancement"])
    let paths = ArsenalPaths(); try paths.create()
    let output = paths.directory("jobs").appendingPathComponent(id)
    let artifact = TestArtifact(binary: URL(fileURLWithPath: binary))
    // SafePath intentionally rejects a symlink at the blob-store root. Keep the large, isolated
    // development Get store persistent; the actual runner's support/cache remain under /tmp.
    let getSupport = URL(fileURLWithPath: env["ARSENAL_SOUND_GET_SUPPORT"] ?? NSHomeDirectory() + "/Library/Application Support/Arsenal-dev/weights/mb-v2-sound/get-support")
    // Every model but RE-USE imports its pinned files from the local Hugging Face cache; no network.
    let client = id == "re-use" ? HuggingFaceClient() : HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!)
    let fetcher = ModelFetcher(support: getSupport, client: client, converter: WorkerPrecisionConverter(artifacts: artifact, paths: paths))
    let network = NetworkCount(), start = Date()
    // RE-USE ships BF16 only (models/re-use/model.toml); the half-only models their one half dtype; the rest FP16.
    let dtype = id == "re-use" ? "bf16" : SoundEngines.halfOnlyModels[id] ?? "fp16"
    let got = try await fetcher.get(model: model, modelDirectory: folder, precision: dtype, progress: { network.add($0) })
    #expect(id == "re-use" || network.value == 0)
    let worker = WorkerProcess()
    do {
        try await worker.start(try await artifact.launch(model: model, runner: model.runners[0], directory: folder, paths: paths))
        let load = try await worker.request(
            op: "load", params: ["model_dir": .string(folder.path), "weights_dir": .string(got.weightsDirectory.path), "dtype": .string(dtype), "fast": true], timeout: 90)
        var params: JSONObject = ["out_dir": .string(output.path), "stream": true]
        let op: String
        if id == "re-use" {
            let sink = try WaveSink(directory: output, rate: 16000)
            let samples = (0..<8000).map { Float(sin(Double($0) * 2 * .pi * 170 / 16000) * 0.1) }
            try sink.append(PCM.encode(samples)); try sink.finish()
            params["audio"] = ["path": .string(sink.file.path)]; op = "enhance"
        } else if model.kind == "enhancement" {
            // ARSENAL_SOUND_INPUT: a real recording; ARSENAL_SOUND_ONE_PASS: the one-pass limit in seconds.
            let path = try #require(env["ARSENAL_SOUND_INPUT"], "enhancement checks need ARSENAL_SOUND_INPUT")
            params["audio"] = ["path": .string(path)]; op = "enhance"
            if let seconds = env["ARSENAL_SOUND_ONE_PASS"].flatMap(Double.init) { params["one_time_decode_length"] = .number(seconds) }
        } else {
            params["prompt"] = "rain on a tin roof"; params["duration"] = 1.5; params["seed"] = 20260927; params["steps"] = 2
            op = id == "stable-audio-3-medium" ? "generate_music" : "generate_sound"
        }
        let events = EventCount()
        let result = try await worker.request(op: op, params: params, timeout: 90, sink: { events.add($0) })
        let path = try #require(result["path"]?.string)
        #expect(result["sha256"]?.string == (try PCM.digest(URL(fileURLWithPath: path))))
        let clip = try PCM.input(URL(fileURLWithPath: path))
        #expect(clip.frames > 0 && clip.samples.allSatisfy(\.isFinite))
        if model.kind == "enhancement", id != "re-use", let input = env["ARSENAL_SOUND_INPUT"] {
            // Same rate, channels and length as the input.
            let source = try PCM.input(URL(fileURLWithPath: input))
            #expect(clip.channels == source.channels && clip.sampleRate == source.sampleRate && clip.frames == source.frames)
        } else {
            #expect(clip.channels == (id == "re-use" ? 1 : 2))
        }
        #expect(events.progress > 0 && events.audioBytes > 0)
        let status = try await worker.request(op: "status")
        await worker.unload()
        #expect(await worker.exited)
        let evidence: JSONObject = [
            "model": .string(id), "dtype": .string(dtype), "network_bytes": .number(Double(network.value)),
            "get_cycle_s": .number(Date().timeIntervalSince(start)), "load": .object(load),
            "run": .object(result), "status": .object(status), "progress_events": .number(Double(events.progress)), "stream_audio_bytes": .number(Double(events.audioBytes)),
            "exited": true, "measurement_lock": false
        ]
        if let directory = env["ARSENAL_SOUND_EVIDENCE"] { try WireJSON.encode(evidence).write(to: URL(fileURLWithPath: directory).appendingPathComponent(id + "-cycle.json")) }
    } catch { await worker.unload(grace: 0.2); throw error }
}

private struct TestArtifact: WorkerArtifact {
    let binary: URL
    func prepare(model: ModelManifest, runner: RunnerSpec, directory: URL) async throws {}
    func launch(model: ModelManifest, runner: RunnerSpec, directory: URL, paths: ArsenalPaths) async throws -> WorkerLaunch {
        var env = RunnerEnvironment.scrub(); env["ARSENAL_SUPPORT_DIR"] = paths.support.path; env["ARSENAL_CACHE_DIR"] = paths.cache.path
        return WorkerLaunch(
            executable: binary, kind: model.kind, runner: "native", environment: env, log: paths.directory("logs").appendingPathComponent(UUID().uuidString + ".log"))
    }
}
private final class NetworkCount: @unchecked Sendable {
    private let lock = NSLock(); private var bytes: Int64 = 0
    private var last: [String: Int64] = [:]
    func add(_ progress: DownloadProgress) {
        lock.withLock {
            guard let n = progress.networkBytes else { return }
            let key = progress.contentID ?? progress.file
            bytes += max(0, n - (last[key] ?? 0)); last[key] = n
        }
    }
    var value: Int64 { lock.withLock { bytes } }
}
private final class EventCount: @unchecked Sendable {
    private let lock = NSLock(); private var counts = (progress: 0, audioBytes: 0)
    func add(_ frame: WireFrame) {
        lock.withLock {
            switch frame {
            case .json(let r): if r["event"]?.string == "progress" { counts.progress += 1 }
            case .binary(_, let data): counts.audioBytes += data.count
            }
        }
    }
    var progress: Int { lock.withLock { counts.progress } }
    var audioBytes: Int { lock.withLock { counts.audioBytes } }
}
