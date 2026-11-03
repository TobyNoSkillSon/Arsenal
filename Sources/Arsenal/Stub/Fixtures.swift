// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

// Stub catalog for the stub service, stub renders and tests: the v1 catalog's twelve models (ids, sizes, licences and
// precisions as in models/), in the states the stub renders cover. Figures are placeholders shaped like the results
// files, never measurements; the report's renders come from the real catalog (`Arsenal --render-real`, real-*.png).

enum Fixtures {
    enum Scene: String, CaseIterable {
        case firstLaunch = "first-launch", downloading, deriving, loaded, reload = "reload-preview", failed = "get-failed"
        case badges, measured
        case localCopy = "local-copy", verifying = "verifying-local-copy", retention = "source-removed"
    }

    /// The four v1 kinds as `kinds/*.toml` declares them (read from the repository when present).
    static var kinds: [KindSpec] {
        let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent().appendingPathComponent("kinds")
        if let loaded = try? KindSpec.load(directory: repo), !loaded.isEmpty { return loaded }
        return fallbackKinds
    }
    static let fallbackKinds: [KindSpec] = [
        KindSpec(
            id: "voices", title: "Voices", order: 0,
            columns: [
                KindColumn(id: "speed", title: "Speed", unit: "×"), KindColumn(id: "first_audio", title: "First audio", unit: "s"),
                KindColumn(id: "roundtrip_wer", title: "Round-trip WER", unit: "%"), KindColumn(id: "peak_ram", title: "Peak RAM", unit: "GB")
            ]),
        KindSpec(
            id: "decisions", title: "Decisions", order: 1,
            columns: [
                KindColumn(id: "accuracy", title: "Accuracy", unit: "%"), KindColumn(id: "latency", title: "Latency", unit: "ms"),
                KindColumn(id: "peak_ram", title: "Peak RAM", unit: "GB")
            ]),
        KindSpec(
            id: "images", title: "Images", order: 2,
            columns: [
                KindColumn(id: "seconds_per_image", title: "Seconds per image", unit: "s"), KindColumn(id: "peak_ram", title: "Peak RAM", unit: "GB")
            ]),
        KindSpec(
            id: "chat", title: "Chat", order: 3,
            columns: [
                KindColumn(id: "context", title: "Context", unit: "tokens"), KindColumn(id: "tokens_per_s", title: "Tokens/s", unit: "tokens/s"),
                KindColumn(id: "first_token", title: "First token", unit: "s"), KindColumn(id: "peak_ram", title: "Peak RAM", unit: "GB")
            ])
    ]

    private static func row(
        _ id: String, _ name: String, _ kind: String, _ params: String, _ licence: String, _ precisions: [String], size: Int64,
        badge: ModelBadge, metrics: JSONObject, summary: String, runner: String = "native", publisher: String = "", released: Int? = nil,
        commercial: Bool = true, fastExact: Bool? = nil
    ) -> ModelRow {
        var row = ModelRow(
            id: id, name: name, kind: kind, params: params, licence: licence, precisions: precisions, selectedPrecision: precisions[0],
            fast: true, runner: runner, badge: badge, metrics: metrics, sizeBytes: size, summary: summary,
            verification: badge == .unverified ? "Not tried on this Mac yet" : badge.rawValue, publisher: publisher, released: released,
            commercial: commercial, runnerLanguage: "swift", fastExact: fastExact ?? (runner == "native"))
        row.purpose = purpose(id, kind)
        return row
    }

    /// What each catalog model's folder declares as its `purpose`; another fixture's from its kind, as Arsenal derives it.
    static let purposes: [String: String] = [
        "birefnet": "Background removal",
        "breeze-tts-2": "Bilingual speech, cloning and design",
        "d1-3b": "Text and image decisions",
        "flux2-klein-4b": "Image generation",
        "kokoro-82m": "Fast English speech",
        "laya-english": "English decisions",
        "laya-multilingual": "Decisions in 100+ languages",
        "laya-typed-decisions": "Typed workflow decisions",
        "pocket-tts": "Streaming speech, voice cloning",
        "qwen3-tts-base": "Speech in 10 languages, voice cloning",
        "qwen3-tts-voicedesign": "Speech in 10 languages, voice design",
        "qwen3.8-27b": "General text chat, long context",
        "re-use": "Speech cleanup",
        "stable-audio-3-medium": "Sound and music generation",
        "stable-audio-3-small-sfx": "Sound effects",
        "von-1.1": "Option scoring, order-sensitive",
        "von-1.2": "Option scoring, order-independent"
    ]
    static func purpose(_ id: String, _ kind: String) -> String {
        let known = ["voices", "sound", "enhancement", "decisions", "images", "chat"].contains(kind)
        return purposes[id] ?? ModelPurpose.derive(kind: kind, operations: known ? StubModelsService.kindOperations(kind) : [])
    }

    /// One model per kind as a first launch sees it: nothing on this Mac.
    /// Figures set by the scenes below are stub placeholders for state coverage only (stub-*.png), never shown as real.
    static let base: [ModelRow] = [
        row(
            "kokoro-82m", "Kokoro 82M", "voices", "82M", "Apache-2.0", ["fp16", "int8"], size: 341_742_463, badge: .unverified,
            metrics: [:], summary: "Fast preset-voice reader; the default for spoken replies.", publisher: "Hexgrad", released: 2025),
        row(
            "flux2-klein-4b", "FLUX.2 Klein 4B", "images", "4B", "Apache-2.0", ["bf16"], size: 15_980_140_843, badge: .unverified,
            metrics: [:], summary: "Text-to-image in four steps.", runner: "mflux", publisher: "Black Forest Labs", released: 2026),
        row(
            "qwen3.8-27b", "Qwen3.8 27B", "chat", "27B", "Apache-2.0", ["bf16", "int8", "int4"], size: 54_000_000_000, badge: .unverified,
            metrics: ["context": 32_768, "native_context": 262_144], summary: "General chat and instructions.", publisher: "Alibaba Qwen", released: 2026, fastExact: false)
    ]

    static let stubVoiceFigures: JSONObject = ["speed": "42.0×", "first_audio": "61 ms", "roundtrip_wer": "1.9%", "peak_ram": "0.4 GB"]

    static func rows(_ scene: Scene) -> [ModelRow] {
        var rows = base
        func set(_ id: String, _ change: (inout ModelRow) -> Void) {
            if let i = rows.firstIndex(where: { $0.id == id }) { change(&rows[i]) }
        }
        func ready(_ id: String, _ precisions: [String]) { set(id) { r in for p in precisions { r.states[p] = PrecisionState(state: .ready) } } }
        func loaded(_ id: String, _ p: String, fast: Bool = true) {
            set(id) { r in
                r.states[p] = PrecisionState(state: .loaded); r.selectedPrecision = p; r.fast = fast
            }
        }
        switch scene {
        case .firstLaunch: break
        case .downloading:
            set("flux2-klein-4b") { r in
                r.states["bf16"] = PrecisionState(state: .downloading, progress: 0.23, bytes: 3_675_000_000, totalBytes: 15_980_140_843)
            }
        case .deriving:
            set("kokoro-82m") { r in
                r.states["fp16"] = PrecisionState(state: .ready)
                r.states["int8"] = PrecisionState(state: .deriving); r.selectedPrecision = "int8"
            }
        case .loaded, .reload, .measured:
            ready("kokoro-82m", ["fp16", "int8"]); loaded("kokoro-82m", "fp16")
            ready("flux2-klein-4b", ["bf16"])
            ready("qwen3.8-27b", ["bf16", "int4"]); loaded("qwen3.8-27b", "int4", fast: false)
            if scene == .measured {
                set("kokoro-82m") { r in
                    r.badge = .measured; r.verification = "Measured"; r.metrics = stubVoiceFigures
                }
            }
        case .failed:
            set("flux2-klein-4b") { r in
                r.states["bf16"] = PrecisionState(
                    state: .failed, error: "Get failed: Hugging Face stopped sending data for 2 minutes. Get resumes from 6.1 GB.")
            }
        case .badges:
            // One of each: a quick check under gpulock (lighter, no badge), another chip's estimate (~), Unverified.
            set("kokoro-82m") { r in
                r.metrics = stubVoiceFigures; r.verification = "quick check"
                r.quickMetrics = ["speed": true, "first_audio": true, "roundtrip_wer": true, "peak_ram": true]
                r.states["fp16"] = PrecisionState(state: .ready)
            }
            set("flux2-klein-4b") { r in
                r.badge = .otherChip; r.verification = r.badge.rawValue
                r.metrics = ["seconds_per_image": "4.1 s \u{00b7} 1024\u{00d7}1024", "peak_ram": "13.2 GB"]
            }
        case .localCopy:
            // The vendor files are already on this Mac (another app's folder): Get imports them, nothing to download.
            set("qwen3.8-27b") { r in
                for p in r.precisions { r.downloadBytes[p] = 0; r.localCopyBytes[p] = 55_600_000_000 }
            }
        case .verifying:
            set("qwen3.8-27b") { r in
                r.localCopyBytes["bf16"] = 55_600_000_000; r.downloadBytes["bf16"] = 0
                r.states["bf16"] = PrecisionState(
                    state: .downloading, progress: 0.41, bytes: 22_800_000_000, totalBytes: 55_600_000_000, phase: "verifyingLocalCopy")
            }
        case .retention:
            // int4 made, then the 16-bit source released (CONTRACTS §14): a Get of bf16 or int8 downloads it again.
            ready("kokoro-82m", ["fp16"])
            set("qwen3.8-27b") { r in
                r.states["int4"] = PrecisionState(state: .ready); r.selectedPrecision = "int4"
                r.states["bf16"] = PrecisionState(state: .notInstalled, phase: PrecisionState.sourceRemovedPhase)
                r.downloadBytes = ["bf16": 55_600_000_000, "int8": 55_600_000_000, "int4": 0]
                r.retentionNote = "16-bit source removed to save 55.6 GB; Get re-downloads it"
            }
        }
        return rows
    }

    // MARK: The unified table (15 Oct): one row per state the Speed and Peak memory cells and the precision control show.
    // Placeholder figures shaped like the schema-2 results, for renders and tests only.

    /// A local kind, so the generic path (a folder's own figures) is covered next to the built-in kinds.
    static let videoKind = KindSpec(
        id: "video", title: "Video", order: 6, columns: [KindColumn(id: "speed", title: "Speed", unit: "s"), KindColumn(id: "peak_ram", title: "Peak RAM", unit: "GB")],
        icon: "film")
    static var figureKinds: [KindSpec] { kinds + [videoKind] }

    static let figureDate = "2026-10-09T14:00:00Z"
    /// One precision's set: the kind's speed figure and peak memory (nil: not measured), with extras for the hover.
    static func stats(
        _ kind: ModelStatKind, _ dtype: String, runner: String = "native", speed: Double? = nil, memory: Double? = nil, state: StatState = .quickCheck,
        chip: String = "m5-max", extras: [StatMetric: Double] = [:], unavailable: String? = nil, memoryState: StatState? = nil, basis: String? = nil,
        quality: Bool? = nil
    ) -> PrecisionStatSet {
        var set = PrecisionStatSet(kind: kind, dtype: dtype, runner: runner, unavailableReason: unavailable)
        set.qualityCheckPassed = quality
        guard unavailable == nil else { return set }
        func value(_ v: Double, _ s: StatState) -> StatValue {
            StatValue(
                state: s, value: v,
                provenance: StatProvenance(chip: chip, date: figureDate, method: s == .quickCheck ? "quick check" : "schema-2 harness", gpuCores: 40),
                basis: s == .derived ? basis : nil)
        }
        if let speed { set.set(TableFigures.speedMetric(kind), value(speed, state)) }
        if let memory { set.set(.peakMemoryGB, value(memory, memoryState ?? state)) }
        for (metric, v) in extras { set.set(metric, value(v, state)) }
        return set
    }

    static func figureRows() -> [ModelRow] {
        func row(_ id: String, _ name: String, _ kind: String, _ params: String, _ precisions: [String], size: Int64 = 1_000_000_000) -> ModelRow {
            var row = ModelRow(
                id: id, name: name, kind: kind, params: params, licence: "Apache-2.0", precisions: precisions, selectedPrecision: precisions[0], fast: true,
                sizeBytes: size, summary: "", publisher: "", fastExact: false)
            row.purpose = purpose(id, kind)
            return row
        }
        func ready(_ r: inout ModelRow) { for p in r.precisions { r.states[p] = PrecisionState(state: .ready) } }

        // Decisions, loaded at int8 (gold), custom kernels (bolt); int8 measured faster than fp16; int4 not measured.
        var laya = row("laya-multilingual", "Laya Multilingual", "decisions", "322M", ["fp16", "int8", "int4"], size: 644_000_000)
        ready(&laya); laya.states["int8"] = PrecisionState(state: .loaded); laya.selectedPrecision = "int8"; laya.hasFastKernels = true
        // int8 passed its quality check; int4 offered, its check not run yet: the int4 cell's triangle.
        laya.precisionStats = [
            "fp16": stats(.decisions, "fp16", speed: 8.1, memory: 1.27, state: .measured, extras: [.batchedDecisionsPerSecond: 1070]),
            "int8": stats(.decisions, "int8", speed: 6.0, memory: 0.74, state: .measured, extras: [.batchedDecisionsPerSecond: 1390], quality: true),
            "int4": stats(
                .decisions, "int4", memory: 0.52, memoryState: .derived,
                basis: "From the fp16 peak: 1.27 GB − 0.64 GB fp16 weights + 0.18 GB int4 weights")
        ]
        // Decisions, as its manifest declares (int4-everywhere): bf16 and int4, no int8; int4 not quality-checked yet.
        var d1 = row("d1-3b", "d1 \u{00b7} 3B", "decisions", "3.12B", ["bf16", "int4"], size: 6_240_000_000)
        ready(&d1)
        d1.precisionStats = ["bf16": stats(.decisions, "bf16", speed: 17.9, memory: 6.03, extras: [.batchedDecisionsPerSecond: 280])]
        // Images, loaded at bf16; int8 passed its quality check, int4 is offered with its failed check flagged.
        var flux = row("flux2-klein-4b", "FLUX.2 Klein 4B", "images", "4B", ["bf16", "int8", "int4"], size: 15_980_140_843)
        ready(&flux); flux.states["bf16"] = PrecisionState(state: .loaded)
        flux.precisionStats = [
            "bf16": stats(.images, "bf16", speed: 1.07, memory: 15.92),
            "int8": stats(.images, "int8", speed: 1.12, memory: 9.6, quality: true),
            "int4": stats(.images, "int4", memory: 9.08, quality: false)
        ]
        // Voices, loaded at fp16 with int8 chosen (Reload pending): int8 slower than fp16.
        var kokoro = row("kokoro-82m", "Kokoro 82M", "voices", "82M", ["fp16", "int8", "int4"], size: 164_000_000)
        ready(&kokoro); kokoro.states["fp16"] = PrecisionState(state: .loaded); kokoro.selectedPrecision = "int8"
        kokoro.precisionStats = [
            "fp16": stats(.voices, "fp16", speed: 1 / 46.8, memory: 0.82, state: .measured, extras: [.firstAudioSeconds: 0.088]),
            "int8": stats(.voices, "int8", speed: 1 / 39.0, memory: 0.61, state: .measured, extras: [.firstAudioSeconds: 0.094]),
            "int4": stats(.voices, "int4", speed: 1 / 46.5, memory: 0.52, state: .measured, quality: true)
        ]
        // Sound, on this Mac, nothing measured: two triangles.
        var audio = row("stable-audio-3-medium", "Stable Audio 3 Medium", "sound", "1.45B", ["fp16", "int8", "int4"], size: 5_800_000_000)
        ready(&audio)
        audio.precisionStats = ["int4": stats(.sound, "int4", quality: false)]
        // Enhancement, not on this Mac (Get): speed measured, memory not.
        var reuse = row("re-use", "RE-USE", "enhancement", "9.6M", ["fp16", "int8", "int4"], size: 38_000_000)
        reuse.precisionStats = ["fp16": stats(.enhancement, "fp16", speed: 1 / 212)]
        // Chat, a model the user added (no chat model ships).
        var chat = row("my-chat-4b", "My Chat 4B", "chat", "4B", ["bf16", "int8", "int4"], size: 8_000_000_000)
        ready(&chat); chat.local = true
        // int4 chosen, its quality not checked: the triangle on the gold cell.
        chat.selectedPrecision = "int4"
        chat.precisionStats = [
            "bf16": stats(.chat, "bf16", speed: 38.2, memory: 9.1, extras: [.firstTokenSeconds: 0.21]),
            "int4": stats(.chat, "int4", speed: 96.4, memory: 3.2)
        ]
        // A local kind: the folder's own figures.
        var video = row("my-video-5b", "My Video 5B", "video", "5B", ["bf16"], size: 10_000_000_000)
        ready(&video); video.local = true; video.badge = .measured
        video.metrics = ["speed": "38 s \u{00b7} 720p, 5 s clip", "peak_ram": .number(21.4)]
        // Background removal: its 16-bit release only so far, not measured (triangles), not loaded.
        var birefnet = row("birefnet", "BiRefNet", "images", "220M", ["fp16"], size: 444_473_596)
        ready(&birefnet); birefnet.tools = ["background"]
        return [laya, d1, flux, birefnet, kokoro, audio, reuse, chat, video]
    }
    /// A table as a user has it (A user has 5–7 models): seven models in five kinds from the figure rows, BiRefNet beside
    /// FLUX under Images.
    static func realisticRows() -> [ModelRow] {
        let ids: Set = ["laya-multilingual", "d1-3b", "flux2-klein-4b", "birefnet", "kokoro-82m", "stable-audio-3-medium", "re-use"]
        return figureRows().filter { ids.contains($0.id) }
    }
}
