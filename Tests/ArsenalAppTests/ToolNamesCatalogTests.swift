// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Model folders validate `tools` names against ArsenalCore's list; the app routes by ToolCatalog. They must agree.
struct ToolNamesCatalogTests {
    @Test func manifestToolNamesMatchTheAppsWindows() {
        #expect(Set(ToolKind.allCases.map(\.rawValue)) == ToolNames.all)
        #expect(Set(ToolCatalog.windows.map(\.rawValue)) == Set(ToolNames.capability.keys))
        for tool in ToolCatalog.windows { #expect(Set(ToolCatalog.operations(tool)) == ToolNames.capability[tool.rawValue], "\(tool)") }
    }

    private var repo: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }

    /// The window placement of every model folder: its tools (schema-3 default: the generic tool) placed by its declared
    /// operations through ToolCatalog. A new folder, or a changed tools line or operation, shows here before any menu does.
    private static let placements: [String: [ToolKind]] = [
        "ace-step-1.5": [.sound],
        "astrobert": [.search],
        "audiobox-aesthetics": [.operation],
        "audiosr-basic": [.operation],
        "big-lama": [.operation],
        "birefnet": [.background],
        "breeze-tts-2": [.voice, .sound],
        "chandra-ocr-2": [.document],
        "chronos-2": [.operation],
        "chronos-bolt-small": [.operation],
        "clap-htsat-fused": [.operation],
        "d1-3b": [],
        "da3-small": [.depth],
        "da3metric-large": [.depth],
        "ddcolor-modelscope": [.operation],
        "deepfilternet3": [],
        "depth-anything-v2-small": [.depth],
        "diarizen-large-v2": [.transcribe],
        "edgetam": [.operation],
        "elephant-1": [],
        "embeddinggemma-2": [.search],
        "eomt-coco-panoptic-large-640": [.select],
        "esm2-t33-650m": [.operation],
        "esm2-t6-8m": [.operation],
        "esm3-sm-open-v1": [.operation],
        "fish-drama-3": [.voice, .sound],
        "florence-2-base": [.operation],
        "florence-2-large": [.operation],
        "flux2-klein-4b": [.image],
        "flux3-image-klein": [.image],
        "geneformer-v2-104m": [.operation],
        "gimm-vfi-r": [.smooth],
        "glm-ocr": [.operation],
        "granite-embedding-small-english-r2": [.search],
        "granite-ttm-r3": [.operation],
        "grounding-dino-base": [.find],
        "harrier-oss-v1-0.6b": [.search],
        "htdemucs-ft": [.separate],
        "idol": [.make3d],
        "index-translate-2b": [.translate],
        "index-translate-9b": [.translate],
        "jina-reranker-v3.5": [.search],
        "kokoro-82m": [.voice, .sound],
        "laya-english": [],
        "laya-multilingual": [],
        "laya-typed-decisions": [],
        "lco-embedding-omni-7b": [.search],
        "locateanything-3b": [.find],
        "manga-ocr-base": [.operation],
        "melband-roformer-kim": [.separate],
        "minimax-music3": [.sound],
        "mitra-classifier-2": [.operation],
        "moge-3-vitl": [.depth],
        "molformer-xl-both-10pct": [.operation],
        "moss-transcribe-diarize": [.transcribe],
        "mossformer2-se-48k": [],
        "nemotron-3-diarization": [.transcribe],
        "nuextract3": [.document, .extract],
        "ovis-vl-embedding-2b": [.search],
        "owlv2-base-patch16-ensemble": [.find],
        "parakeet-v3-ultra": [.transcribe],
        "pocket-tts": [.voice, .sound],
        "pyannote-community-1": [.transcribe],
        "qwen-image-2.1": [.image],
        "qwen-image-2.1-turbo": [.image],
        "qwen3-asr-0.6b": [.transcribe],
        "qwen3-asr-1.7b": [.transcribe],
        "qwen3-embedding-0.6b": [.search],
        "qwen3-embedding-8b": [.search],
        "qwen3-reranker-8b": [.search],
        "qwen3-tts-base": [.voice, .sound],
        "qwen3-tts-voicedesign": [.voice, .sound],
        "qwen3.8-27b": [.chat],
        "re-use": [],
        "real-esrgan-x4plus": [.upscale],
        "rf-detr-nano": [.find],
        "rf-detr-large": [.find],
        "rife-4.25": [.smooth],
        "sam-audio-large": [.separate],
        "sortformer-4spk-v2": [.transcribe],
        "sam-audio-small": [.separate],
        "sam2.1-hiera-large": [.select],
        "sam2.1-hiera-tiny": [.select],
        "sam3.1": [.select],
        "seedvr2-3b": [.upscale],
        "flashvsr-v1.1": [.upscale],
        "smart-turn-v3": [],
        "stable-audio-3-medium": [.sound],
        "stable-audio-3-small-sfx": [.sound],
        "t0-alpha": [.operation],
        "t0-gamma": [.operation],
        "teleocr": [.document],
        "tiger-dnr": [.separate],
        "tiger-speech": [.separate],
        "timesfm-2.5-200m": [.operation],
        "timestral-209m": [.operation],
        "tipsv2-l14": [.search, .select, .find],
        "trellis2-4b": [.make3d],
        "triposr": [.make3d],
        "ui-venus-1.5-2b": [.find],
        "ui-venus-2-9b": [.find],
        "ultra-50g": [.operation],
        "unipase": [],
        "uvdoc": [.operation],
        "vitpose-base-simple": [.operation],
        "vjepa2-vitl-fpc64-256": [.operation],
        "voice-gender-ecapa": [.operation],
        "von-1.1": [],
        "von-1.2": [],
        "voxcpm2": [.voice, .sound],
        "yue2-3b": [.sound]
    ]

    @Test func everyModelFolderPlacesInTheWindowsItDeclares() throws {
        let shipped = try FileManager.default.contentsOfDirectory(at: repo.appendingPathComponent("models"), includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
        // Non-open models withdrawn to retired-models/ keep their pinned placement, so the table still covers them.
        let withdrawn = Self.placements.keys.filter { id in !shipped.contains { $0.lastPathComponent == id } }
            .map { repo.appendingPathComponent("retired-models/" + $0) }
        let folders = shipped + withdrawn
        #expect(Set(folders.map(\.lastPathComponent)) == Set(Self.placements.keys), "pin every new or removed folder")
        for folder in folders {
            let id = folder.lastPathComponent
            let model = try ModelManifest.load(folder)
            let tools = model.resolvedTools.compactMap(ToolKind.init(rawValue:))
            #expect(ToolCatalog.resolve(tools: tools, declared: Set(model.operations.map(\.name))) == Self.placements[id], "\(id)")
        }
    }

    /// Session operations are driven by their window or live transcription; a model declaring one must not get a Run… entry.
    @Test func sessionOperationsAreBehindTheScenes() {
        #expect(ToolCatalog.behindTheScenes.isSuperset(of: ["transcribe_session", "diarize_session"]))
        #expect(ToolCatalog.resolve(tools: [.operation], declared: ["transcribe_session"]).isEmpty)
        #expect(ToolCatalog.resolve(tools: [.operation], declared: ["diarize_session"]).isEmpty)
        #expect(ToolCatalog.resolve(tools: [.operation], declared: ["transcribe", "transcribe_session"]) == [.transcribe])
    }
}
