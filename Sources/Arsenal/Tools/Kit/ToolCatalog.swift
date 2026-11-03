// SPDX-License-Identifier: MIT
import Foundation

// Which window offers which operation. A model needs no tool declaration of its own for these windows: the operations
// its folder declares place it (a model that declares `transcribe` is in Transcribe, one that declares `detect` and
// `segment` is in Find Objects and Select Objects). Operations the windows use behind the scenes (sessions) and those
// meant for agents only (token counts, end-of-turn checks) put a model in no window; the generic Run… window remains
// for any operation no window offers.

enum ToolCatalog {
    /// The capability windows in menu order.
    static let windows: [ToolKind] = [.transcribe, .separate, .translate, .document, .extract, .search, .select, .find, .depth, .upscale, .smooth, .make3d]

    /// The operations a window offers, in its own order (the first is its everyday one).
    static func operations(_ tool: ToolKind) -> [String] {
        switch tool {
        case .transcribe: return ["transcribe", "diarize"]
        case .separate: return ["separate_audio"]
        case .search: return ["rerank", "embed"]
        case .translate: return ["translate"]
        case .document: return ["read_document"]
        case .extract: return ["extract"]
        case .select: return ["segment", "track_video"]
        case .find: return ["detect", "ground_screen"]
        case .depth: return ["estimate_depth"]
        case .upscale: return ["upscale_image", "upscale_video"]
        case .smooth: return ["interpolate_video"]
        case .make3d: return ["image_to_3d"]
        default: return []
        }
    }

    /// Operations a window drives itself (a reusable image or video session) or that only agents call: they place a
    /// model in no window and never make a generic Run… entry.
    static let behindTheScenes: Set<String> = [
        "segment_session", "video_session", "transcribe_session", "diarize_session", "count_tokens", "end_of_turn", "voices", "delete_voice", "delete_saved_voice"
    ]

    /// Operations that serve agents and other models rather than a person: decisions, embeddings, rankings, turn ends,
    /// token counts. A model that declares nothing else works in the background: it stays in the Models table, the API
    /// and MCP, and has no main-menu entry and no Show in menu switch.
    static let backgroundOperations: Set<String> = ["decide", "judge", "embed", "rerank", "end_of_turn", "count_tokens"]
    /// Kinds whose models all work in the background unless they declare an operation a person uses (TIPS's detection).
    static let backgroundKinds: Set<String> = ["decisions", "embeddings"]

    /// A model that only works in the background. `declared` nil: its operations are not read yet, so its kind decides.
    static func backgroundOnly(kind: String, declared: Set<String>?) -> Bool {
        guard let declared, !declared.isEmpty else { return backgroundKinds.contains(kind) }
        return declared.subtracting(backgroundOperations).subtracting(behindTheScenes).isEmpty
    }

    /// The capability windows a model's declared operations place it in, in menu order.
    static func windows(declared: Set<String>) -> [ToolKind] {
        windows.filter { tool in operations(tool).contains(where: declared.contains) }
    }

    /// The tools a model's menu section offers: its folder's own tools, with the generic tool replaced by the
    /// capability windows its operations place it in (the generic one stays for operations none of them offer). A
    /// model that declares only behind-the-scenes operations offers no tool.
    static func resolve(tools: [ToolKind], declared: Set<String>?) -> [ToolKind] {
        guard tools.contains(.operation), let declared, !declared.isEmpty else { return tools }
        let windows = windows(declared: declared).filter { !tools.contains($0) }
        let offered = Set(windows.flatMap(operations))
        let rest = declared.subtracting(offered).subtracting(behindTheScenes)
        return tools.flatMap { $0 == .operation ? (rest.isEmpty ? windows : windows + [.operation]) : [$0] }
    }

    /// The folder a window shows when nothing is on show: where most of what it makes goes (ToolFiles.folder(mime:)
    /// places each file by its type).
    static func folder(_ tool: ToolKind) -> URL {
        switch tool {
        case .separate: return ToolFiles.folder(mime: "audio/wav")
        case .smooth: return ToolFiles.folder(mime: "video/mp4")
        case .select, .find, .depth, .upscale: return ToolFiles.folder(mime: "image/png")
        default: return ToolFiles.folder(mime: "text/plain")
        }
    }
}
