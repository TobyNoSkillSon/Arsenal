// SPDX-License-Identifier: MIT
import Foundation

/// The `tools` names a model folder may declare. Capability windows normally place a model by its declared operations
/// alone (the app's ToolCatalog); a folder may also name one explicitly, and then must declare one of its operations.
public enum ToolNames {
    /// The app's own tool surfaces.
    public static let basic: Set<String> = ["chat", "image", "voice", "sound", "operation", "background"]
    /// Each capability window and the operations it offers (kept equal to the app's ToolCatalog by a test).
    public static let capability: [String: Set<String>] = [
        "transcribe": ["transcribe", "diarize"], "separate": ["separate_audio"], "search": ["rerank", "embed"], "translate": ["translate"],
        "document": ["read_document"], "extract": ["extract"], "select": ["segment", "track_video"], "find": ["detect", "ground_screen"],
        "depth": ["estimate_depth"], "upscale": ["upscale_image", "upscale_video"], "smooth": ["interpolate_video"], "make3d": ["image_to_3d"]
    ]
    public static var all: Set<String> { basic.union(capability.keys) }
}
