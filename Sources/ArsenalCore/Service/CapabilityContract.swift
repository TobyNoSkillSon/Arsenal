// SPDX-License-Identifier: MIT
import Foundation

/// Public capability names are stable. Actual support and options come exclusively from the selected model's schema.
public struct CapabilityContract: Sendable {
    public let operation: String
    public let route: String
    public let tool: String
    public let title: String
    public static let all: [Self] = [
        .init(operation: "transcribe", route: "audio/transcribe", tool: "transcribe", title: "Transcribe"),
        .init(operation: "diarize", route: "audio/diarize", tool: "diarize", title: "Who spoke when"),
        .init(operation: "end_of_turn", route: "audio/end-of-turn", tool: "end_of_turn", title: "End of turn"),
        .init(operation: "separate_audio", route: "audio/separate", tool: "separate_audio", title: "Separate audio"),
        .init(operation: "embed", route: "embeddings/embed", tool: "embed", title: "Embed text, images or audio"),
        .init(operation: "rerank", route: "text/rerank", tool: "rerank", title: "Rerank documents"),
        .init(operation: "translate", route: "text/translate", tool: "translate", title: "Translate"),
        .init(operation: "read_document", route: "documents/read", tool: "read_document", title: "Read document"),
        .init(operation: "extract", route: "text/extract", tool: "extract", title: "Extract structured data"),
        .init(operation: "detect", route: "images/detect", tool: "detect", title: "Find objects"),
        .init(operation: "ground_screen", route: "images/ground", tool: "ground_screen", title: "Ground a screen instruction"),
        .init(operation: "segment", route: "images/segment", tool: "segment", title: "Select and segment"),
        .init(operation: "segment_session", route: "images/segment-session", tool: "segment_session", title: "Create or close an image session"),
        .init(operation: "video_session", route: "video/session", tool: "video_session", title: "Create or close a video session"),
        .init(operation: "track_video", route: "video/segment", tool: "track_video", title: "Track a video selection"),
        .init(operation: "estimate_depth", route: "images/depth", tool: "estimate_depth", title: "Estimate depth"),
        .init(operation: "upscale_image", route: "images/upscale", tool: "upscale_image", title: "Upscale image"),
        .init(operation: "upscale_video", route: "video/upscale", tool: "upscale_video", title: "Upscale video"),
        .init(operation: "interpolate_video", route: "video/interpolate", tool: "interpolate_video", title: "Interpolate video frames"),
        .init(operation: "colorize", route: "images/colorize", tool: "colorize", title: "Colorize a black-and-white photo"),
        .init(operation: "inpaint", route: "images/inpaint", tool: "inpaint", title: "Fill the masked part of a photo"),
        .init(operation: "image_to_3d", route: "images/to-3d", tool: "image_to_3d", title: "Make a GLB from an image"),
        .init(operation: "generate_music", route: "audio/music", tool: "generate_music", title: "Generate music with optional lyrics"),
        .init(operation: "enhance", route: "audio/enhance", tool: "enhance_audio", title: "Enhance audio")
    ]
}
