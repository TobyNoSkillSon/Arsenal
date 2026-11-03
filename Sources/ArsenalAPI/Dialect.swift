// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

/// A dialect owns its route descriptions and translation onto model operations.
/// Keep this list in the same module as the handler: discovery cannot silently
/// claim a route that the adapter does not implement.
protocol Dialect: Sendable {
    var id: String { get }
    var routes: [DialectRoute] { get }
    func accepts(_ path: String) -> Bool
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws
}

struct DialectRoute: Sendable {
    let method: String
    let path: String
    let operation: String
    let kind: String?
    let summary: String
    let requestMediaType: String
    init(_ method: String, _ path: String, _ operation: String, kind: String? = nil, summary: String, requestMediaType: String = "application/json") {
        self.method = method; self.path = path; self.operation = operation; self.kind = kind; self.summary = summary; self.requestMediaType = requestMediaType
    }
}

struct DialectRegistry: Sendable {
    let modules: [any Dialect]
    static let standard = Self(modules: [FoundationDialect(), TypeSafeDialect(), OpenAIDialect(), AnthropicDialect(), ElevenLabsDialect(), ReplicateDialect()])
    func module(for path: String) -> (any Dialect)? { modules.first { $0.accepts(path) } }
    var routes: [DialectRoute] { modules.flatMap(\.routes) }
}

struct OpenAIDialect: Dialect {
    let id = "openai"
    let routes: [DialectRoute] = [
        .init("get", "/v1/models", "models", summary: "Models in your table"),
        .init("get", "/v1/models/{model}", "model", summary: "Model in your table"),
        .init("post", "/v1/audio/speech", "speak", kind: "voices", summary: "Speech as WAV or PCM"),
        .init("post", "/v1/images/generations", "generate", kind: "images", summary: "PNG image generation"),
        .init("post", "/v1/chat/completions", "chat", kind: "chat", summary: "Chat completions and SSE")
    ]
    func accepts(_ path: String) -> Bool { path.hasPrefix("/v1/") }
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws { try await router.openai(request, connection) }
}
struct AnthropicDialect: Dialect {
    let id = "anthropic"
    let routes: [DialectRoute] = [
        .init("get", "/providers/anthropic/v1/models", "models", summary: "Chat models in your table"),
        .init("post", "/providers/anthropic/v1/messages", "chat", kind: "chat", summary: "Text Messages and named SSE"),
        .init("post", "/providers/anthropic/v1/messages/count_tokens", "count_tokens", kind: "chat", summary: "Count input tokens")
    ]
    func accepts(_ path: String) -> Bool { path.hasPrefix("/providers/anthropic/") }
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws { try await router.anthropic(request, connection) }
}
struct ElevenLabsDialect: Dialect {
    let id = "elevenlabs"
    let routes: [DialectRoute] = [
        .init("get", "/providers/elevenlabs/v1/models", "models", summary: "Audio models in your table"),
        .init("get", "/providers/elevenlabs/v1/voices", "voices", kind: "voices", summary: "Local voices"),
        .init("get", "/providers/elevenlabs/v2/voices", "voices", kind: "voices", summary: "Local voices"),
        .init("get", "/providers/elevenlabs/v1/voices/{voice_id}", "voices", kind: "voices", summary: "Local voice"),
        .init("post", "/providers/elevenlabs/v1/text-to-speech/{voice_id}", "speak", kind: "voices", summary: "Text to speech"),
        .init("post", "/providers/elevenlabs/v1/text-to-speech/{voice_id}/stream", "speak", kind: "voices", summary: "HTTP audio stream"),
        .init("post", "/providers/elevenlabs/v1/sound-generation", "generate_sound", kind: "sound", summary: "Sound effects"),
        .init("post", "/providers/elevenlabs/v1/music", "generate_music", kind: "sound", summary: "Instrumental music"),
        .init("post", "/providers/elevenlabs/v1/music/stream", "generate_music", kind: "sound", summary: "Generated music as HTTP chunks"),
        .init(
            "post", "/providers/elevenlabs/v1/audio-isolation", "enhance", kind: "enhancement", summary: "Clean speech from uploaded audio", requestMediaType: "multipart/form-data"
        ),
        .init(
            "post", "/providers/elevenlabs/v1/audio-isolation/stream", "enhance", kind: "enhancement", summary: "Clean speech as HTTP chunks",
            requestMediaType: "multipart/form-data"),
        .init("post", "/providers/elevenlabs/v1/text-to-voice/design", "design_voice", kind: "voices", summary: "Three voice previews"),
        .init("post", "/providers/elevenlabs/v1/text-to-voice/create-previews", "design_voice", kind: "voices", summary: "Three voice previews"),
        .init("post", "/providers/elevenlabs/v1/text-to-voice", "clone_voice", kind: "voices", summary: "Save a designed preview"),
        .init("get", "/providers/elevenlabs/v1/text-to-voice/{generated_voice_id}/stream", "preview", kind: "voices", summary: "Read preview audio"),
        .init("delete", "/providers/elevenlabs/v1/voices/{voice_id}", "delete_voice", kind: "voices", summary: "Delete a saved voice; presets cannot be deleted"),
        .init("post", "/providers/elevenlabs/v1/voices/add", "clone_voice", kind: "voices", summary: "Clone an uploaded reference", requestMediaType: "multipart/form-data")
    ]
    func accepts(_ path: String) -> Bool { path.hasPrefix("/providers/elevenlabs/") }
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws { try await router.elevenlabs(request, connection) }
}
struct ReplicateDialect: Dialect {
    let id = "replicate"
    let routes: [DialectRoute] = [
        .init("get", "/providers/replicate/v1/models", "models", summary: "Local prediction models"),
        .init("get", "/providers/replicate/v1/models/local/{model}", "model", summary: "Local prediction model"),
        .init("get", "/providers/replicate/v1/models/local/{model}/versions/{version}", "version", summary: "Local model version schema"),
        .init("post", "/providers/replicate/v1/predictions", "generate", summary: "Create a durable prediction"),
        .init("post", "/providers/replicate/v1/models/local/{model}/predictions", "generate", summary: "Create an official-model prediction"),
        .init("get", "/providers/replicate/v1/predictions/{prediction}", "prediction", summary: "Read a prediction"),
        .init("post", "/providers/replicate/v1/predictions/{prediction}/cancel", "cancel", summary: "Cancel a prediction")
    ]
    func accepts(_ path: String) -> Bool { path.hasPrefix("/providers/replicate/") }
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws { try await router.replicate(request, connection) }
}
struct TypeSafeDialect: Dialect {
    let id = "typesafe"
    let routes: [DialectRoute] = [
        .init("get", "/providers/typesafe/v1/models", "models", kind: "decisions", summary: "Decision models in your table"),
        .init("post", "/v1/systemone", "decide", kind: "decisions", summary: "Typed System One decisions"),
        .init("post", "/v1/judge", "judge", kind: "decisions", summary: "Batch typed decisions"),
        .init("post", "/providers/typesafe/v1/systemone", "decide", kind: "decisions", summary: "Typed System One decisions"),
        .init("post", "/providers/typesafe/v1/judge", "judge", kind: "decisions", summary: "Batch typed decisions")
    ]
    func accepts(_ path: String) -> Bool { path.hasPrefix("/providers/typesafe/") || ["/v1/systemone", "/v1/judge"].contains(path) }
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws { try await router.decisions(request, connection) }
}

struct FoundationDialect: Dialect {
    let id = "capabilities"
    let routes: [DialectRoute] =
        [
            .init("post", "/v1/audio/transcriptions", "transcribe", summary: "Transcription with timestamps and diarization", requestMediaType: "multipart/form-data"),
            .init("post", "/v1/audio/translations", "transcribe", summary: "Translate audio into English", requestMediaType: "multipart/form-data"),
            .init("post", "/v1/embeddings", "embed", summary: "Text, image and audio embeddings"),
            .init("post", "/v1/rerank", "rerank", summary: "Jina reranking"),
            .init("post", "/v2/rerank", "rerank", summary: "Cohere reranking"),
            .init("post", "/providers/cohere/v2/rerank", "rerank", summary: "Cohere reranking"),
            .init("post", "/v2/embed", "embed", summary: "Cohere multimodal embeddings"),
            .init("post", "/providers/cohere/v2/embed", "embed", summary: "Cohere multimodal embeddings"),
            .init("post", "/providers/deepl/v2/translate", "translate", summary: "DeepL text translation"),
            .init("get", "/providers/deepl/v2/languages", "translate", summary: "Declared supported languages"),
            .init("get", "/v2/languages", "translate", summary: "Declared supported languages"),
            .init("post", "/providers/mistral/v1/ocr", "read_document", summary: "Mistral page OCR"),
            .init("get", "/v1/realtime", "transcribe", summary: "WebSocket transcription session"),
            .init("get", "/arsenal/v1/jobs/{job}", "job", summary: "Read a durable operation job"),
            .init("post", "/arsenal/v1/jobs/{job}/cancel", "cancel", summary: "Cancel a durable operation job")
        ] + CapabilityContract.all.map { .init("post", "/arsenal/v1/" + $0.route, $0.operation, summary: $0.title) }
    func accepts(_ path: String) -> Bool { routes.contains { $0.path == path } || path == "/v1/realtime" }
    func handle(_ request: HTTPRequest, connection: HTTPConnection, router: APIRouter) async throws {
        switch request.path {
        case "/v1/audio/transcriptions": try await router.transcribe(request, connection)
        case "/v1/audio/translations": try await router.transcribe(request, connection, translate: true)
        case "/v1/embeddings": try await router.embedding(request, connection)
        case "/v2/embed", "/providers/cohere/v2/embed": try await router.embedding(request, connection, cohere: true)
        case "/v1/rerank", "/v2/rerank", "/providers/cohere/v2/rerank": try await router.rerank(request, connection)
        case "/providers/deepl/v2/translate", "/providers/deepl/v2/languages", "/v2/languages": try await router.deepl(request, connection)
        case "/providers/mistral/v1/ocr": try await router.mistralOCR(request, connection)
        case "/v1/realtime": try await router.realtime(request, connection)
        default: throw HTTPFailure(404, "Unknown capability route.")
        }
    }
}
