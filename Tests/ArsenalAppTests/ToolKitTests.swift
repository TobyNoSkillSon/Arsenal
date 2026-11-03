// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import Testing
@testable import Arsenal

/// The tool kit: every tool window is composed from the kit's pieces rather than its own copies,
/// and the pieces' shared rules hold for every tool.
@MainActor @Suite struct ToolKitTests {
    private static let tools = ImageFixture.repo.appendingPathComponent("Sources/Arsenal/Tools")

    /// Each tool window draws the kit's frame, options, run button and result panel, and none declares its own copy of a
    /// kit piece (a model pop-up, a field control, a slider, a progress overlay).
    @Test func everyToolWindowIsAKitComposition() throws {
        let windows = ["Image/ImageView.swift", "Sound/SoundView.swift", "Background/BackgroundView.swift", "Operation/OperationView.swift"]
        #expect(windows.count == 4)
        for name in windows {
            let source = try String(contentsOf: Self.tools.appendingPathComponent(name), encoding: .utf8)
            for piece in ["ToolFrame(", "SessionBand(", "OptionsColumn(", "RunButton(", "ResultPanel("] {
                #expect(source.contains(piece), "\(name) does not use \(piece)")
            }
            for copy in ["Menu {", "struct .*FieldControl", "Slider(", "ProgressView(", "TextEditor(", "NSOpenPanel("] {
                #expect(source.range(of: copy, options: .regularExpression) == nil, "\(name) has its own \(copy)")
            }
        }
        // Each session is the kit's.
        for name in ["Image/ImageSession.swift", "Sound/SoundSession.swift", "Background/BackgroundSession.swift", "Operation/OperationSession.swift"] {
            let source = try String(contentsOf: Self.tools.appendingPathComponent(name), encoding: .utf8)
            #expect(source.contains(": ToolSession {"), "\(name) is not a ToolSession")
            #expect(!source.contains("activity/stream"), "\(name) watches the activity stream itself")
        }
    }

    /// The pills: the tool's own operations first, the rest in the order the model declares them.
    @Test func operationsKeepTheModelsOrderAfterTheToolsOwn() {
        func op(_ name: String, _ title: String) -> OperationDescriptor {
            OperationDescriptor(name: name, title: title, inputSchema: ["type": "object", "properties": ["prompt": ["type": "string"]]], output: "audio", stream: false)
        }
        let session = SoundSession(models: SoundFixture.models, selected: "stable-audio-3-medium", client: { nil }, inbox: URL(fileURLWithPath: "/tmp"))
        let ops = [op("zeta", "Zeta"), op("generate_music", "Generate Music"), op("alpha", "Alpha"), op("generate_sound", "Generate Sound")]
        #expect(session.ordered(ops).map(\.name) == ["generate_sound", "generate_music", "zeta", "alpha"])
        #expect(OperationPills.title("Generate Music") == "Music" && OperationPills.title("Alpha Matte") == "Alpha Matte")
    }

    /// One progress line for every tool: a runner's label leads its count; a tool whose idle line says what it does keeps it.
    @Test func progressLinesReadTheSameInEveryTool() {
        var run = ToolRun(model: "m", operation: "o")
        #expect(run.line(model: "M") == "Starting" && run.line(model: "M", idle: "Removing background") == "Removing background")
        run.step = 3; run.total = 8
        #expect(run.line(model: "M") == "Step 3 of 8" && run.line(model: "M", idle: "Removing background") == "Removing background \u{00b7} 3 of 8")
        run.label = "Separating layers"; run.unit = "layers"
        #expect(run.line(model: "M") == "Separating layers \u{00b7} 3 of 8 layers")
        run.total = 1
        #expect(run.line(model: "M") == "Separating layers")
        run.phase = "loading"
        #expect(run.line(model: "FLUX.2 Klein 4B") == "Loading FLUX.2 Klein 4B")
    }

    /// A schema-2 operation that made several files answers multipart/mixed: each part is a result file of its type.
    @Test func severalFilesComeBackAsMultipart() throws {
        let png = try #require(Data(base64Encoded: "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+j6ioAAAAASUVORK5CYII="))
        var body = Data()
        for i in 0..<2 {
            body += Data("--b1\r\nContent-Type: image/png\r\nContent-Disposition: attachment; name=\"output-\(i)\"\r\n\r\n".utf8) + png + Data("\r\n".utf8)
        }
        body += Data("--b1--\r\n".utf8)
        #expect(ToolFiles.boundary("multipart/mixed; boundary=b1") == "b1")
        let parts = ToolFiles.parts(body, boundary: "b1")
        #expect(parts.count == 2)
        #expect(parts.allSatisfy { $0.type == "image/png" && $0.body == png })
    }

    /// Options lead with what the operation works on: a file field comes before the short settings.
    @Test func filesLeadTheOptions() {
        let form = ToolForm(schema: LayersFixture.input)
        #expect(form.fields.map(\.key) == ["image", "layers", "fill_hidden"])
        #expect(form.fields.filter(OptionFields.isContent).map(\.key) == ["image"])
    }

    /// A model that detected no language ("und") shows none, never "Unknown language".
    @Test func undetectedLanguageIsNeverShownAsACode() {
        for placeholder in ["und", "UND", "unknown", "", " ", nil] { #expect(LanguageList.detected(placeholder) == nil, "\(placeholder ?? "nil")") }
        #expect(LanguageList.detected("pl") == "pl" && LanguageList.name("pl") == "Polish")
    }
}
