// SPDX-License-Identifier: MIT
import ArsenalCore
import SwiftUI

// Extract: one window for every model that declares `extract`. Text, an image or a document (as the model takes
// them); the fields the answer must have (a JSON Schema built as named fields) or labels to find; the answer as JSON,
// any way it breaks the fields asked for stated in red, and found entities with their labels in colour.

@MainActor final class ExtractSession: CapabilitySession {
    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .extract, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func mainKeys(_ op: OperationDescriptor) -> [String: String] { ["text": "Text to read"] }
    override var resultNoun: String { "extraction" }
    override func outputText(_ o: Outcome) -> String? { Self.data(o).flatMap(OperationForm.pretty) }
    override var outputNoun: String { "Extraction" }
    override func groups(_ op: OperationDescriptor) -> [OptionGroup] {
        [
            OptionGroup(id: "source", options: [.init(title: "Text", keys: ["text"]), .init(title: "File", keys: ["image", "document"])]),
            OptionGroup(id: "want", options: [.init(title: "Fields", keys: ["json_schema"]), .init(title: "Labels", keys: ["labels"])])
        ]
    }
    override var idleLine: String { "Extracting" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Extract" }
    /// Something to read (text, an image or a document) and something to look for (fields or labels): the operation's
    /// two `anyOf` groups.
    override var readyToRun: Bool {
        guard let op = descriptor else { return false }
        let source = ToolForm.alternatives(op.inputSchema)
        let wants = (op.inputSchema["allOf"]?.array ?? []).compactMap { $0.object }.map(ToolForm.alternatives)
        return (source.isEmpty || source.contains { $0.allSatisfy(filled) }) && wants.allSatisfy { group in group.contains { $0.allSatisfy(filled) } }
    }

    /// The answer's data, and how it breaks the schema the run asked for (nil when it follows it or none was asked).
    static func data(_ o: Outcome) -> JSONValue? { o.reply["data"] }
    static func violation(_ o: Outcome) -> String? {
        guard let schema = o.input["json_schema"]?.object else { return nil }
        guard let data = data(o) else { return "The answer has no data for the fields asked for." }
        do { try OperationSchemaValidator.validate(data, schema: .object(schema)); return nil } catch { return "Does not match the fields: \(error.localizedDescription)" }
    }
    struct Entity: Equatable { var label: String; var text: String; var score: Double }
    static func entities(_ o: Outcome) -> [Entity] {
        (o.reply["entities"]?.array ?? []).compactMap { v in
            guard let label = v["label"]?.string, let text = v["text"]?.string else { return nil }
            return Entity(label: label, text: text, score: v["score"]?.number ?? 0)
        }
    }
}

struct ExtractView: View {
    @ObservedObject var session: ExtractSession
    @Environment(\.palette) private var p
    var body: some View {
        CapabilityView(session: session, subjectFills: false, subjectHeight: 120) {
            OutcomePanel(
                session: session, what: "answer", copyText: { o in ExtractSession.data(o).flatMap(OperationForm.pretty) ?? "" },
                content: { o in
                    VStack(alignment: .leading, spacing: 10) {
                        if let violation = ExtractSession.violation(o) {
                            Label(violation, systemImage: "exclamationmark.triangle.fill").font(.system(size: 12)).foregroundStyle(p.red.swiftUI)
                                .fixedSize(horizontal: false, vertical: true).padding(.horizontal, 16)
                        }
                        let entities = ExtractSession.entities(o)
                        if !entities.isEmpty { entityList(entities) }
                        if let data = ExtractSession.data(o) { JSONView(value: data) }
                    }
                })
        }
    }

    private func entityList(_ entities: [ExtractSession.Entity]) -> some View {
        let labels = entities.map(\.label).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        return ScrollView(.vertical) {
            FlowLayout(spacing: 6) {
                ForEach(Array(entities.enumerated()), id: \.offset) { _, e in
                    HStack(spacing: 5) {
                        Text(e.label).font(.system(size: 11, weight: .semibold)).foregroundStyle(Color.white).padding(.horizontal, 5).frame(height: 18)
                            .background(SpatialColours.colour(e.label, in: labels), in: RoundedRectangle(cornerRadius: 4))
                        Text(e.text).font(.system(size: 13)).textSelection(.enabled)
                        Text("\(Int((e.score * 100).rounded()))%").font(.system(size: 11).monospacedDigit()).foregroundStyle(p.t3.swiftUI)
                    }
                    .padding(.leading, 3).padding(.trailing, 7).frame(height: 24)
                    .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 6))
                }
            }.padding(.horizontal, 16)
        }.frame(maxHeight: 180)
    }
}
