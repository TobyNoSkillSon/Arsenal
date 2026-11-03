// SPDX-License-Identifier: MIT
import ArsenalCore
import SwiftUI

// Separate Audio: one window for every model that declares `separate_audio`. A clip; what to pull out, as the model
// takes it (fixed stems, a description of the sound, extract or remove); each returned stem as a row that plays, with
// its envelope, Copy and Show in Finder, the original first to compare. Stems are saved to ~/Music/Arsenal.

@MainActor final class SeparateSession: CapabilitySession {
    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .separate, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func mainKeys(_ op: OperationDescriptor) -> [String: String] { ["prompt": "Describe the sound to pull out"] }
    override var resultNoun: String { "separation" }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["mode": "Keep"] }
    /// Extract or remove, said as what the result keeps.
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = super.makeForm(op)
        form.update("mode") { $0.choiceTitles = ["extract": "Only these sounds", "remove": "Everything else"] }
        return form
    }
    override func groups(_ op: OperationDescriptor) -> [OptionGroup] {
        [OptionGroup(id: "what", options: [.init(title: "Stems", keys: ["stems"]), .init(title: "Describe", keys: ["prompt"])])]
    }
    override var idleLine: String { "Separating" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Separate" }
    override func check(_ outcome: Outcome) throws {
        guard outcome.files.contains(where: { $0.isAudio }) else { throw Undelivered(.init(message: "\(outcome.modelName) returned no audio.", action: nil)) }
    }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        [Figure(title: "Stems", value: outcome.map { "\($0.files.filter(\.isAudio).count)" }, reference: nil, help: "Sounds the model returned.", width: 64)]
    }
}

struct SeparateView: View {
    @ObservedObject var session: SeparateSession
    var body: some View {
        CapabilityView(session: session, subjectFills: false) {
            // Each stem has its own Save As…, Copy and Show in Finder; the header saves them all into a folder.
            OutcomePanel(
                session: session, what: "stem", copies: false, actions: { _ in EmptyView() },
                trailing: { o in
                    AnyView(
                        SaveAsIcon(enabled: !session.running, help: "Save All\u{2026}", id: "save-all") {
                            session.saveAll(o.files.filter(\.isAudio).map { ($0.url, $0.suggestedName) })
                        })
                },
                content: { o in
                    StemList(
                        items: (o.inputs.first.map { [StemList.Item(id: "original", title: "Original", url: $0, original: true)] } ?? [])
                            + o.files.filter(\.isAudio).map { StemList.Item(id: $0.id, title: $0.label, url: $0.url) },
                        player: session.player, copy: { session.copy($0, to: session.copyPasteboard) }, save: { session.saveAs($0) })
                })
        }
    }
}
