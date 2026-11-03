// SPDX-License-Identifier: MIT
import ArsenalCore
import SwiftUI

// Translate: one window for every model that declares `translate`. The text, the language to translate into (and
// from, when the model takes it), context when the model takes it; this window's translations as a conversation,
// newest last, each with the language it was read as.

@MainActor final class TranslateSession: CapabilitySession {
    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .translate, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func mainKeys(_ op: OperationDescriptor) -> [String: String] { ["text": "Text to translate"] }
    override var resultNoun: String { "translation" }
    /// The text box first: an image to translate (a model that also reads one) is the other way in, under it.
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = super.makeForm(op)
        if let i = form.fields.firstIndex(where: { $0.key == "text" }), i > 0 { form.fields.insert(form.fields.remove(at: i), at: 0) }
        return form
    }
    override func outputText(_ o: Outcome) -> String? { Self.translations(o).map(\.text).joined(separator: "\n\n") }
    override var outputNoun: String { "Translation" }
    override var idleLine: String { "Translating" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Translate" }
    /// The target language is required: Translate stays off until one is chosen.
    override var canRun: Bool {
        guard super.canRun, let form else { return false }
        return form.fields.filter { $0.required }.allSatisfy { f in
            switch f.control {
            case .language, .main: return !value(f.key).text.trimmingCharacters(in: .whitespaces).isEmpty
            default: return true
            }
        }
    }
    override func check(_ outcome: Outcome) throws {
        guard !Self.translations(outcome).isEmpty else { throw Undelivered(.init(message: "\(outcome.modelName) returned no translation.", action: nil)) }
    }
    static func translations(_ o: Outcome) -> [(text: String, detected: String?)] {
        (o.reply["translations"]?.array ?? []).compactMap { v in v["text"]?.string.map { ($0, LanguageList.detected(v["detected_source_language"]?.string)) } }
    }
}

struct TranslateView: View {
    @ObservedObject var session: TranslateSession
    @Environment(\.palette) private var p
    var body: some View {
        CapabilityView(session: session) {
            // The panel ends where every tool window’s does: above the place its siblings keep for their history.
            VStack(spacing: 12) {
                ZStack {
                    RoundedRectangle(cornerRadius: 10).fill(p.sidebar.swiftUI)
                    if session.outcomes.isEmpty, !session.running {
                        Image(systemName: ToolKind.translate.symbol).font(.system(size: 40, weight: .light)).foregroundStyle(p.t4.swiftUI)
                    } else {
                        // Newest last; the list starts at the top and follows each new translation.
                        ScrollViewReader { proxy in
                            ScrollView(.vertical) {
                                LazyVStack(alignment: .leading, spacing: 18) {
                                    ForEach(session.outcomes) { o in card(o).id(o.id) }
                                }.padding(16)
                            }
                            .onChange(of: session.outcomes.count) { _, _ in if let last = session.outcomes.last { proxy.scrollTo(last.id, anchor: .bottom) } }
                        }
                    }
                    if let run = session.run, session.running {
                        ProgressOverlay(run: run, over: !session.outcomes.isEmpty, line: run.line(model: session.model?.name ?? "", idle: session.idleLine), reduceMotion: true)
                    }
                }
                Color.clear.frame(height: ImageStrip.height)
            }
        }
    }

    private func card(_ o: Outcome) -> some View {
        let translations = TranslateSession.translations(o)
        let target = o.input["target_lang"]?.string.map(LanguageList.name) ?? ""
        return VStack(alignment: .leading, spacing: 6) {
            Text(o.text).font(.system(size: 12.5)).foregroundStyle(p.t3.swiftUI).lineLimit(4).textSelection(.enabled)
            ForEach(Array(translations.enumerated()), id: \.offset) { _, t in
                Text(t.text).font(.system(size: 15)).lineSpacing(3).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
            }
            HStack(spacing: 6) {
                let from = translations.first?.detected.map(LanguageList.name) ?? o.input["source_lang"]?.string.map(LanguageList.name)
                Text([from, target].compactMap { $0?.isEmpty == false ? $0 : nil }.joined(separator: " \u{2192} ")).font(.system(size: 11.5)).foregroundStyle(p.t3.swiftUI)
                if Set(session.outcomes.map(\.model)).count > 1 { Text(o.modelName).font(.system(size: 11.5)).foregroundStyle(p.t3.swiftUI) }
                Spacer(minLength: 4)
                ResultIcon(symbol: "doc.on.doc", help: "Copy the translation", id: "copy-\(o.id)") { session.copy(text: translations.map(\.text).joined(separator: "\n\n")) }
            }
        }
        .padding(12)
        .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 10))
    }
}
