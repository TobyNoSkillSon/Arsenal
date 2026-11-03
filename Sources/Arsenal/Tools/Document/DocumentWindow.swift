// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// Read Document: one window for every model that declares `read_document`. A PDF or a page image; the pages to read
// and the table format, as the model takes them; each page's markdown as it reads (headings, paragraphs, tables), its
// images, Copy and Save as Markdown.

@MainActor final class DocumentSession: CapabilitySession {
    /// The page on show (an index into the result's pages).
    @Published var page = 0
    override func outputText(_ o: Outcome) -> String? { Self.markdown(o) }

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .document, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override var resultNoun: String { "reading" }
    override var idleLine: String { "Reading" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Read" }
    override func helps(_ op: OperationDescriptor) -> [String: String] { ["pages": "Pages to read, e.g. 1\u{2013}3, 5. Empty reads them all."] }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["include_image_base64": "Include pictures", "document_annotation_format": "Fields to extract"] }
    /// Fields to extract are an extra: only their Add shows until one is added.
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = super.makeForm(op)
        form.update("document_annotation_format") { $0.startsEmpty = true }
        return form
    }
    override func check(_ outcome: Outcome) throws {
        guard !Self.pages(outcome).isEmpty else { throw Undelivered(.init(message: "\(outcome.modelName) read no pages.", action: nil)) }
    }
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        try super.deliver(result, op: op, model: model, input: input)
        page = 0
    }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        [Figure(title: "Pages", value: outcome.map { "\(Self.pages($0).count)" }, reference: nil, help: "Pages the model read.", width: 64)]
    }

    struct Page: Equatable {
        var index: Int
        var markdown: String
        var images: [NSImage]
    }
    static func pages(_ o: Outcome) -> [Page] {
        (o.reply["pages"]?.array ?? []).compactMap { v in
            guard let p = v.object, let markdown = p["markdown"]?.string else { return nil }
            let images = (p["images"]?.array ?? []).compactMap { $0["image_base64"]?.string }.compactMap { b64 -> NSImage? in
                let clean = b64.range(of: ",").map { String(b64[$0.upperBound...]) } ?? b64
                return Data(base64Encoded: clean).flatMap(NSImage.init(data:))
            }
            return Page(index: p["index"]?.number.flatMap(\.roundedInt) ?? 0, markdown: markdown, images: images)
        }
    }
    static func markdown(_ o: Outcome) -> String { pages(o).map(\.markdown).joined(separator: "\n\n---\n\n") }
}

struct DocumentView: View {
    @ObservedObject var session: DocumentSession
    @Environment(\.palette) private var p
    var body: some View {
        CapabilityView(session: session, subjectFills: false) {
            OutcomePanel(
                session: session, what: "text of every page", copyText: DocumentSession.markdown, actions: { o in actions(o) },
                trailing: { o in
                    AnyView(
                        ResultIcon(symbol: "square.and.arrow.down", help: "Save as Markdown", id: "save-markdown", enabled: !session.running) {
                            _ = session.save(text: DocumentSession.markdown(o), name: OutcomeHistory.title(o), ext: "md")
                        })
                },
                content: { o in
                    let pages = DocumentSession.pages(o)
                    let page = pages.indices.contains(session.page) ? pages[session.page] : pages.first
                    ScrollView(.vertical) {
                        VStack(alignment: .leading, spacing: 12) {
                            if let page {
                                MarkdownBlocks(markdown: page.markdown)
                                ForEach(Array(page.images.enumerated()), id: \.offset) { _, image in
                                    Image(nsImage: image).resizable().scaledToFit().frame(maxHeight: 220).clipShape(RoundedRectangle(cornerRadius: 6))
                                }
                            }
                        }.padding(.horizontal, 16).padding(.bottom, 16)
                    }
                })
        }
    }

    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        let pages = DocumentSession.pages(o)
        if pages.count > 1 {
            BandButton(symbol: "chevron.left", help: "Previous page", enabled: session.page > 0) { session.page -= 1 }.controlID("page-previous")
            // The document's own page number; "of" counts the pages read when they are not the first ones in order.
            let number = pages.indices.contains(session.page) ? pages[session.page].index + 1 : 1
            let inOrder = pages.map(\.index) == Array(0..<pages.count)
            Text(inOrder ? "Page \(number) of \(pages.count)" : "Page \(number) \u{00b7} \(session.page + 1) of \(pages.count)")
                .font(.system(size: 12).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
            BandButton(symbol: "chevron.right", help: "Next page", enabled: session.page < pages.count - 1) { session.page += 1 }.controlID("page-next")
        }
    }
}
