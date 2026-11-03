// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// A model's README in Get more models: the detail's Docs action opens it in a popover beside the button. The text is
// the folder's signed README, read when Docs is clicked and never cached; Markdown blocks are drawn natively (headings,
// lists, tables, code), inline styling through AttributedString. AppKit owns the popover, so no SwiftUI view state.

struct ModelReadmeButton: NSViewRepresentable {
    let title: String
    let palette: ArsenalPalette
    let load: () async -> String?

    func makeNSView(context: Context) -> ReadmeButton { ReadmeButton() }
    func updateNSView(_ button: ReadmeButton, context: Context) {
        button.palette = palette; button.load = load
        button.attributedTitle = NSAttributedString(
            string: title, attributes: [.font: NSFont.systemFont(ofSize: 12.5, weight: .medium), .foregroundColor: palette.accentText])
        button.setAccessibilityLabel(title)
    }

    final class ReadmeButton: NSButton, NSPopoverDelegate {
        var palette = ArsenalPalette.dark
        var load: () async -> String? = { nil }
        private var popover: NSPopover?
        private var loading: Task<Void, Never>?
        /// A transient popover closes on the mouse-down that clicks Docs again; that click must not reopen it.
        private var closedAt = Date.distantPast

        init() {
            super.init(frame: .zero)
            isBordered = false; bezelStyle = .inline; setButtonType(.momentaryChange)
            target = self; action = #selector(toggle)
        }
        required init?(coder: NSCoder) { fatalError("init(coder:) is not used") }
        override var intrinsicContentSize: NSSize { let s = attributedTitle.size(); return NSSize(width: ceil(s.width), height: ceil(s.height)) }
        /// The pressed state shows the alternate title; keeping it equal means the label never blanks.
        override var attributedTitle: NSAttributedString {
            didSet { attributedAlternateTitle = attributedTitle }
        }
        /// A rebuilt or collapsed row takes the button away: its popover goes with it rather than float unanchored.
        override func viewWillMove(toWindow newWindow: NSWindow?) {
            super.viewWillMove(toWindow: newWindow)
            if newWindow == nil { loading?.cancel(); loading = nil; popover?.close() }
        }

        @objc func toggle() {
            if let popover, popover.isShown { popover.performClose(nil); return }
            guard loading == nil, Date().timeIntervalSince(closedAt) > 0.3 else { return }
            loading = Task { @MainActor [weak self] in
                guard let self else { return }
                let text = await self.load()
                defer { self.loading = nil }
                guard !Task.isCancelled, self.window != nil else { return }
                let shown = NSPopover()
                shown.behavior = .transient; shown.delegate = self
                shown.appearance = NSAppearance(named: self.palette.dark ? .darkAqua : .aqua)
                // An opaque AppKit ground under the README, so the popover never shows its own material through it.
                let size = ModelReadmeView.size(text)
                let ground = FillView(frame: NSRect(origin: .zero, size: size))
                ground.color = self.palette.canvas
                let hosting = NSHostingView(rootView: ModelReadmeView(palette: self.palette, text: text))
                hosting.frame = ground.bounds; hosting.autoresizingMask = [.width, .height]
                ground.addSubview(hosting)
                let content = NSViewController()
                content.view = ground
                shown.contentViewController = content
                shown.contentSize = size
                self.popover = shown
                shown.show(relativeTo: self.bounds, of: self, preferredEdge: .maxY)
            }
        }
        func popoverDidClose(_ notification: Notification) { closedAt = Date(); popover = nil }
        var isShowingReadme: Bool { popover?.isShown == true }
        /// The shown popover's content (the opaque ground holding the README), for offscreen renders.
        var readmeContent: NSView? { popover?.contentViewController?.view }
    }
}

struct ModelReadmeView: View {
    /// The README scrolls in a fixed reading size; the unreadable case is a small popover holding its triangle.
    static func size(_ text: String?) -> NSSize { text == nil ? NSSize(width: 160, height: 64) : NSSize(width: 600, height: 640) }
    let palette: ArsenalPalette
    /// nil: the README could not be read.
    let text: String?

    var body: some View {
        Group {
            if let text {
                ScrollView {
                    VStack(alignment: .leading, spacing: 10) {
                        ForEach(Array(MarkdownBlock.parse(text).enumerated()), id: \.offset) { _, block in blockView(block) }
                    }
                    .textSelection(.enabled)
                    .padding(20).frame(maxWidth: .infinity, alignment: .leading)
                }
            } else {
                Image(systemName: "exclamationmark.triangle.fill").foregroundStyle(palette.amber.swiftUI).help("The README could not be read.")
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
            }
        }
        .frame(width: Self.size(text).width, height: Self.size(text).height)
        .background(palette.canvas.swiftUI)
    }

    private var p: ArsenalPalette { palette }

    @ViewBuilder private func blockView(_ block: MarkdownBlock) -> some View {
        switch block {
        case .heading(let level, let line):
            Text(Self.inline(line)).font(.system(size: level == 1 ? 20 : level == 2 ? 15 : 13, weight: .semibold)).foregroundStyle(p.t1.swiftUI)
                .padding(.top, level == 1 ? 0 : 8)
        case .paragraph(let line):
            Text(Self.inline(line)).font(.system(size: 13)).foregroundStyle(p.t1.swiftUI).fixedSize(horizontal: false, vertical: true)
        case .bullets(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    HStack(alignment: .firstTextBaseline, spacing: 8) {
                        Text("\u{2022}").foregroundStyle(p.t3.swiftUI)
                        Text(Self.inline(item)).foregroundStyle(p.t1.swiftUI).fixedSize(horizontal: false, vertical: true)
                    }.font(.system(size: 13))
                }
            }
        case .numbered(let items):
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.offset) { _, item in
                    Text(Self.inline(item)).font(.system(size: 13)).foregroundStyle(p.t1.swiftUI).fixedSize(horizontal: false, vertical: true)
                }
            }
        case .code(let code):
            ScrollView(.horizontal, showsIndicators: false) {
                Text(code).font(.system(size: 11.5, design: .monospaced)).foregroundStyle(p.t1.swiftUI).padding(10)
            }
            .background(RoundedRectangle(cornerRadius: 6).fill(p.raised.swiftUI))
        case .table(let rows):
            Grid(alignment: .leadingFirstTextBaseline, horizontalSpacing: 14, verticalSpacing: 6) {
                ForEach(Array(rows.enumerated()), id: \.offset) { index, row in
                    GridRow {
                        ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                            Text(Self.inline(cell)).font(.system(size: 12, weight: index == 0 ? .semibold : .regular))
                                .foregroundStyle((index == 0 ? p.t2 : p.t1).swiftUI).fixedSize(horizontal: false, vertical: true)
                        }
                    }
                    if index == 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                }
            }
        }
    }

    static func inline(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }
}

/// The Markdown subset model READMEs use, one block per element.
enum MarkdownBlock: Equatable {
    case heading(Int, String)
    case paragraph(String)
    case bullets([String])
    /// A numbered list keeps its numbers as written.
    case numbered([String])
    case code(String)
    case table([[String]])

    static func parse(_ text: String) -> [MarkdownBlock] {
        var blocks: [MarkdownBlock] = []
        var paragraph: [String] = [], bullets: [String] = [], numbered = false, table: [[String]] = [], code: [String]?
        func flush() {
            if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: " "))); paragraph = [] }
            if !bullets.isEmpty { blocks.append(numbered ? .numbered(bullets) : .bullets(bullets)); bullets = [] }
            if !table.isEmpty { blocks.append(.table(table)); table = [] }
        }
        for raw in text.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if var lines = code {
                if line.hasPrefix("```") { blocks.append(.code(lines.joined(separator: "\n"))); code = nil } else { lines.append(raw); code = lines }
                continue
            }
            if line.hasPrefix("```") { flush(); code = []; continue }
            if line.hasPrefix("<!--") { continue }
            if line.isEmpty { flush(); continue }
            if line.hasPrefix("#") {
                let level = line.prefix { $0 == "#" }.count
                if level <= 6, line.dropFirst(level).first == " " { flush(); blocks.append(.heading(level, String(line.dropFirst(level + 1)))); continue }
            }
            if line.hasPrefix("|") {
                if !paragraph.isEmpty || !bullets.isEmpty { flush() }
                var row = Substring(line).dropFirst()
                if row.hasSuffix("|") { row = row.dropLast() }
                let cells = row.components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
                // The header separator row (|---|:--|) draws as the divider instead.
                if !cells.allSatisfy({ !$0.isEmpty && $0.allSatisfy { "-: ".contains($0) } }) { table.append(cells) }
                continue
            }
            if line.hasPrefix("- ") || line.hasPrefix("* ") {
                if !paragraph.isEmpty || !table.isEmpty || numbered && !bullets.isEmpty { flush() }
                numbered = false; bullets.append(String(line.dropFirst(2)))
                continue
            }
            if let dot = line.firstIndex(where: { $0 == "." || $0 == ")" }), dot > line.startIndex, line[..<dot].allSatisfy(\.isNumber),
                line[line.index(after: dot)...].hasPrefix(" ")
            {
                if !paragraph.isEmpty || !table.isEmpty || !numbered && !bullets.isEmpty { flush() }
                numbered = true; bullets.append(String(line))
                continue
            }
            if !bullets.isEmpty, raw.hasPrefix("  ") { bullets[bullets.count - 1] += " " + line; continue }
            if !bullets.isEmpty || !table.isEmpty { flush() }
            paragraph.append(line)
        }
        if let code { blocks.append(.code(code.joined(separator: "\n"))) }
        flush()
        return blocks
    }
}
