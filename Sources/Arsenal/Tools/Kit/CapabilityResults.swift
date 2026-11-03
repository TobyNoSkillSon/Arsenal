// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// The tool kit's result pieces for capability windows: the panel with the result's actions and this window's earlier
// results under it (thumbnails for pictures, a line each otherwise), a transcript with its speakers in colour and its
// times, a list of stems that play, a ranked list with its scores, a document's markdown with its tables, and JSON.

/// The result side of a capability window: the shown result with its actions above it, the history under the panel
/// (its place kept from the first frame).
struct OutcomePanel<Actions: View, Content: View>: View {
    @ObservedObject var session: CapabilitySession
    /// "transcript", "stems": Copy's object.
    var what = "result"
    var copyText: ((Outcome) -> String)?
    /// False where each item has its own Copy.
    var copies = true
    @ViewBuilder let actions: (Outcome) -> Actions
    /// Actions beside Copy (Save as text, Open).
    var trailing: ((Outcome) -> AnyView)?
    @ViewBuilder let content: (Outcome) -> Content
    /// The corner Copy's column: its glyph (13 pt in a 28 pt button, 10 pt from the edge) and 4 pt of air.
    static var copyGutter: CGFloat { 34 }

    var body: some View {
        VStack(spacing: 12) {
            ResultPanel(session: session, symbol: session.tool.symbol, empty: session.current == nil, highlighted: session.dropTargeted && !session.fileFields.isEmpty) {
                VStack(spacing: 0) {
                    if let outcome = session.current {
                        let copy = {
                            if let copyText {
                                session.copy(text: copyText(outcome))
                            } else if let file = session.copyFile(outcome) {
                                session.copy(file, to: session.copyPasteboard)
                            }
                        }
                        // A result whose Copy takes a file can be saved as that file too.
                        let file = copies && copyText == nil && session.outputText(outcome) == nil ? session.copyFile(outcome) : nil
                        let save = file.map { url in { session.saveAs(url) } }
                        if Actions.self == EmptyView.self, trailing == nil, save == nil {
                            // Nothing but Copy: it sits in the content's corner, not on a row of its own, in a gutter the
                            // content leaves free (a score, a list's last column, never under it).
                            content(outcome).padding(.trailing, copies ? Self.copyGutter : 0).frame(maxWidth: .infinity, maxHeight: .infinity).padding(.top, 10)
                                .overlay(alignment: .topTrailing) {
                                    if copies {
                                        ResultIcon(symbol: "doc.on.doc", help: "Copy the \(what)", id: "copy", enabled: !session.running, action: copy)
                                            .padding(.top, 7).padding(.trailing, 10)
                                    }
                                }
                        } else {
                            ResultActions(
                                what: what, enabled: !session.running, copy: copy, leading: { actions(outcome) }, trailing: trailing?(outcome), showsCopy: copies, save: save)
                            content(outcome).frame(maxWidth: .infinity, maxHeight: .infinity)
                        }
                    } else {
                        Color.clear
                    }
                }
                .contentShape(Rectangle())
                .onDrop(of: [.fileURL, .image], isTargeted: $session.dropTargeted) { session.fileFields.isEmpty ? false : session.receive($0) }
            }
            // From the second result: a lone item reads as a stray tag, not as history (its place is kept all along).
            OutcomeHistory(session: session).historySlot(OutcomeHistory.shows(session.outcomes.count))
        }
    }
}

extension OutcomePanel where Actions == EmptyView {
    init(
        session: CapabilitySession, what: String = "result", copyText: ((Outcome) -> String)? = nil, copies: Bool = true,
        @ViewBuilder content: @escaping (Outcome) -> Content
    ) {
        self.init(session: session, what: what, copyText: copyText, copies: copies, actions: { _ in EmptyView() }, trailing: nil, content: content)
    }
}

/// This window's results: pictures as thumbnails, everything else as a line (what it was made from, the model).
struct OutcomeHistory: View {
    @ObservedObject var session: CapabilitySession
    @Environment(\.palette) private var p
    /// The strip shows once there are two results to choose between.
    static func shows(_ count: Int) -> Bool { count >= 2 }

    var body: some View {
        if session.outcomes.contains(where: { session.historyPicture($0) != nil }) {
            ImageStrip(
                items: session.outcomes.compactMap { o in session.historyPicture(o).map { .init(id: o.id, url: $0, label: Self.title(o), controlID: "outcome-\(o.id)") } },
                current: session.current?.id, enabled: !session.running, select: { session.show($0) })
        } else {
            ScrollView(.horizontal) {
                HStack(spacing: 6) {
                    ForEach(session.outcomes) { o in
                        Pill(title: Self.title(o), on: session.current?.id == o.id, enabled: !session.running, help: Self.title(o)) { session.show(o.id) }
                            .controlID("outcome-\(o.id)")
                    }
                }
            }.scrollIndicators(.never).frame(height: ImageStrip.height)
        }
    }
    /// The picture a result shows best as: its own image, else the picture it was made from.
    static func picture(_ o: Outcome) -> URL? {
        o.file("image/")?.url ?? o.inputs.first(where: ToolFiles.isImageFile)
    }
    /// "beach.wav", or the start of the text it was made from.
    static func title(_ o: Outcome) -> String {
        if let input = o.inputs.first { return input.deletingPathExtension().lastPathComponent }
        let text = o.text.split(whereSeparator: \.isNewline).first.map(String.init) ?? o.operationTitle
        return text.count > 28 ? String(text.prefix(27)) + "\u{2026}" : text
    }
}

// MARK: Transcript

/// One stretch of speech: its times, its speaker and its words.
struct TranscriptSegment: Equatable, Identifiable {
    var id: Int
    var start: Double
    var end: Double
    var speaker: String?
    var text: String

    /// The reply's segments; else its words grouped into lines (a new line at a new speaker or a pause over 0.8 s);
    /// else its text as one line.
    static func list(_ reply: JSONObject) -> [TranscriptSegment] {
        let segments = (reply["segments"]?.array ?? []).enumerated().compactMap { i, v -> TranscriptSegment? in
            guard let s = v.object, let start = s["start"]?.number, let end = s["end"]?.number else { return nil }
            return TranscriptSegment(id: i, start: start, end: end, speaker: s["speaker"]?.string, text: s["text"]?.string?.trimmingCharacters(in: .whitespaces) ?? "")
        }
        if !segments.isEmpty { return segments }
        var lines: [TranscriptSegment] = []
        for v in reply["words"]?.array ?? [] {
            guard let w = v.object, let word = w["word"]?.string, let start = w["start"]?.number, let end = w["end"]?.number else { continue }
            let speaker = w["speaker"]?.string
            if var last = lines.last, last.speaker == speaker, start - last.end < 0.8 {
                last.text += (word.hasPrefix(" ") || last.text.isEmpty ? "" : " ") + word.trimmingCharacters(in: .whitespaces)
                last.end = end
                lines[lines.count - 1] = last
            } else {
                lines.append(TranscriptSegment(id: lines.count, start: start, end: end, speaker: speaker, text: word.trimmingCharacters(in: .whitespaces)))
            }
        }
        if !lines.isEmpty { return lines }
        // Text without times (a live transcript, a model that returns none): a line each, untimed.
        if let text = reply["text"]?.string, !text.isEmpty {
            return text.split(whereSeparator: \.isNewline).enumerated().map { TranscriptSegment(id: $0.offset, start: -1, end: -1, speaker: nil, text: String($0.element)) }
        }
        return []
    }

    /// Speakers in the order they first speak.
    static func speakers(_ segments: [TranscriptSegment], reply: JSONObject) -> [String] {
        var names = reply["speakers"]?.array?.compactMap(\.string) ?? []
        for s in segments { if let speaker = s.speaker, !names.contains(speaker) { names.append(speaker) } }
        return names
    }

    /// "SPEAKER_00" → "Speaker 1"; any other name as it is.
    static func speakerName(_ id: String, in names: [String]) -> String {
        if id.range(of: "^(speaker|SPEAKER)[ _-]?[0-9]+$", options: .regularExpression) != nil, let i = names.firstIndex(of: id) { return "Speaker \(i + 1)" }
        return id
    }

    /// "0:42" or "1:02:03".
    static func time(_ s: Double) -> String {
        guard s.isFinite, s >= 0 else { return "\u{2014}" }
        let whole = Int(s)
        return whole >= 3600 ? String(format: "%d:%02d:%02d", whole / 3600, whole / 60 % 60, whole % 60) : String(format: "%d:%02d", whole / 60, whole % 60)
    }

    var timed: Bool { start >= 0 }

    /// Plain text with speakers and times ("[0:00] Speaker 1: …"), for Copy; untimed lines as they are.
    static func text(_ segments: [TranscriptSegment], speakers: [String]) -> String {
        segments.map { s in
            let who = s.speaker.map { speakerName($0, in: speakers) + ": " } ?? ""
            return s.timed ? "[\(time(s.start))] \(who)\(s.text)" : who + s.text
        }.joined(separator: "\n")
    }

    /// SubRip subtitles.
    static func srt(_ segments: [TranscriptSegment], speakers: [String]) -> String {
        func stamp(_ s: Double) -> String {
            let ms = Int((max(0, s) * 1000).rounded())
            return String(format: "%02d:%02d:%02d,%03d", ms / 3_600_000, ms / 60000 % 60, ms / 1000 % 60, ms % 1000)
        }
        return segments.enumerated().map { i, s in
            let who = s.speaker.map { speakerName($0, in: speakers) + ": " } ?? ""
            return "\(i + 1)\n\(stamp(s.start)) --> \(stamp(s.end))\n\(who)\(s.text)\n"
        }.joined(separator: "\n")
    }
}

/// Who spoke when, as one bar across the clip's length: each stretch in its speaker's colour.
struct SpeakerTimeline: View {
    let segments: [TranscriptSegment]
    let speakers: [String]
    let length: Double
    @Environment(\.palette) private var p
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(p.t4.swiftUI.opacity(0.5))
                ForEach(segments) { s in
                    let x = g.size.width * CGFloat(s.start / max(0.01, length)), w = g.size.width * CGFloat((s.end - s.start) / max(0.01, length))
                    Rectangle().fill(s.speaker.map { SpatialColours.colour($0, in: speakers) } ?? p.accent.swiftUI).frame(width: max(2, w)).offset(x: x)
                }
            }.clipShape(Capsule())
        }
        .frame(height: 8)
        .accessibilityLabel("Who spoke when")
    }
}

/// The transcript: a line per stretch with its time and speaker; a click on the time plays the clip from there.
struct TranscriptList: View {
    let segments: [TranscriptSegment]
    let speakers: [String]
    /// The clip the transcript is of (Play from a line), when the window has it.
    let audio: URL?
    @ObservedObject var player: SoundPlayer
    @Environment(\.palette) private var p

    var body: some View {
        ScrollView(.vertical) {
            LazyVStack(alignment: .leading, spacing: 10) {
                ForEach(segments) { s in row(s) }
            }
            .padding(.horizontal, 16).padding(.vertical, 4)
            .frame(maxWidth: .infinity, alignment: .leading)
        }
        .scrollIndicators(.automatic)
    }

    private func row(_ s: TranscriptSegment) -> some View {
        HStack(alignment: .firstTextBaseline, spacing: 10) {
            if s.timed {
                Button {
                    if let audio { player.play(audio, at: s.start) }
                } label: {
                    Text(TranscriptSegment.time(s.start)).font(.system(size: 11.5).monospacedDigit()).foregroundStyle((audio == nil ? p.t3 : p.accentText).swiftUI)
                        .frame(width: 44, alignment: .leading).contentShape(Rectangle())
                }.buttonStyle(.plain).disabled(audio == nil).windowHelp(audio == nil ? nil : "Play from here").controlID("segment-\(s.id)")
            }
            VStack(alignment: .leading, spacing: 2) {
                if let speaker = s.speaker {
                    HStack(spacing: 5) {
                        Circle().fill(SpatialColours.colour(speaker, in: speakers)).frame(width: 7, height: 7)
                        Text(TranscriptSegment.speakerName(speaker, in: speakers)).font(.system(size: 11.5, weight: .semibold)).foregroundStyle(p.t2.swiftUI)
                    }
                }
                if !s.text.isEmpty {
                    Text(s.text).font(.system(size: 13.5)).lineSpacing(2).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                } else {
                    Text("\(TranscriptSegment.time(s.start))\u{2013}\(TranscriptSegment.time(s.end))").font(.system(size: 12.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
                }
            }
        }
    }
}

extension SoundPlayer {
    /// Plays a clip from a time in seconds.
    func play(_ url: URL, at seconds: Double) {
        guard let length = AudioCache.shape(url)?.seconds, length > 0 else { play(url); return }
        play(url, from: seconds / length)
    }
}

// MARK: Stems

/// Sound files that play, one row each: Play, the name, its envelope, its length, Save As… and Copy; drags out as the
/// file.
struct StemList: View {
    struct Item: Identifiable {
        var id: String
        var title: String
        var url: URL
        var original = false
    }
    let items: [Item]
    @ObservedObject var player: SoundPlayer
    var copy: (URL) -> Void
    /// Save As… for one stem.
    var save: ((URL) -> Void)?
    @Environment(\.palette) private var p

    var body: some View {
        ScrollView(.vertical) {
            VStack(spacing: 6) {
                ForEach(items) { item in row(item) }
            }.padding(.horizontal, 12).padding(.bottom, 12)
        }
    }

    private func row(_ item: Item) -> some View {
        let shape = AudioCache.shape(item.url)
        return HStack(spacing: 10) {
            Button {
                player.toggle(item.url)
            } label: {
                Image(systemName: player.playing == item.url ? "stop.fill" : "play.fill").font(.system(size: 11)).foregroundStyle(item.original ? p.t1.swiftUI : Color.white)
                    .frame(width: 28, height: 28).background((item.original ? p.field : p.accent).swiftUI, in: Circle()).contentShape(Circle())
            }.buttonStyle(.plain).accessibilityLabel(player.playing == item.url ? "Stop \(item.title)" : "Play \(item.title)").controlID("play-\(item.id)")
            Text(item.title).font(.system(size: 13, weight: item.original ? .regular : .medium)).foregroundStyle((item.original ? p.t2 : p.t1).swiftUI).lineLimit(1)
                .frame(width: 110, alignment: .leading)
            // The envelope takes the width the row leaves and shrinks with it; the duration keeps its fixed trailing width
            // (a narrow window once drew the envelope over it: "0:01" read "1").
            Group {
                if let shape { WaveformView(shape: shape, url: item.url, player: player).frame(height: 30) } else { Spacer(minLength: 0) }
            }
            .frame(minWidth: 0, maxWidth: .infinity).clipped().controlID("wave-\(item.id)")
            Text(shape.map { AudioShape.clock($0.seconds) } ?? "").font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
                .lineLimit(1).fixedSize().frame(width: 38, alignment: .trailing).layoutPriority(1).controlID("duration-\(item.id)")
            if !item.original {
                if let save { SaveAsIcon(help: "Save \(item.title) As\u{2026}", id: "save-\(item.id)") { save(item.url) } }
                // No folder per stem: the band's folder opens the outputs.
                ResultIcon(symbol: "doc.on.doc", help: "Copy \(item.title)", id: "copy-\(item.id)") { copy(item.url) }
            }
        }
        .padding(.horizontal, 10).frame(height: 46)
        .background((item.original ? Color.clear : p.raised.swiftUI), in: RoundedRectangle(cornerRadius: 8))
        .onDrag { NSItemProvider(contentsOf: item.url) ?? NSItemProvider() }
    }
}

// MARK: Ranked

/// Items in their order with their score as a bar against the best one.
struct RankedList: View {
    struct Item: Identifiable {
        var id: Int
        var text: String
        var score: Double
        var image: URL?
    }
    let items: [Item]
    @Environment(\.palette) private var p

    var body: some View {
        let best = max(items.map(\.score).max() ?? 1, 1e-9)
        ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 4) {
                ForEach(Array(items.enumerated()), id: \.element.id) { rank, item in
                    HStack(alignment: .center, spacing: 10) {
                        Text("\(rank + 1)").font(.system(size: 12, weight: .semibold).monospacedDigit()).foregroundStyle(p.t3.swiftUI).frame(width: 18, alignment: .trailing)
                        if let image = item.image {
                            Thumbnail(url: image).frame(width: 40, height: 40).clipShape(RoundedRectangle(cornerRadius: 5))
                        }
                        Text(item.text).font(.system(size: 13)).lineLimit(3).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                        VStack(alignment: .trailing, spacing: 3) {
                            Text(String(format: "%.3f", item.score)).font(.system(size: 12).monospacedDigit()).foregroundStyle(p.t1.swiftUI)
                            ZStack(alignment: .leading) {
                                Capsule().fill(p.t4.swiftUI)
                                Capsule().fill(p.accent.swiftUI).frame(width: 64 * CGFloat(max(0, item.score) / best))
                            }.frame(width: 64, height: 4)
                        }.controlID("score-\(rank)")
                    }
                    .padding(.horizontal, 10).padding(.vertical, 7)
                    .background((rank == 0 ? p.accent.withAlphaComponent(0.1) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 7))
                }
            }.padding(.horizontal, 10).padding(.bottom, 12)
        }
    }
}

// MARK: Markdown

/// A page's markdown as it reads: headings, paragraphs (with inline emphasis, code and links), list items, code
/// blocks and tables as grids. No web view.
struct MarkdownBlocks: View {
    enum Block: Equatable {
        case heading(Int, String), paragraph(String), item(String), code(String), table([[String]]), rule
    }
    let markdown: String
    @Environment(\.palette) private var p

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            ForEach(Array(Self.blocks(markdown).enumerated()), id: \.offset) { _, block in view(block) }
        }
        .textSelection(.enabled)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    @ViewBuilder private func view(_ block: Block) -> some View {
        switch block {
        case .heading(let level, let text):
            Text(Self.inline(text)).font(.system(size: level == 1 ? 18 : level == 2 ? 15.5 : 13.5, weight: .semibold)).padding(.top, level <= 2 ? 4 : 0)
        case .paragraph(let text): Text(Self.inline(text)).font(.system(size: 13.5)).lineSpacing(2).fixedSize(horizontal: false, vertical: true)
        case .item(let text):
            HStack(alignment: .firstTextBaseline, spacing: 6) {
                Text("\u{2022}").foregroundStyle(p.t3.swiftUI)
                Text(Self.inline(text)).font(.system(size: 13.5)).fixedSize(horizontal: false, vertical: true)
            }
        case .code(let text):
            Text(text).font(.system(size: 12, design: .monospaced)).padding(8).frame(maxWidth: .infinity, alignment: .leading)
                .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6))
        case .table(let rows):
            ScrollView(.horizontal) {
                Grid(alignment: .leading, horizontalSpacing: 14, verticalSpacing: 5) {
                    ForEach(Array(rows.enumerated()), id: \.offset) { i, row in
                        GridRow {
                            ForEach(Array(row.enumerated()), id: \.offset) { _, cell in
                                Text(Self.inline(cell)).font(.system(size: 12.5, weight: i == 0 ? .semibold : .regular).monospacedDigit())
                            }
                        }
                        if i == 0 { Divider().gridCellUnsizedAxes(.horizontal) }
                    }
                }.padding(8)
            }
            .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 6))
        case .rule: Divider()
        }
    }

    static func inline(_ text: String) -> AttributedString {
        (try? AttributedString(markdown: text, options: .init(interpretedSyntax: .inlineOnlyPreservingWhitespace))) ?? AttributedString(text)
    }

    static func blocks(_ markdown: String) -> [Block] {
        var blocks: [Block] = [], paragraph: [String] = [], table: [[String]] = [], code: [String]?
        func flush() {
            if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: " "))); paragraph = [] }
            if !table.isEmpty { blocks.append(.table(table)); table = [] }
        }
        for raw in markdown.components(separatedBy: "\n") {
            let line = raw.trimmingCharacters(in: .whitespaces)
            if line.hasPrefix("```") {
                if let open = code { blocks.append(.code(open.joined(separator: "\n"))); code = nil } else { flush(); code = [] }
                continue
            }
            if code != nil { code?.append(raw); continue }
            if line.isEmpty { flush(); continue }
            if line.hasPrefix("|") {
                if !paragraph.isEmpty { blocks.append(.paragraph(paragraph.joined(separator: " "))); paragraph = [] }
                let cells = line.trimmingCharacters(in: CharacterSet(charactersIn: "|")).components(separatedBy: "|").map { $0.trimmingCharacters(in: .whitespaces) }
                // The header's separator row (|---|:--:|) is not a row.
                if cells.allSatisfy({ $0.range(of: "^:?-{2,}:?$", options: .regularExpression) != nil }) { continue }
                table.append(cells)
                continue
            }
            if !table.isEmpty { blocks.append(.table(table)); table = [] }
            if let hashes = line.range(of: "^#{1,6} ", options: .regularExpression) {
                flush(); blocks.append(.heading(line[hashes].count - 1, String(line[hashes.upperBound...]))); continue
            }
            if line.range(of: "^([-*+]|[0-9]+[.)]) ", options: .regularExpression) != nil {
                flush(); blocks.append(.item(String(line.drop { $0 != " " }.dropFirst()))); continue
            }
            if line.range(of: "^(-{3,}|\\*{3,}|_{3,})$", options: .regularExpression) != nil { flush(); blocks.append(.rule); continue }
            paragraph.append(line)
        }
        if let open = code { blocks.append(.code(open.joined(separator: "\n"))) }
        flush()
        return blocks
    }
}

// MARK: JSON

/// A value as indented JSON, keys quiet, values plain; selectable.
struct JSONView: View {
    let value: JSONValue
    @Environment(\.palette) private var p
    var body: some View {
        ScrollView([.vertical, .horizontal]) {
            Text(Self.styled(value, palette: p)).font(.system(size: 12, design: .monospaced)).textSelection(.enabled)
                .frame(maxWidth: .infinity, alignment: .topLeading).padding(12)
        }
    }
    static func styled(_ value: JSONValue, palette p: ArsenalPalette) -> AttributedString {
        var out = AttributedString()
        func add(_ s: String, _ colour: NSColor) {
            var a = AttributedString(s); a.foregroundColor = Color(nsColor: colour); out += a
        }
        func walk(_ v: JSONValue, _ indent: String) {
            switch v {
            case .object(let o):
                if o.isEmpty { add("{}", p.t2); return }
                add("{\n", p.t3)
                for (i, key) in o.keys.sorted().enumerated() {
                    add(indent + "  \"\(key)\"", p.t3); add(": ", p.t3)
                    walk(o[key]!, indent + "  ")
                    add(i == o.count - 1 ? "\n" : ",\n", p.t3)
                }
                add(indent + "}", p.t3)
            case .array(let a):
                if a.isEmpty { add("[]", p.t2); return }
                add("[\n", p.t3)
                for (i, item) in a.enumerated() {
                    add(indent + "  ", p.t3); walk(item, indent + "  ")
                    add(i == a.count - 1 ? "\n" : ",\n", p.t3)
                }
                add(indent + "]", p.t3)
            case .string(let s): add(OperationForm.pretty(.string(s)) ?? "\"\(s)\"", p.t1)
            default: add(OperationForm.pretty(v) ?? "null", p.accentText)
            }
        }
        walk(value, "")
        return out
    }
}
