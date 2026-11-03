// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// The generic operation window, composed from the tool kit: the band (the model's name, its operations as pills), the
// live figures, the options its operation declares with Run or Stop at their foot, and the result on the right: ordered
// files with their labels and Save, or the reply's fields. No dashboard, no instructions.

struct OperationView: View {
    @ObservedObject var session: OperationSession
    /// The kind's symbol, drawn faintly where the result will appear.
    var symbol = "square.stack.3d.up"
    static let size = ToolLayout.size, minSize = ToolLayout.minSize

    var body: some View {
        ToolFrame(figures: session.figures(), model: session.model?.model) {
            SessionBand(session: session, picks: false)
        } controls: {
            OptionsColumn(session: session, run: RunButton(session: session, title: session.descriptor?.title ?? "Run", id: "run"))
        } result: {
            ResultArea(session: session, symbol: symbol)
        }
    }
}

// MARK: Result

/// The CLT toolchain has no SwiftUI macro plugin; the property wrapper is used through its type.
private typealias MetadataOpen = SwiftUI.State<Bool>

struct ResultArea: View {
    @ObservedObject var session: OperationSession
    let symbol: String
    @Environment(\.palette) private var p
    /// The reply's own keys and JSON beside its files: folded, a click away.
    @MetadataOpen private var metadata = false

    var body: some View {
        ResultPanel(session: session, symbol: symbol, empty: session.result == nil) {
            if let result = session.result { content(result).padding(12) }
        }
    }

    @ViewBuilder private func content(_ r: OperationResult) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            // No "Done" or "4 files" header: the files say it. The reply's other keys and its JSON fold under Details.
            let folded = !r.files.isEmpty && (!r.fields.isEmpty || r.json != nil || r.files.contains { !$0.extras.isEmpty })
            HStack(spacing: 10) {
                if folded {
                    Button {
                        metadata.toggle()
                    } label: {
                        HStack(spacing: 5) {
                            Text("Details").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
                            Image(systemName: "chevron.right").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                                .rotationEffect(.degrees(metadata ? 90 : 0))
                        }.frame(height: 24).contentShape(Rectangle())
                    }.buttonStyle(.plain).controlID("result-details")
                }
                Spacer(minLength: 8)
                if r.files.count > 1 {
                    // The same save glyph as every result pane; its words are the hover and the accessibility label.
                    SaveAsIcon(help: "Save All\u{2026}", id: "save-all") { session.saveAll() }
                }
            }
            if !r.files.isEmpty, !r.fields.isEmpty, metadata {
                Text(r.fields.map(\.line).joined(separator: " \u{00b7} ")).font(.system(size: 11.5)).foregroundStyle(p.t3.swiftUI)
                    .lineLimit(2).textSelection(.enabled)
            }
            if !r.files.isEmpty {
                FileGrid(files: r.files, session: session, showsExtras: metadata)
            }
            if let json = r.json, r.files.isEmpty || metadata {
                ScrollView([.vertical, .horizontal]) {
                    Text(json).font(.system(size: 12, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .topLeading)
                }.frame(height: r.files.isEmpty ? nil : min(160, CGFloat(json.split(separator: "\n").count) * 15 + 4))
            } else if r.files.isEmpty {
                ScrollView(.vertical) {
                    Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 6) {
                        ForEach(r.fields, id: \.self) { f in
                            GridRow {
                                Text(f.label).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI)
                                Text(f.shown).font(.system(size: 13).monospacedDigit()).foregroundStyle(p.t1.swiftUI).textSelection(.enabled)
                            }
                        }
                    }.frame(maxWidth: .infinity, alignment: .topLeading)
                }
            }
        }
        .frame(maxWidth: .infinity, maxHeight: .infinity, alignment: .topLeading)
    }

}

/// The result's files in their order: images in a grid sized to their count, each with its label, its own fields and
/// Save; other files as a line each.
struct FileGrid: View {
    let files: [ResultFile]
    @ObservedObject var session: OperationSession
    /// Each file's own keys (depth_min, …): shown with the result's Details.
    var showsExtras = true
    @Environment(\.palette) private var p

    /// The least an image tile shrinks to.
    static let minTile: CGFloat = 36
    /// The narrowest column whose label and Save still read.
    static let minColumn: CGFloat = 190
    static func columns(_ n: Int) -> Int { n <= 3 ? max(1, n) : n == 4 ? 2 : n <= 6 ? 3 : 4 }

    var body: some View {
        let images = files.filter(\.isImage), others = files.filter { !$0.isImage }
        GeometryReader { g in
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 14) {
                    if !images.isEmpty {
                        // As many columns as keep a label readable (two at the window's smallest), never more than the count asks.
                        let n = min(Self.columns(images.count), max(1, Int((g.size.width + 12) / Self.minColumn))), spacing: CGFloat = 12
                        let width = max(80, (g.size.width - CGFloat(n - 1) * spacing) / CGFloat(n))
                        let rows = Int((Double(images.count) / Double(n)).rounded(.up))
                        // Fill the area's height when all rows fit; never taller than the files are wide.
                        let room = (g.size.height - CGFloat(rows - 1) * spacing) / CGFloat(rows) - (showsExtras && images.contains { !$0.extras.isEmpty } ? 50 : 32)
                        LazyVGrid(columns: Array(repeating: GridItem(.fixed(width), spacing: spacing, alignment: .top), count: n), alignment: .leading, spacing: spacing) {
                            // The image gives up height before its label and Save do (a small window keeps every row's words).
                            ForEach(images) { file in tile(file, width: width, height: max(Self.minTile, min(width / aspect(file), room))) }
                        }
                    }
                    ForEach(others) { file in line(file) }
                }
            }.scrollIndicators(.automatic)
        }
    }

    private func aspect(_ f: ResultFile) -> CGFloat {
        guard let w = f.width, let h = f.height, w > 0, h > 0 else { return 1 }
        return min(4, max(0.25, CGFloat(w) / CGFloat(h)))
    }

    private func tile(_ file: ResultFile, width: CGFloat, height: CGFloat) -> some View {
        // The checkerboard covers the image itself, not the tile: a layer's own transparency reads as such.
        let a = aspect(file)
        let w = min(width, height * a)
        return VStack(alignment: .leading, spacing: 5) {
            ZStack {
                Checkerboard()
                Thumbnail(url: file.url, fit: true)
            }
            .frame(width: w, height: w / a)
            .background(p.canvas.swiftUI)
            .clipShape(RoundedRectangle(cornerRadius: 6))
            .frame(width: w, height: height)
            .onDrag { NSItemProvider(contentsOf: file.url) ?? NSItemProvider() }
            .accessibilityLabel(file.label)
            HStack(spacing: 6) {
                // The label wins the room; the pixel size shows only where a tile has space for it.
                Text(file.label).font(.system(size: 12.5, weight: .semibold)).lineLimit(1).layoutPriority(1).windowHelp(file.label)
                if let w = file.width, let h = file.height, width >= 220 {
                    Text("\(w) \u{00d7} \(h)").font(.system(size: 11).monospacedDigit()).foregroundStyle(p.t3.swiftUI).lineLimit(1)
                }
                Spacer(minLength: 4)
                saveButton(file)
            }.frame(width: width)
            if showsExtras, !file.extras.isEmpty {
                Text(file.extras.map(\.line).joined(separator: " \u{00b7} ")).font(.system(size: 11).monospacedDigit()).foregroundStyle(p.t3.swiftUI)
                    .lineLimit(1).truncationMode(.tail).frame(width: width, alignment: .leading)
                    .windowHelp(file.extras.map { "\($0.label): \($0.shown)" }.joined(separator: "\n"))
            }
        }.frame(width: width)
    }

    private func line(_ file: ResultFile) -> some View {
        HStack(spacing: 10) {
            Image(systemName: file.isAudio ? "waveform" : "doc").font(.system(size: 15)).foregroundStyle(p.t2.swiftUI).frame(width: 22)
            VStack(alignment: .leading, spacing: 2) {
                Text(file.label).font(.system(size: 12.5, weight: .semibold)).lineLimit(1)
                Text(([formatBytes(Int64(file.bytes)), file.mime] + (showsExtras ? file.extras.map(\.line) : [])).joined(separator: " \u{00b7} "))
                    .font(.system(size: 11).monospacedDigit()).foregroundStyle(p.t3.swiftUI).lineLimit(1)
            }
            Spacer(minLength: 6)
            if file.isAudio {
                PlayButton(url: file.url, player: session.player, label: file.label).controlID("play-\(file.id)")
            }
            saveButton(file)
        }
        .padding(.horizontal, 10).padding(.vertical, 8)
        .background(p.raised.swiftUI, in: RoundedRectangle(cornerRadius: 8))
    }

    private func saveButton(_ file: ResultFile) -> some View {
        SaveAsIcon(help: "Save \(file.label) As\u{2026}", id: "save-\(file.id)") { session.save(file) }
    }
}

/// Play or stop a file through the window's player.
struct PlayButton: View {
    let url: URL
    @ObservedObject var player: SoundPlayer
    let label: String
    @Environment(\.palette) private var p
    var body: some View {
        Button {
            player.toggle(url)
        } label: {
            Image(systemName: player.playing == url ? "stop.fill" : "play.fill").font(.system(size: 12)).frame(width: 26, height: 26).contentShape(Rectangle())
        }.buttonStyle(.plain).foregroundStyle(p.t2.swiftUI).accessibilityLabel(player.playing == url ? "Stop \(label)" : "Play \(label)")
    }
}
