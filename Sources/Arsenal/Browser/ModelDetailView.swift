// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// An expanded row's details: the description, then About / Sizes / Runner side by
// side, with runners always on the right. About and Sizes stack on narrower windows; every block uses the same key column so two
// open cards line up for comparison. Values come from `ModelDetail`; the radio list calls back with a runner name.

struct ModelDetailView: View {
    let detail: ModelDetail
    let palette: ArsenalPalette
    let width: CGFloat
    let licenceAmber: Bool
    let choose: (String) -> Void
    /// A detail action (open the tool, use the previous revision).
    var act: (ModelDetail.Action) -> Void = { _ in }
    /// The model's README, read when Docs opens.
    var readme: () async -> String? = { nil }
    static let key: CGFloat = 78, about: CGFloat = 300, sizes: CGFloat = 320, runner: CGFloat = 320, gap: CGFloat = 28

    private var p: ArsenalPalette { palette }
    static func runnerWidth(_ width: CGFloat) -> CGFloat { stacked(width) ? 260 : runner }
    static func stacked(_ width: CGFloat) -> Bool { width < about + sizes + runner + 2 * gap }
    struct LayoutKey: Hashable { var stacked: Bool; var text: CGFloat }
    /// Fixed block widths keep wide live resize free of repeated SwiftUI measurement. Narrow rows remeasure
    /// only where the description can wrap; the runner remains a fixed-width right-hand column.
    static func layoutKey(_ width: CGFloat) -> LayoutKey {
        LayoutKey(stacked: stacked(width), text: min(description, width - runnerWidth(width) - gap).rounded(.down))
    }
    static let description: CGFloat = 640

    var body: some View {
        HStack(alignment: .top, spacing: Self.gap) {
            VStack(alignment: .leading, spacing: 14) {
                if !detail.description.isEmpty {
                    Text(detail.description).font(.system(size: 13)).foregroundStyle(p.t1.swiftUI)
                        .lineLimit(3).fixedSize(horizontal: false, vertical: true).frame(maxWidth: Self.description, alignment: .leading)
                }
                if Self.stacked(width) {
                    VStack(alignment: .leading, spacing: 14) {
                        aboutBlock; sizesBlock
                    }
                } else {
                    HStack(alignment: .top, spacing: Self.gap) {
                        aboutBlock; sizesBlock
                    }
                }
            }
            .frame(maxWidth: .infinity, alignment: .leading)
            BrowserRunnerList(detail: detail, palette: palette, choose: choose)
                .frame(width: Self.runnerWidth(width), alignment: .leading)
        }
        .padding(.top, 2).padding(.bottom, 14)
        .frame(maxWidth: .infinity, alignment: .leading)
    }

    private func keyText(_ k: String) -> some View {
        Text(k).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).frame(width: Self.key, alignment: .leading)
    }
    private func color(_ tone: ModelDetail.Tone) -> Color { (tone == .amber ? p.amber : tone == .secondary ? p.t2 : p.t1).swiftUI }

    private var aboutBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(detail.about, id: \.key) { line in
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    keyText(line.key)
                    (Text(line.value).foregroundStyle(color(line.tone)) + Text(line.note.map { " \u{00b7} " + $0 } ?? "").foregroundStyle(p.t2.swiftUI))
                        .font(.system(size: 12.5)).lineLimit(2).fixedSize(horizontal: false, vertical: true)
                }
            }
            if !detail.actions.isEmpty {
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    keyText("")
                    HStack(spacing: 14) {
                        ForEach(detail.actions, id: \.self) { action in
                            if action == .docs {
                                ModelReadmeButton(title: action.title, palette: p, load: readme).fixedSize()
                            } else {
                                Button {
                                    act(action)
                                } label: {
                                    Text(action.title).font(.system(size: 12.5, weight: .medium)).foregroundStyle(p.accentText.swiftUI)
                                }.buttonStyle(.plain).accessibilityLabel(action.title)
                            }
                        }
                    }
                }
            }
            if !detail.links.isEmpty {
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    keyText("Links")
                    HStack(spacing: 14) {
                        ForEach(detail.links, id: \.url) { link in
                            Button {
                                if let url = URL(string: link.url) { NSWorkspace.shared.open(url) }
                            } label: {
                                Text(link.title + " \u{2197}").font(.system(size: 12.5)).foregroundStyle(p.accentText.swiftUI)
                            }.buttonStyle(.plain).help(link.url)
                        }
                    }
                }
            }
        }.frame(maxWidth: Self.about, alignment: .leading)
    }

    private var sizesBlock: some View {
        VStack(alignment: .leading, spacing: 6) {
            ForEach(detail.sizes, id: \.dtype) { s in
                HStack(alignment: .firstTextBaseline, spacing: 0) {
                    Text(s.dtype == detail.sizes.first?.dtype ? "Sizes" : "").font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).frame(width: 44, alignment: .leading)
                    Text(s.label.isEmpty ? s.dtype : s.label).font(.system(size: 12.5)).foregroundStyle(p.t2.swiftUI).lineLimit(1).fixedSize().frame(width: 38, alignment: .leading)
                    Text(s.size).font(.system(size: 12.5).monospacedDigit()).foregroundStyle(p.t1.swiftUI).frame(width: 62, alignment: .trailing)
                    Text(s.note).font(.system(size: 12)).foregroundStyle(p.t3.swiftUI).lineLimit(1).fixedSize().padding(.leading, 8)
                }
            }
        }.frame(maxWidth: Self.sizes, alignment: .leading)
    }

}
