// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// The browser's sidebar: band (traffic lights' line, collapse button), search, sort, the
// kind list, the modality grid, Fits and the facet sections, the count line pinned at the bottom. Every section keeps
// its place: choosing a value changes fills, colours and the count text only.

struct BrowserSidebar: View {
    @ObservedObject var controller: BrowserController
    @ObservedObject var appearance = Appearance.shared
    let search: NSSearchField
    let band: CGFloat
    /// The default width; the divider drags it between `minWidth` and `maxWidth`.
    static let width: CGFloat = 236, minWidth: CGFloat = 208, maxWidth: CGFloat = 300
    /// The fade over the bottom of the scrolling sections (no wider than their bottom padding plus a little).
    static let fade: CGFloat = 22

    private var p: ArsenalPalette { appearance.palette }
    private var state: BrowserState { controller.state }

    var body: some View {
        VStack(alignment: .leading, spacing: 0) {
            // The band: the root view's hide-sidebar button sits over its trailing edge.
            Color.clear.frame(height: band).contentShape(Rectangle()).gesture(WindowDragGesture())
            ScrollView(.vertical) {
                VStack(alignment: .leading, spacing: 0) {
                    SearchFieldHost(field: search, palette: p).frame(height: 30).padding(.horizontal, 12).padding(.top, 4)
                    sortRow
                    hairline
                    group(.kinds) { kindList }
                    hairline
                    group(.modality) { modalityGrid }
                    hairline
                    facets
                }.padding(.bottom, 12)
            }.scrollIndicators(.automatic)
                // Sections that run past the count line fade out under it instead of being cut through a line of text.
                .mask {
                    VStack(spacing: 0) {
                        Color.black
                        LinearGradient(colors: [.black, .black.opacity(0)], startPoint: .top, endPoint: .bottom).frame(height: Self.fade)
                    }
                }
            Rectangle().fill(p.hairline.swiftUI).frame(height: 1).padding(.horizontal, 12)
            HStack {
                let line = controller.footer
                if line.error {
                    // A refused action, in red until the next action or a click on it.
                    Button {
                        controller.dismissError()
                    } label: {
                        Text(line.text).font(.system(size: 11.5)).foregroundStyle(p.red.swiftUI).lineLimit(4)
                            .fixedSize(horizontal: false, vertical: true).multilineTextAlignment(.leading)
                            .frame(maxWidth: .infinity, alignment: .leading)
                    }.buttonStyle(.plain).help((line.help ?? line.text) + "\nClick to dismiss.")
                } else {
                    Text(line.text).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t3.swiftUI)
                    Spacer(minLength: 8)
                }
                // Keeps its place for the count; an error line takes the room unless filters are on.
                if !line.error || state.filtering {
                    Button("Clear") { controller.state.clearFilters() }.buttonStyle(.plain)
                        .font(.system(size: 11.5, weight: .medium)).foregroundStyle(p.accentText.swiftUI)
                        .opacity(state.filtering ? 1 : 0).disabled(!state.filtering)
                }
            }.padding(.horizontal, 20).padding(.vertical, 6).frame(minHeight: 34)
        }
        // The sidebar is dragged between 208 and 300 pt: its content follows the host's width.
        .frame(maxWidth: .infinity)
        .ignoresSafeArea()
        .environment(\.colorScheme, appearance.colorScheme)
    }

    /// A section that folds under its name, as the facets do (with many kinds the sidebar ran long).
    /// Folded, the kind list keeps the chosen kind's row, so the filter in force stays visible.
    private func group<Content: View>(_ g: SidebarGroup, @ViewBuilder content: () -> Content) -> some View {
        let open = !controller.foldedGroups.contains(g)
        // Folded Kind keeps the chosen kind's row in view, so only the grid needs the dot.
        let active = g == .modality && (!state.inputs.isEmpty || !state.outputs.isEmpty)
        return VStack(alignment: .leading, spacing: 7) {
            Button {
                controller.toggleFold(g)
            } label: {
                HStack(spacing: 5) {
                    Text(g.title).font(.system(size: 12, weight: .medium)).foregroundStyle(p.t2.swiftUI)
                    if active && !open { Circle().fill(p.accentText.swiftUI).frame(width: 5, height: 5) }
                    Spacer()
                    Image(systemName: open ? "chevron.down" : "chevron.right").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                }.contentShape(Rectangle())
            }.buttonStyle(.plain).padding(.horizontal, 20)
            if open {
                content()
            } else if g == .kinds, let kind = state.kind, let row = controller.kindRows.first(where: { $0.kind.id == kind }) {
                kindRow(kind, title: row.kind.title, icon: row.kind.icon ?? KindIcon.fallback(kind), count: row.count).padding(.horizontal, 10)
            }
        }
    }

    private var hairline: some View { Rectangle().fill(p.hairline.swiftUI).frame(height: 1).padding(.horizontal, 12).padding(.vertical, 7) }

    private var sortRow: some View {
        HStack(spacing: 6) {
            Text("Sort").font(.system(size: 12)).foregroundStyle(p.t3.swiftUI)
            Spacer()
            Menu {
                ForEach(SortKey.menu, id: \.self) { key in
                    Button {
                        controller.state.sort = BrowserSort(key: key, ascending: true)
                    } label: {
                        if state.sort.key == key { Label(key.title, systemImage: "checkmark") } else { Text(key.title) }
                    }
                }
                Divider()
                Button {
                    controller.state.sort.ascending = true
                } label: {
                    if state.sort.ascending { Label("Ascending", systemImage: "checkmark") } else { Text("Ascending") }
                }
                Button {
                    controller.state.sort.ascending = false
                } label: {
                    if !state.sort.ascending { Label("Descending", systemImage: "checkmark") } else { Text("Descending") }
                }
            } label: {
                HStack(spacing: 5) {
                    Text(state.sort.key.title).font(.system(size: 12.5, weight: .medium)).foregroundStyle(p.t1.swiftUI)
                    Image(systemName: "chevron.up.chevron.down").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                }
            }.menuStyle(.button).buttonStyle(.plain).menuIndicator(.hidden).fixedSize()
        }.padding(.horizontal, 20).frame(height: 30).padding(.top, 4)
    }

    private var kindList: some View {
        VStack(spacing: 1) {
            kindRow(nil, title: "All", icon: "square.grid.2x2", count: controller.items.count)
            ForEach(controller.kindRows, id: \.kind.id) { row in
                kindRow(row.kind.id, title: row.kind.title, icon: row.kind.icon ?? KindIcon.fallback(row.kind.id), count: row.count)
            }
        }.padding(.horizontal, 10)
    }
    private func kindRow(_ id: String?, title: String, icon: String, count: Int) -> some View {
        let on = state.kind == id
        return Button {
            if let id { controller.state.choose(kind: id) } else { controller.state.kind = nil }
        } label: {
            HStack(spacing: 9) {
                Image(systemName: icon).font(.system(size: 12.5)).foregroundStyle((on ? p.accentText : p.t3).swiftUI).frame(width: 18)
                Text(title).font(.system(size: 13, weight: on ? .semibold : .regular)).foregroundStyle((on ? p.t1 : p.t2).swiftUI)
                Spacer()
                Text("\(count)").font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t3.swiftUI)
            }
            .padding(.horizontal, 10).frame(height: 26)
            .background(RoundedRectangle(cornerRadius: 6).fill(on ? p.selFill.swiftUI : .clear))
            .overlay(alignment: .leading) { if on { RoundedRectangle(cornerRadius: 1).fill(p.accent.swiftUI).frame(width: 2.5, height: 15).offset(x: 1) } }
            .contentShape(Rectangle())
        }.buttonStyle(.plain)
    }

    private var modalityGrid: some View {
        // Cells share the width left after the In/Out labels (38 pt at the default 236 pt, 30 pt at the narrowest).
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 4) {
                Color.clear.frame(width: 26, height: 1)
                ForEach(Modality.allCases) { m in
                    Text(m.title).font(.system(size: 10)).foregroundStyle(p.t3.swiftUI).lineLimit(1).minimumScaleFactor(0.8).frame(maxWidth: .infinity)
                }
            }
            ForEach([true, false], id: \.self) { input in
                HStack(spacing: 4) {
                    Text(input ? "In" : "Out").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI).frame(width: 26, alignment: .leading)
                    ForEach(Modality.allCases) { m in modalityCell(m, input: input) }
                }
            }
        }.padding(.horizontal, 20)
    }
    private func modalityCell(_ m: Modality, input: Bool) -> some View {
        let available = controller.available(m, input: input)
        let on = (input ? state.inputs : state.outputs).contains(m.rawValue)
        let help = available ? "\(input ? "Takes" : "Makes") \(m.rawValue)" : "No model \(input ? "takes" : "makes") \(m.rawValue) yet"
        return Button {
            if input { controller.state.toggle(input: m) } else { controller.state.toggle(output: m) }
        } label: {
            Image(systemName: m.symbol).font(.system(size: 12, weight: .medium))
                .foregroundStyle((available ? on ? p.accentText : p.t2 : p.t4).swiftUI)
                .frame(maxWidth: .infinity).frame(height: 24)
                .background(
                    RoundedRectangle(cornerRadius: 6).fill(
                        on ? p.accent.withAlphaComponent(0.2).swiftUI : p.field.withAlphaComponent(p.field.alphaComponent * (available ? 1 : 0.5)).swiftUI)
                )
                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(on ? p.accent.withAlphaComponent(0.55).swiftUI : .clear, lineWidth: 1))
                .contentShape(Rectangle())
        }.buttonStyle(.plain).disabled(!available).help(help)
    }

    private var facets: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack {
                Text("Fits this Mac").font(.system(size: 12.5)).foregroundStyle(p.t2.swiftUI)
                Spacer()
                Toggle("", isOn: Binding(get: { controller.state.fits }, set: { controller.state.fits = $0 })).toggleStyle(.switch).controlSize(.mini).labelsHidden()
            }.help(controller.fitsHelp)
            ForEach(BrowserFacet.allCases) { f in facet(f) }
        }.padding(.horizontal, 20)
    }
    private func facet(_ f: BrowserFacet) -> some View {
        let open = controller.openSections.contains(f)
        let chosen = state.values[f] ?? []
        return VStack(alignment: .leading, spacing: 7) {
            Button {
                if NSEvent.modifierFlags.contains(.option) {
                    controller.state.clearFacet(f)
                } else if open {
                    controller.openSections.remove(f)
                } else {
                    controller.openSections.insert(f)
                }
            } label: {
                HStack(spacing: 5) {
                    Text(f.name).font(.system(size: 12, weight: .medium)).foregroundStyle(p.t2.swiftUI)
                    if !chosen.isEmpty { Circle().fill(p.accentText.swiftUI).frame(width: 5, height: 5) }
                    Spacer()
                    Image(systemName: open ? "chevron.down" : "chevron.right").font(.system(size: 9, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                }.contentShape(Rectangle())
            }.buttonStyle(.plain).help(f.help ?? "")
            if open {
                PillFlow(spacing: 6) {
                    ForEach(f.values(controller.items), id: \.self) { value in
                        let on = chosen.contains(value)
                        Button {
                            controller.state.toggle(f, value)
                        } label: {
                            Text(f.title(value)).font(.system(size: 11.5, weight: on ? .semibold : .regular))
                                .foregroundStyle((on ? p.accentText : p.t2).swiftUI)
                                .padding(.horizontal, 8).frame(height: 22)
                                .background(RoundedRectangle(cornerRadius: 6).fill(on ? p.accent.withAlphaComponent(0.2).swiftUI : p.field.swiftUI))
                                .overlay(RoundedRectangle(cornerRadius: 6).strokeBorder(on ? p.accent.withAlphaComponent(0.55).swiftUI : .clear, lineWidth: 1))
                        }.buttonStyle(.plain)
                    }
                }
            }
        }
    }
}

/// The sidebar's foldable sections besides the facets.
enum SidebarGroup: String, CaseIterable {
    case kinds, modality
    var title: String { self == .kinds ? "Kind" : "Input & output" }
}

/// Pills that wrap onto further lines.
struct PillFlow: Layout {
    var spacing: CGFloat = 6
    func sizeThatFits(proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) -> CGSize {
        let w = proposal.width ?? 200; var x: CGFloat = 0, y: CGFloat = 0, line: CGFloat = 0
        for s in subviews {
            let z = s.sizeThatFits(.unspecified)
            if x + z.width > w, x > 0 { x = 0; y += line + spacing; line = 0 }
            x += z.width + spacing; line = max(line, z.height)
        }
        return CGSize(width: w, height: y + line)
    }
    func placeSubviews(in b: CGRect, proposal: ProposedViewSize, subviews: Subviews, cache: inout ()) {
        var x = b.minX, y = b.minY, line: CGFloat = 0
        for s in subviews {
            let z = s.sizeThatFits(.unspecified)
            if x + z.width > b.maxX, x > b.minX { x = b.minX; y += line + spacing; line = 0 }
            s.place(at: CGPoint(x: x, y: y), proposal: .unspecified); x += z.width + spacing; line = max(line, z.height)
        }
    }
}

/// The search field without its system bezel, on the palette's field fill.
struct SearchFieldHost: NSViewRepresentable {
    let field: NSSearchField
    let palette: ArsenalPalette
    func makeNSView(context: Context) -> NSView {
        let box = NSView(); box.wantsLayer = true; box.layer?.cornerRadius = 8
        field.translatesAutoresizingMaskIntoConstraints = false
        field.isBordered = false; field.isBezeled = false; field.drawsBackground = false; field.focusRingType = .none
        field.font = .systemFont(ofSize: 13)
        // Our own magnifier: without its bezel the field's search button overlaps the placeholder in some appearances.
        (field.cell as? NSSearchFieldCell)?.searchButtonCell = nil
        let glass = NSImageView(
            image: NSImage(systemSymbolName: "magnifyingglass", accessibilityDescription: nil)?.withSymbolConfiguration(.init(pointSize: 12, weight: .regular)) ?? NSImage())
        glass.translatesAutoresizingMaskIntoConstraints = false; glass.identifier = .init("glass")
        box.addSubview(glass); box.addSubview(field)
        NSLayoutConstraint.activate([
            glass.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 9), glass.centerYAnchor.constraint(equalTo: box.centerYAnchor),
            field.leadingAnchor.constraint(equalTo: box.leadingAnchor, constant: 26), field.trailingAnchor.constraint(equalTo: box.trailingAnchor, constant: -4),
            field.centerYAnchor.constraint(equalTo: box.centerYAnchor)
        ])
        return box
    }
    func updateNSView(_ box: NSView, context: Context) {
        box.layer?.backgroundColor = palette.field.cgColor
        (box.subviews.first { $0.identifier?.rawValue == "glass" } as? NSImageView)?.contentTintColor = palette.t3
        field.textColor = palette.t1
        field.placeholderAttributedString = NSAttributedString(string: "Search", attributes: [.foregroundColor: palette.t3, .font: NSFont.systemFont(ofSize: 13)])
    }
}
