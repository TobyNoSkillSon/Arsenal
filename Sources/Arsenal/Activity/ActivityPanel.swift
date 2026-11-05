// SPDX-License-Identifier: MIT
import ArsenalCore
import SwiftUI

// The activity panel, the Activity item's submenu (left of the main menu): "the fans are loud — why?". The machine on
// top as numbers (CPU and GPU busy and their temperature in °C; memory as one bar coloured by what holds it), then one
// card per loaded model in its kind's standard form: chat streams and their speed, an image's steps, a voice speaking
// or idle, any other operation's label and count. Each model's colour is its memory segment's. Motion
// only answers new data; nothing is subscribed or drawn while the submenu is closed.

struct ActivityPanel: View {
    @ObservedObject var model: ActivityModel
    @ObservedObject var appearance = Appearance.shared
    @Environment(\.accessibilityReduceMotion) private var reduceMotion
    /// Fixed: two machine tiles side by side and a card's name, speed and memory on one line.
    static let width: CGFloat = 364
    /// The machine block and the empty line; the panel grows with its cards up to `maxHeight`, then scrolls.
    static let minHeight: CGFloat = 150
    static let maxHeight: CGFloat = 640
    static let padding: CGFloat = 14
    private var p: ArsenalPalette { appearance.menuPalette }

    var body: some View {
        let colours = ActivityColours.assign(model.feed.models.map(\.id), over: model.installed)
        VStack(alignment: .leading, spacing: 0) {
            MachineBlock(machine: model.feed.machine, sampled: model.sampled, models: memory(colours), p: p, reduceMotion: reduceMotion)
                .padding(.horizontal, Self.padding).padding(.top, 13).padding(.bottom, 12)
            Rectangle().fill(p.hairline.swiftUI).frame(height: 1)
            if model.feed.models.isEmpty {
                Text("No model loaded").font(.system(size: 12)).foregroundStyle(p.t3.swiftUI)
                    .padding(.horizontal, Self.padding).padding(.vertical, 13)
            } else {
                // As tall as the cards; past the menu's room, the same cards scroll.
                ViewThatFits(in: .vertical) {
                    cards(colours)
                    ScrollView(.vertical) { cards(colours) }.scrollIndicators(.never)
                }
            }
            Spacer(minLength: 0)
        }
        .frame(width: Self.width, alignment: .top)
        .frame(minHeight: Self.minHeight, alignment: .top)
        .foregroundStyle(p.t1.swiftUI)
        .environment(\.colorScheme, appearance.colorScheme)
        .accessibilityElement(children: .contain).accessibilityLabel("Activity")
    }

    /// Each loaded model's memory segment: its colour and size, in the cards' order.
    private func memory(_ colours: [String: NSColor]) -> [MemorySegment] {
        model.feed.models.compactMap { m in
            (m.memoryMB ?? model.memory[m.id]).map { MemorySegment(id: m.id, name: model.name(m.id), mb: $0, colour: colours[m.id] ?? p.accent) }
        }
    }

    private func cards(_ colours: [String: NSColor]) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            ForEach(model.feed.models) { m in
                ModelActivityCard(
                    model: m, name: model.name(m.id), memoryMB: m.memoryMB ?? model.memory[m.id], colour: colours[m.id] ?? p.accent, section: model.section(m),
                    resident: model.rows[m.id]?.residentDType != nil, p: p, reduceMotion: reduceMotion,
                    scaleMB: model.feed.machine.memoryTotalMB ?? Double(ProcessInfo.processInfo.physicalMemory) / 1e6
                )
                .transition(reduceMotion ? .identity : .opacity.combined(with: .move(edge: .top)))
            }
        }.padding(.horizontal, 12).padding(.vertical, 12)
    }
}

/// A model's colour in the memory bar and on its card: chosen by a stable hash of its id, with collisions settled over
/// every model on this Mac in id order, so a model keeps its colour whatever else is loaded. Two loaded models never share
/// one while the palette has room (only possible with more models installed than colours). No blue: the accent belongs to
/// the app's controls, and the meters are grey.
enum ActivityColours {
    static let palette: [NSColor] = [0xB18CFF, 0x2EC4B6, 0xFF9F43, 0xFF6B9A, 0xE8C547, 0x7BD389, 0xC9A27E].map { ArsenalPalette.rgb($0) }
    static func assign(_ ids: [String], over installed: [String] = []) -> [String: NSColor] {
        func probe(_ ids: [String], from start: (String) -> Int) -> [String: Int] {
            var taken = Set<Int>(), slots: [String: Int] = [:]
            for id in ids {
                var slot = start(id)
                if taken.count < palette.count { while taken.contains(slot) { slot = (slot + 1) % palette.count } }
                taken.insert(slot); slots[id] = slot
            }
            return slots
        }
        let stable = probe(Set(installed).union(ids).sorted()) { hash($0) % palette.count }
        return probe(Set(ids).sorted()) { stable[$0] ?? 0 }.mapValues { palette[$0] }
    }
    /// FNV-1a over the id's UTF-8 bytes (Swift's own hash changes per launch).
    static func hash(_ id: String) -> Int {
        Int(id.utf8.reduce(UInt64(0xcbf2_9ce4_8422_2325)) { ($0 ^ UInt64($1)) &* 0x100_0000_01b3 } % UInt64(Int32.max))
    }
    /// A model's colour as a line or a thin fill over the material: as is in dark; over the brighter light material a pale
    /// colour (yellow) is darkened to read, keeping its hue.
    static func ink(_ colour: NSColor, _ p: ArsenalPalette) -> NSColor { p.dark ? colour : colour.blended(withFraction: 0.3, of: .black) ?? colour }
    /// The app itself in the bar (its models have their own colours).
    static let arsenal = ArsenalPalette.rgb(0x8E9AAF)
    /// Everything else: the label colour at a third (half in light, where the material is brighter), well above the free
    /// track's overlay, so it reads over whatever the translucent material shows.
    static func other(_ p: ArsenalPalette) -> NSColor { p.t1.withAlphaComponent(p.dark ? 0.34 : 0.5) }
}

struct MemorySegment: Equatable, Identifiable {
    var id: String
    var name: String
    var mb: Double
    var colour: NSColor
}

// MARK: Machine

/// CPU and GPU side by side (busy share and die temperature), then memory: used of total, and one bar coloured by what
/// holds it (each model, Arsenal, everything else, free). Before the first sample the values are dashes; a value this
/// Mac does not report shows a dash with the reason as tooltip. Every number keeps its place.
struct MachineBlock: View {
    let machine: ActivityFeed.Machine
    let sampled: Bool
    let models: [MemorySegment]
    let p: ArsenalPalette
    let reduceMotion: Bool

    var body: some View {
        VStack(alignment: .leading, spacing: 10) {
            HStack(alignment: .top, spacing: 24) {
                UnitTile(
                    symbol: "cpu", title: "CPU", busy: machine.cpuBusyPct, temp: machine.cpuTempC, sampled: sampled, p: p,
                    help: "All cores, busy share over the last second; mean CPU die temperature.")
                UnitTile(
                    symbol: "square.grid.3x3.fill", title: "GPU", busy: machine.gpuBusyPct, temp: machine.gpuTempC, sampled: sampled, p: p,
                    help: "GPU utilisation, read every second; mean GPU die temperature.")
            }
            MemoryBar(machine: machine, models: models, p: p, reduceMotion: reduceMotion)
        }
    }
}

struct UnitTile: View {
    let symbol: String
    let title: String
    let busy: Double?
    let temp: Double?
    let sampled: Bool
    let p: ArsenalPalette
    let help: String

    var body: some View {
        VStack(alignment: .leading, spacing: 6) {
            HStack(spacing: 5) {
                Image(systemName: symbol).font(.system(size: 11, weight: .medium)).foregroundStyle(p.t2.swiftUI)
                Text(title).font(.system(size: 11, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                Spacer(minLength: 4)
                Image(systemName: "thermometer.medium").font(.system(size: 10)).foregroundStyle(tempColour.swiftUI)
                Text(ActivityText.celsius(temp)).font(.system(size: 11.5, weight: .medium).monospacedDigit()).foregroundStyle(tempColour.swiftUI).fixedNumber()
            }
            HStack(spacing: 8) {
                NumberUnit(value: ActivityText.percent(busy), unit: "%", size: 17, p: p).frame(width: 52, alignment: .leading).fixedNumber()
                // A new reading switches the fill in place, as it does the number beside it: easing every second kept the
                // panel animating 40% of the time it was open, for steps of a pixel or two (measured: 10–24% of a core
                // open, 1–3% without).
                // Snapped to the whole percent printed beside it, so the bar never moves while the number stands still.
                Meter(fraction: ActivityText.busyFraction(busy), color: p.t2, p: p, height: 4)
            }
        }
        .frame(maxWidth: .infinity)
        // One hover for the whole tile, over it (TooltipCell's `cellTooltip`); the missing-sensor note joins it rather than
        // nesting a second hover the tile's would cover.
        .cellTooltip(temp == nil && sampled ? help + "\nThis Mac\u{2019}s \(title) temperature sensors are not known to Arsenal." : help)
    }
    /// Hot dies read amber, then red (Apple silicon throttles around 100 °C).
    private var tempColour: NSColor {
        guard let temp else { return p.t3 }
        return temp >= 95 ? p.red : temp >= 85 ? p.amber : p.t2
    }
}

/// "23" large with its unit small; a dash when unknown.
struct NumberUnit: View {
    let value: String?
    let unit: String
    let size: CGFloat
    let p: ArsenalPalette
    var body: some View {
        if let value {
            (Text(value).font(.system(size: size, weight: .semibold).monospacedDigit()) + Text(" " + unit).font(.system(size: size * 0.62)).foregroundColor(p.t3.swiftUI))
                .lineLimit(1)
        } else {
            Text("\u{2014}").font(.system(size: size, weight: .semibold)).foregroundStyle(p.t4.swiftUI)
        }
    }
}

struct MemoryBar: View {
    let machine: ActivityFeed.Machine
    let models: [MemorySegment]
    let p: ArsenalPalette
    let reduceMotion: Bool

    var body: some View {
        let parts = ActivityText.memoryParts(machine: machine, models: models, p: p)
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 5) {
                Image(systemName: "memorychip").font(.system(size: 11, weight: .medium)).foregroundStyle(p.t2.swiftUI)
                Text("Memory").font(.system(size: 11, weight: .semibold)).foregroundStyle(p.t3.swiftUI)
                Spacer(minLength: 4)
                NumberUnit(value: ActivityText.gb(machine.memoryUsedMB), unit: "of " + (ActivityText.gb(machine.memoryTotalMB, decimals: 0) ?? "\u{2014}") + " GB", size: 13, p: p)
                    .fixedNumber()
            }
            // The track is drawn from the first frame; the segments are plain rectangles (at least 3 pt wide, so a sliver
            // never turns into a crescent at the cap) clipped once by the bar's own outline.
            GeometryReader { g in
                ZStack(alignment: .leading) {
                    Rectangle().fill(p.field.swiftUI)
                    HStack(spacing: 1) {
                        ForEach(parts.filter { $0.kind != .free && $0.mb > 0 }) { part in
                            Rectangle().fill(part.colour.swiftUI).frame(width: max(3, g.size.width * part.fraction))
                        }
                    }
                    // Eases only when a holder arrives or leaves (a model loads or unloads); a second's drift switches in place.
                    .animation(reduceMotion ? nil : .smooth(duration: 0.4), value: parts.map(\.id))
                }
                .clipShape(RoundedRectangle(cornerRadius: 2))
            }
            .frame(height: 8)
            .cellTooltip(parts.map { "\($0.name): \(ActivityText.gb($0.mb) ?? "\u{2014}") GB" }.joined(separator: "\n"))
            // The models together (each is named on its card, by colour), the app, everything else and free: the legend
            // adds up to the header. Before the first reading the entries hold their place with dashes.
            HStack(spacing: 12) {
                ForEach(ActivityText.shownLegend(parts: parts, models: models, totalMB: machine.memoryTotalMB, p: p)) { entry in
                    HStack(spacing: 4) {
                        Swatch(colours: entry.colours, outlined: entry.kind == .free, p: p)
                        (Text(entry.name + " ").foregroundColor(p.t3.swiftUI) + Text(ActivityText.gb(entry.mb) ?? "\u{2014}").foregroundColor(p.t2.swiftUI))
                            .font(.system(size: 11).monospacedDigit()).lineLimit(1).fixedNumber()
                    }
                    .cellTooltip(entry.help)
                }
                Spacer(minLength: 0)
            }
        }
    }
}

/// A legend's 8 pt square: one colour, or the loaded models' colours side by side.
struct Swatch: View {
    let colours: [NSColor]
    let outlined: Bool
    let p: ArsenalPalette
    var body: some View {
        HStack(spacing: 0) {
            ForEach(Array(colours.enumerated()), id: \.offset) { _, colour in Rectangle().fill(colour.swiftUI) }
        }
        .frame(width: 8, height: 8)
        .clipShape(RoundedRectangle(cornerRadius: 2))
        .overlay(RoundedRectangle(cornerRadius: 2).strokeBorder(outlined ? p.raisedBorder.swiftUI : .clear, lineWidth: 1))
    }
}

extension View {
    /// A number's text changes in place, never cross-fading old and new digits (whatever animation is running).
    func fixedNumber() -> some View { transaction { $0.animation = nil } }
}

/// A thin capsule meter; its fill follows the value (animated by the panel's data animation).
struct Meter: View {
    let fraction: Double?
    let color: NSColor
    let p: ArsenalPalette
    var height: CGFloat = 5
    var body: some View {
        GeometryReader { g in
            ZStack(alignment: .leading) {
                Capsule().fill(p.field.swiftUI)
                Capsule().fill(color.swiftUI).frame(width: max(fraction == nil ? 0 : height, g.size.width * min(1, max(0, fraction ?? 0))))
            }
        }.frame(height: height)
    }
}

// MARK: Model cards

/// One loaded model in its kind's standard form. The header names it (in its memory colour) with its kind's symbol, the
/// figure that sums it up and its memory. Under it, what it is doing now, the same on every card: how many requests run,
/// how many wait and how long the wait is; then, except for decisions, the running requests by client (Stream A, B, … when the client gave
/// no name) with their speed, steps or audio. Last, its minute: how busy it was each second, and what it finished.
struct ModelActivityCard: View {
    let model: ActivityFeed.Model
    let name: String
    let memoryMB: Double?
    let colour: NSColor
    let section: ActivitySection
    /// Already loaded: a request's brief "loading" phase is its start, not a model load.
    let resident: Bool
    let p: ArsenalPalette
    let reduceMotion: Bool
    /// The Mac's memory: every card's memory bar is drawn to it, so cards compare and the machine bar above agrees.
    var scaleMB: Double = Double(ProcessInfo.processInfo.physicalMemory) / 1e6

    var body: some View {
        let lines = section == .decision ? [] : ActivityText.lines(model)
        let shown = Array(lines.prefix(ActivityText.maxLines))
        VStack(alignment: .leading, spacing: 7) {
            HStack(spacing: 7) {
                Circle().fill(colour.swiftUI).frame(width: 8, height: 8)
                Image(systemName: section.symbol).font(.system(size: 11)).foregroundStyle(p.t2.swiftUI).frame(width: 15)
                // A long name gives way in its middle (both ends identify a model); the figures never do.
                Text(name).font(.system(size: 13, weight: .semibold)).lineLimit(1).truncationMode(.middle).cellTooltip(name)
                Spacer(minLength: 6)
                if let summary = ActivityText.summary(model, section, resident: resident) {
                    Text(summary.text).font(.system(size: 12, weight: summary.strong ? .semibold : .regular).monospacedDigit())
                        .foregroundStyle((summary.strong ? p.t1 : p.t3).swiftUI).lineLimit(1).fixedSize().fixedNumber()
                }
                Text(ActivityText.gb(memoryMB).map { $0 + " GB" } ?? "\u{2014}").font(.system(size: 11.5).monospacedDigit())
                    .foregroundStyle((memoryMB == nil ? p.t4 : p.t3).swiftUI).fixedSize().fixedNumber().frame(minWidth: 46, alignment: .trailing)
                    .cellTooltip(ActivityText.nowHelp(memoryMB))
            }
            // Only a queue to count beside the running lines: it follows the first line ("Claude Code · 30.8 tok/s ·
            // 1 waiting"), never a line of its own that reads as another job.
            let trailing = ActivityText.trailingCounts(model, lines: shown)
            if trailing == nil, let counts = ActivityText.counts(model) {
                CountsLine(counts: counts, help: ActivityText.countsHelp(model), p: p)
            }
            ForEach(shown, id: \.request.id) { line in
                RequestLine(
                    line: line, section: section, resident: resident, colour: colour, p: p, reduceMotion: reduceMotion,
                    trailing: line.request.id == shown.first?.request.id ? trailing : nil, trailingHelp: ActivityText.countsHelp(model)
                )
                .transition(.asymmetric(insertion: .opacity.combined(with: .move(edge: .top)), removal: .opacity))
            }
            if let recent = model.recent {
                MinuteLine(recent: recent, section: section, colour: colour, p: p)
            }
            if let memory = model.memory {
                ModelMemoryLine(memory: memory, scaleMB: scaleMB, colour: colour, p: p)
            }
        }
        .padding(.horizontal, 11).padding(.vertical, 9)
        .background(Self.fill(p).swiftUI, in: RoundedRectangle(cornerRadius: 9))
    }
    /// A card is a faint lift of the menu's own material, never a solid panel over it: the wallpaper still reads through
    /// it in light as in dark.
    static func fill(_ p: ArsenalPalette) -> NSColor { NSColor.white.withAlphaComponent(p.dark ? 0.05 : 0.2) }
}

/// What the model's memory is for, on the Mac's scale: its weights in its colour, its working memory (activations, KV
/// cache or context, scratch) lighter and MLX's reuse cache lighter still, what its process holds outside MLX in grey; the
/// most the current or last job took is a faint band behind them ending in a tick. Under it, the parts' figures in GB.
/// Every value is measured (C3's `memory`); redrawn only when a reading changes, never animated.
struct ModelMemoryLine: View {
    let memory: ActivityFeed.Memory
    let scaleMB: Double
    let colour: NSColor
    let p: ArsenalPalette
    static let barHeight: CGFloat = 6

    var body: some View {
        let parts = ActivityText.memoryBar(memory)
        // The memory chip in the header symbol's column marks this bar as memory, apart from the progress bars above it.
        HStack(alignment: .top, spacing: 7) {
            Image(systemName: "memorychip").font(.system(size: 10)).foregroundStyle(p.t3.swiftUI).frame(width: 15, height: Self.barHeight)
            bars(parts)
        }
        .padding(.leading, 15).padding(.top, 2)
        .contentShape(Rectangle())
        .cellTooltip(ActivityText.memoryHelp(memory))
        .transaction { $0.animation = nil }
        .accessibilityElement(children: .ignore)
        .accessibilityLabel(ActivityText.memoryHelp(memory))
    }
    private func bars(_ parts: [ActivityText.MemoryPiece]) -> some View {
        VStack(alignment: .leading, spacing: 6) {
            Canvas { context, size in
                let track = Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 2)
                context.fill(track, with: .color(p.field.swiftUI))
                context.clip(to: track)
                func width(_ mb: Double) -> CGFloat { size.width * CGFloat(min(1, max(0, mb / max(1, scaleMB)))) }
                // Each part is drawn at least 2 pt wide: the peak's band and tick never end inside the drawn bar.
                let drawn = parts.filter { $0.mb > 0 }.reduce(CGFloat(0)) { $0 + max(2, width($1.mb)) + 1 }
                let peak = memory.peakMB.map { max(3, width($0), drawn - 1) }
                if let peak { context.fill(Path(CGRect(x: 0, y: 0, width: peak, height: size.height)), with: .color(Self.peakFill(colour, p).swiftUI)) }
                var x: CGFloat = 0
                for part in parts where part.mb > 0 {
                    let w = max(2, width(part.mb))
                    context.fill(Path(CGRect(x: x, y: 0, width: w, height: size.height)), with: .color(part.layer.fill(colour, p).swiftUI))
                    x += w + 1
                }
                if let peak {
                    context.fill(Path(CGRect(x: max(0, peak - 1.5), y: 0, width: 1.5, height: size.height)), with: .color(ActivityColours.ink(colour, p).swiftUI))
                }
            }
            .frame(height: Self.barHeight)
            HStack(spacing: 12) {
                ForEach(ActivityText.memoryLegend(memory), id: \.name) { entry in
                    HStack(spacing: 4) {
                        Group {
                            if let layer = entry.layer {
                                RoundedRectangle(cornerRadius: 2).fill(layer.fill(colour, p).swiftUI)
                            } else {
                                // The peak: its faint band and its tick.
                                ZStack(alignment: .trailing) {
                                    RoundedRectangle(cornerRadius: 2).fill(Self.peakFill(colour, p).swiftUI)
                                    Rectangle().fill(ActivityColours.ink(colour, p).swiftUI).frame(width: 1.5)
                                }.clipShape(RoundedRectangle(cornerRadius: 2))
                            }
                        }.frame(width: 8, height: 8)
                        (Text(entry.name + " ").foregroundColor(p.t3.swiftUI)
                            + Text(ActivityText.gb(entry.mb) ?? "\u{2014}").foregroundColor((entry.mb == nil ? p.t4 : p.t2).swiftUI))
                            .font(.system(size: 11).monospacedDigit()).lineLimit(1).fixedSize()
                    }
                }
                Spacer(minLength: 0)
            }
        }
    }
    static func peakFill(_ colour: NSColor, _ p: ArsenalPalette) -> NSColor { colour.withAlphaComponent(p.dark ? 0.26 : 0.3) }
}

/// "1 running · 40 waiting · 8 s wait": the numbers carry the line, their words stay quiet; zeros are left out, and with
/// nothing running or waiting (but work in the last minute) it reads a quiet "Idle", keeping its place.
struct CountsLine: View {
    let counts: [(String, String)]
    let help: String?
    let p: ArsenalPalette
    var body: some View {
        Group {
            if counts.isEmpty {
                // As tall as a line of counts (their numbers are larger), so the card does not move when it empties.
                ZStack(alignment: .leading) {
                    Text("0").font(.system(size: 13, weight: .semibold).monospacedDigit()).hidden()
                    Text("Idle").font(.system(size: 12)).foregroundStyle(p.t3.swiftUI)
                }
            } else {
                counts.enumerated().reduce(Text("")) { line, item in
                    let (i, (value, word)) = item
                    return line + Text(i == 0 ? "" : "  \u{00b7}  ").foregroundColor(p.t4.swiftUI)
                        + Text(value).font(.system(size: 13, weight: .semibold).monospacedDigit()).foregroundColor(p.t1.swiftUI)
                        + Text(" " + word).font(.system(size: 12)).foregroundColor(p.t3.swiftUI)
                }
            }
        }
        .lineLimit(1).fixedNumber().padding(.leading, 15)
        .cellTooltip(help)
        .accessibilityLabel(counts.isEmpty ? "Idle" : counts.map { "\($0.0) \($0.1)" }.joined(separator: ", "))
    }
}

/// The model's last minute: how busy it was each second as a small graph in its colour, and what it finished, with the
/// full figures on hover.
struct MinuteLine: View {
    let recent: ActivityFeed.Recent
    let section: ActivitySection
    let colour: NSColor
    let p: ArsenalPalette
    static let graphSize = CGSize(width: 96, height: 16)
    var body: some View {
        // The caption follows its graph; a longer caption grows into the free space and never moves the graph.
        HStack(alignment: .center, spacing: 10) {
            BusyGraph(values: recent.busy, colour: colour, p: p).frame(width: Self.graphSize.width, height: Self.graphSize.height)
            let caption = ActivityText.minute(recent, section)
            (Text(caption.text).foregroundColor(p.t2.swiftUI)
                + Text(caption.failed.map { (caption.text.isEmpty ? "" : "  \u{00b7}  ") + $0 } ?? "").foregroundColor(p.amber.swiftUI))
                .font(.system(size: 11.5).monospacedDigit()).lineLimit(1).fixedSize().fixedNumber()
            Spacer(minLength: 0)
        }
        .padding(.leading, 15)
        .cellTooltip(ActivityText.minuteHelp(recent, section))
        .transaction { $0.animation = nil }
    }
}

/// The last sixty seconds, oldest at the left: how much of each second the model was working, inside a faint track that
/// is the whole second. A full band is a model busy without a break (with a queue, the queue is waiting on it); gaps
/// are idle time. One scale for every kind and every model, so nothing rescales as the minute scrolls. Redrawn only when
/// a new second arrives; never animated.
struct BusyGraph: View {
    let values: [Double]
    let colour: NSColor
    let p: ArsenalPalette
    var body: some View {
        Canvas { context, size in
            let track = Path(roundedRect: CGRect(origin: .zero, size: size), cornerRadius: 2)
            context.fill(track, with: .color(p.field.swiftUI))
            guard values.count > 1, values.contains(where: { $0 > 0 }) else { return }
            context.clip(to: track)
            let step = size.width / CGFloat(values.count - 1)
            // The line's middle stays inside the track at both ends of the scale.
            func y(_ v: Double) -> CGFloat { size.height - 0.75 - (size.height - 1.5) * CGFloat(min(1, max(0, v))) }
            var line = Path()
            for (i, v) in values.enumerated() {
                let point = CGPoint(x: CGFloat(i) * step, y: y(v))
                if i == 0 { line.move(to: point) } else { line.addLine(to: point) }
            }
            var area = line
            area.addLine(to: CGPoint(x: size.width, y: size.height)); area.addLine(to: CGPoint(x: 0, y: size.height)); area.closeSubpath()
            context.fill(area, with: .color(colour.withAlphaComponent(p.dark ? 0.32 : 0.4).swiftUI))
            context.stroke(line, with: .color(ActivityColours.ink(colour, p).swiftUI), style: StrokeStyle(lineWidth: 1.25, lineJoin: .round))
        }
        .accessibilityLabel("Busy \(Int((values.reduce(0, +) / Double(max(1, values.count)) * 100).rounded()))% of the last minute")
    }
}

struct RequestLine: View {
    let line: ActivityText.Line
    let section: ActivitySection
    let resident: Bool
    let colour: NSColor
    let p: ArsenalPalette
    let reduceMotion: Bool
    /// The card's queue when it has nothing else to count ("1 waiting"), after this line's figure.
    var trailing: [(String, String)]?
    var trailingHelp: String?

    var body: some View {
        let r = line.request, fraction = ActivityText.fraction(r)
        let detail = ActivityText.detail(r, section, resident: resident)
        VStack(alignment: .leading, spacing: 5) {
            HStack(spacing: 6) {
                // A lone request without a client name is named by what it does ("Reading 46K tokens"), never "Request".
                let name = line.name.isEmpty ? detail : line.name
                (Text(name) + Text(ActivityText.doing(r, section).map { " \u{00b7} " + $0 } ?? "").font(.system(size: 12)).foregroundColor(p.t3.swiftUI))
                    .font(.system(size: 12, weight: .medium)).lineLimit(1).truncationMode(.tail)
                    .cellTooltip(line.named ? "The client\u{2019}s own name." : "This client sent no name.")
                Spacer(minLength: 6)
                (Text(line.name.isEmpty ? "" : detail).foregroundColor(p.t2.swiftUI)
                    + (trailing ?? []).enumerated().reduce(Text("")) { text, item in
                        let (i, (value, word)) = item
                        return text + Text(i == 0 && line.name.isEmpty ? "" : "  \u{00b7}  ").foregroundColor(p.t4.swiftUI)
                            + Text(value).foregroundColor(p.t2.swiftUI) + Text(" " + word).foregroundColor(p.t3.swiftUI)
                    })
                    .font(.system(size: 12).monospacedDigit()).lineLimit(1).fixedSize().fixedNumber()
                    .cellTooltip(trailing == nil ? nil : trailingHelp)
            }
            if ActivityText.showsBar(r, section) {
                Meter(fraction: fraction ?? 0, color: ActivityColours.ink(colour, p), p: p, height: 3)
                    .animation(reduceMotion ? nil : .smooth(duration: 0.32), value: fraction)
            }
        }
        .padding(.leading, 15)
    }
}

struct StateDot: View {
    let color: NSColor
    var body: some View { Circle().fill(color.swiftUI).frame(width: 7, height: 7).frame(width: 16, height: 14) }
}

struct Tag: View {
    let text: String
    let p: ArsenalPalette
    var tint: NSColor?
    var body: some View {
        Text(text).font(.system(size: 10.5, weight: .medium).monospacedDigit()).foregroundStyle((tint ?? p.t2).swiftUI)
            .padding(.horizontal, 6).frame(height: 17)
            .background((tint?.withAlphaComponent(0.14) ?? p.field).swiftUI, in: RoundedRectangle(cornerRadius: 4))
    }
}

extension ActivitySection {
    /// The same symbols as the menu's tool items (Voice: person.wave.2).
    var symbol: String {
        switch self {
        case .chat: return "text.bubble"
        case .image: return "photo"
        case .voice: return "person.wave.2"
        case .decision: return "checkmark.circle"
        case .default: return "square.stack.3d.up"
        }
    }
}

/// The panel's words and numbers (unit-tested): figures with units, few words.
enum ActivityText {
    /// A request with the name the panel shows: the client's own; with several unnamed, "Request 1", "Request 2", … in
    /// arrival order; one unnamed has no name (its line names it by what it does). Never an internal label.
    struct Line { var request: ActivityFeed.Request; var name: String; var named: Bool }
    /// Names the request lists did not get from their client ("API": no X-Arsenal-Client and no MCP client name).
    static func isNamed(_ requester: String) -> Bool {
        let r = requester.trimmingCharacters(in: .whitespaces)
        return !r.isEmpty && r != "API" && r != "MCP"
    }
    /// The running requests (the waiting ones are counted, not listed).
    static func lines(_ m: ActivityFeed.Model) -> [Line] {
        var unnamed = 0
        let running = m.requests.filter { !["queued", "loading"].contains($0.phase) }.sorted { ($0.startedAt ?? .distantPast) < ($1.startedAt ?? .distantPast) }
        let several = running.count { !isNamed($0.requester) } > 1
        return running.map { r in
            if isNamed(r.requester) { return Line(request: r, name: r.requester, named: true) }
            unnamed += 1
            return Line(request: r, name: several ? "Request \(unnamed)" : "", named: false)
        }
    }
    /// A card's counts when they are only a queue (nothing counted as running) and request lines are shown: they follow
    /// the first line instead of taking a line of their own. Nil otherwise (the counts line shows them).
    static func trailingCounts(_ m: ActivityFeed.Model, lines: [Line]) -> [(String, String)]? {
        guard !lines.isEmpty, m.running == 0, let counts = counts(m), !counts.isEmpty else { return nil }
        return counts
    }

    private static func active(_ m: ActivityFeed.Model) -> [ActivityFeed.Request] { m.requests.filter { !["queued", "done"].contains($0.phase) } }

    /// What sums a model up in its header: a chat model's total speed across its streams, a voice speaking, a load;
    /// "Idle" when nothing runs and the card has no counts line to say so. Strong when it is a live figure.
    static func summary(_ m: ActivityFeed.Model, _ section: ActivitySection, resident: Bool) -> (text: String, strong: Bool)? {
        let active = active(m)
        if m.state == "reloading" { return ("Updating", false) }
        if m.state == "loading" || (!resident && m.requests.contains { $0.phase == "loading" }) { return ("Loading", false) }
        if active.isEmpty { return counts(m) == nil ? ("Idle", false) : nil }
        switch section {
        case .chat:
            // The total when several streams write; one stream's speed is already on its line.
            guard m.requests.filter({ $0.phase == "decode" }).count > 1, let total = chatSpeed(m) else { return nil }
            return (FigureText.rate(total) ?? "", true)
        // Said once: a card listing its requests ("Claude Code · Speaking") leaves the header to its memory figure,
        // as chat and image cards do.
        case .voice: return lines(m).isEmpty ? ("Speaking", true) : nil
        case .image, .decision, .default: return nil
        }
    }

    /// Running request lines a card lists (its counts line has the total).
    static let maxLines = 5
    /// Every card's counts: "1 running · 40 waiting · 8 s wait", zeros left out. Empty (the line reads Idle) when nothing
    /// runs or waits but the last minute had work, so the card keeps its height between one-at-a-time requests; nil when
    /// the minute is empty too.
    static func counts(_ m: ActivityFeed.Model) -> [(String, String)]? {
        var counts: [(String, String)] = []
        if m.running > 0 { counts.append((m.running.grouped, "running")) }
        if m.queued > 0 { counts.append((m.queued.grouped, "waiting")) }
        if let wait = waitS(m) { counts.append((estimate(wait), "wait")) }
        return counts.isEmpty && m.recent == nil ? nil : counts
    }
    /// How long the queue takes to clear at the last minute's pace: what is left of the running requests, then those
    /// waiting times the time each took, shared across the requests running side by side. Nil without a queue or a
    /// finished request to time.
    static func waitS(_ m: ActivityFeed.Model, now: Date = Date()) -> Double? {
        guard m.queued > 0, let each = m.recent?.meanRunS, each > 0, each.isFinite else { return nil }
        return (remainderS(m, now: now) ?? 0) + Double(m.queued) * each / Double(max(1, m.running))
    }
    /// What is left of the running requests at the minute's pace, on average: a request with steps has its steps left
    /// (3 of 4 done: a quarter of the time each); any other has the time each less the time it has run (since it was
    /// accepted, from a service that does not say when it started; nothing when neither is known). Nil when none runs.
    static func remainderS(_ m: ActivityFeed.Model, now: Date = Date()) -> Double? {
        guard let each = m.recent?.meanRunS, each > 0, each.isFinite else { return nil }
        let running = m.requests.filter { !["queued", "loading", "done"].contains($0.phase) }
        guard !running.isEmpty else { return nil }
        let left = running.map { r -> Double in
            if r.phase != "prefill", let f = fraction(r) { return (1 - f) * each }
            guard let since = r.runningSince ?? r.startedAt else { return 0 }
            return max(0, each - now.timeIntervalSince(since))
        }
        return left.reduce(0, +) / Double(left.count)
    }
    /// The counts line on hover: where the wait comes from.
    static func countsHelp(_ m: ActivityFeed.Model, now: Date = Date()) -> String? {
        guard waitS(m, now: now) != nil, let each = duration(m.recent?.meanRunS) else { return nil }
        let remainder = remainderS(m, now: now).flatMap { $0 >= 0.0005 ? duration($0) : nil }
        let left = remainder.map { "\($0) left on the running " + (m.running > 1 ? "ones" : "one") + ", then " } ?? ""
        return "Wait: " + left + "\(m.queued.grouped) waiting \u{00d7} \(each) each" + (m.running > 1 ? ", \(m.running) at a time." : ".")
    }
    /// An estimate, so whole seconds: "<1 s", "8 s", "3 min".
    static func estimate(_ s: Double) -> String {
        if s < 1 { return "<1 s" }
        if s < 59.5 { return "\(Int(s.rounded())) s" }
        return s < 3570 ? "\(Int((s / 60).rounded())) min" : String(format: "%.1f h", s / 3600)
    }
    /// The minute's caption in the kind's own terms, one time base per quantity: "250 decisions/min · 190 ms each",
    /// "2 replies/min · avg 29.8 tok/s", "2 images/min · 6.1 s each", "1.2× real time"; failures separately, so they can
    /// read amber.
    static func minute(_ r: ActivityFeed.Recent, _ section: ActivitySection) -> (text: String, failed: String?) {
        let each = duration(r.meanRunS).map { " \u{00b7} \($0) each" } ?? ""
        let text: String
        switch section {
        case .chat:
            let replies = r.finished > 0 ? plural(r.finished, "reply", "replies") + "/min" : nil
            // Tokens over the seconds it was working: the speed it wrote at, not diluted by idle time.
            let speed = r.tokens > 0 && r.busyS > 0 ? FigureText.rate(Double(r.tokens) / r.busyS).map { "avg " + $0 } : nil
            text = [replies, speed].compactMap { $0 }.joined(separator: " \u{00b7} ")
        case .image:
            let images = r.images > 0 ? r.images : r.finished
            text = images > 0 ? plural(images, "image") + "/min" + each : r.steps > 0 ? plural(r.steps, "step") + "/min" : ""
        case .voice:
            text = speechRate(r) ?? (r.finished > 0 ? plural(r.finished, "clip") + "/min" : "")
        case .decision:
            text = r.finished > 0 ? plural(r.finished, "decision") + "/min" + each : ""
        case .default:
            text =
                speechRate(r)
                ?? (r.finished > 0 ? plural(r.finished, "request") + "/min" + each : r.steps > 0 ? plural(r.steps, "step") + "/min" : "")
        }
        let failed = r.failed > 0 ? "\(r.failed.grouped) failed" : nil
        return (text, failed)
    }
    /// The finished requests' audio over the seconds they worked, so idle gaps between requests do not dilute it:
    /// "1.2× real time". Until one finishes, the minute's audio so far.
    static func speechRate(_ r: ActivityFeed.Recent) -> String? {
        if let x = r.realTime, x.isFinite, x > 0 { return realTime(x) }
        return r.audioS > 0 ? audio(r.audioS) + " audio/min" : nil
    }
    static func realTime(_ x: Double) -> String { (x < 9.95 ? String(format: "%.1f", x) : (x.roundedInt ?? 0).grouped) + "\u{00d7} real time" }
    /// The minute in full, on hover.
    static func minuteHelp(_ r: ActivityFeed.Recent, _ section: ActivitySection) -> String {
        var lines = ["Last minute: \(r.finished.grouped) finished, \(r.failed.grouped) failed; working \(Int(r.busyS.rounded())) s of 60."]
        if let run = duration(r.meanRunS) {
            // The minute's past average, worded apart from the counts line's projected wait.
            lines.append("Each worked \(run)" + (duration(r.meanWaitS).map { " and waited \($0) on average." } ?? " on average."))
        }
        if let x = r.realTime, x.isFinite, x > 0 {
            lines.append("Each request made \(realTime(x)): seconds of audio per second it worked, over the minute's finished requests.")
        }
        lines.append("Graph: how much of each second it was working; full height is the whole second.")
        return lines.joined(separator: "\n")
    }
    /// 0.0184 → "18 ms", 1.24 → "1.2 s", 42 → "42 s".
    static func duration(_ s: Double?) -> String? {
        guard let s, s.isFinite, s >= 0, s < 1e7 else { return nil }
        if s < 1 { return "\((s * 1000).roundedInt ?? 0) ms" }
        return s < 10 ? String(format: "%.1f s", s) : "\((s.roundedInt ?? 0).grouped) s"
    }
    /// Seconds of audio: "8.4 s", "42 s", "3.5 min".
    static func audio(_ s: Double) -> String {
        s < 10 ? String(format: "%.1f s", s) : s < 600 ? "\(s.roundedInt ?? 0) s" : String(format: "%.1f min", s / 60)
    }
    static func plural(_ n: Int, _ one: String, _ many: String? = nil) -> String { "\(n.grouped) " + (n == 1 ? one : many ?? one + "s") }

    /// The sum of the decode speeds of a chat model's writing streams; nil when none reports one.
    static func chatSpeed(_ m: ActivityFeed.Model) -> Double? {
        let rates = m.requests.filter { $0.phase == "decode" }.compactMap { $0.timing["decode_tps"] }.filter { $0.isFinite && $0 > 0 && $0 < 1e6 }
        return rates.isEmpty ? nil : rates.reduce(0, +)
    }
    /// Every chat model's streams together (the panel's total).
    static func chatSpeed(_ models: [ActivityFeed.Model], sections: (ActivityFeed.Model) -> ActivitySection) -> Double? {
        let rates = models.filter { sections($0) == .chat }.compactMap(chatSpeed)
        return rates.isEmpty ? nil : rates.reduce(0, +)
    }

    /// A request's figure: chat "30.8 tok/s" or its prompt read so far, an image "3 / 4", any other operation its count
    /// in its unit ("2 / 4 layers").
    static func detail(_ r: ActivityFeed.Request, _ section: ActivitySection, resident: Bool = false) -> String {
        let count = r.step.map { s in r.total.map { "\(s) / \($0)" } ?? "\(s)" }
        switch r.phase {
        case "loading": return resident ? "Starting" : "Loading"
        case "done": return "Done"
        case "prefill":
            // With a known total the bar shows how far; without one, how much prompt is being read.
            if fraction(r) != nil { return "Reading" }
            return r.promptTokens.map { "Reading \(compact(max(0, $0 - (r.cachedTokens ?? 0)))) tokens" } ?? "Reading"
        case "decode":
            // Only the decode rate: a request's `tps` may still be its prompt rate.
            return FigureText.rate(r.phaseRate) ?? r.generatedTokens.map { "\(compact($0)) tok" } ?? "Writing"
        default:
            switch section {
            case .chat: return "Writing"
            case .voice: return r.audioS.map { audio($0) + " audio" } ?? "Speaking"
            case .image: return count.map { "\($0) steps" } ?? "Starting"
            case .decision: return "Deciding"
            case .default:
                // The unit once per line: "Separating layers · 2 / 4", not "… layers · 2 / 4 layers".
                let unit = r.unit.flatMap { u in doing(r, section)?.lowercased().contains(u.lowercased()) == true ? nil : u }
                return count.map { c in unit.map { "\(c) \($0)" } ?? c } ?? fraction(r).flatMap { ($0 * 100).roundedInt }.map { "\($0)%" } ?? "Working"
            }
        }
    }
    /// What a request of a default-section model is doing in the runner's words ("Separating layers"), else its
    /// operation's title; nil for the built-in sections, whose figures already say it.
    static func doing(_ r: ActivityFeed.Request, _ section: ActivitySection) -> String? {
        guard section == .default, !["loading", "done", "prefill", "decode"].contains(r.phase) else { return nil }
        return r.label ?? r.title
    }
    /// A step bar under images, other operations with a total, and a prompt being read.
    static func showsBar(_ r: ActivityFeed.Request, _ section: ActivitySection) -> Bool {
        guard fraction(r) != nil else { return false }
        return r.phase == "prefill" || section == .image || section == .default
    }
    static func fraction(_ r: ActivityFeed.Request) -> Double? {
        guard let step = r.step, let total = r.total, total > 0 else { return nil }
        return min(1, Double(step) / Double(total))
    }
    /// 46,072 → "46K".
    static func compact(_ n: Int) -> String { n >= 10_000 ? "\((Double(n) / 1000).roundedInt ?? n / 1000)K" : n.grouped }

    static func percent(_ v: Double?) -> String? { v.flatMap { min(100, max(0, $0)).roundedInt }.map(String.init) }
    /// A busy bar's fill: the printed whole percent (23.6 → "24" → 0.24).
    static func busyFraction(_ v: Double?) -> Double? { percent(v).flatMap(Double.init).map { $0 / 100 } }
    static func celsius(_ v: Double?) -> String { v.flatMap { $0.roundedInt }.map { "\($0) \u{00b0}C" } ?? "\u{2014}" }
    /// MB (10⁶ bytes, as the service reports) → GB as macOS shows memory (2³⁰ bytes: a 128 GB Mac reads 128), one
    /// decimal, whole GB when asked. Every memory figure in the panel uses it, so the bar, its legend and the cards agree.
    static func gb(_ mb: Double?, decimals: Int = 1) -> String? {
        guard let mb, mb.isFinite, mb >= 0, mb < 1e12 else { return nil }
        return String(format: "%.\(decimals)f", mb * 1e6 / 1_073_741_824)
    }

    /// The memory macOS reports as free: a short legend word, so "Free 71.0" never truncates; its hover names it in
    /// full (system free) and sets it apart from the menu's "Room for models".
    static let freeName = "Free"
    /// Its hover: what it is in full, and how it differs from the menu's Room for models (Arsenal's room under its
    /// Mac-wide limit).
    static func freeHelp(totalMB: Double?) -> String {
        let limit = totalMB.map { " " + MemoryDisplay.text(decimalGB: MacMemoryGuard.safeInUseMB(totalMB: $0) / 1000) } ?? ""
        return "System free: memory macOS has free now. The menu\u{2019}s Room for models is different: how much more models can use under "
            + "Arsenal\u{2019}s\(limit) limit, which keeps a margin for the rest of the Mac."
    }
    /// A card's memory figure: what the model holds now (the Models table's Peak memory is the most it uses).
    static func nowHelp(_ mb: Double?) -> String {
        gb(mb).map { "\($0) GB in memory now. The Models table\u{2019}s Peak memory is the most it uses." } ?? "Memory now: not reported yet."
    }

    /// The layers of a model's memory bar, drawn in its colour.
    enum MemoryLayer: Equatable {
        case weights, working, cache, outside
        func fill(_ colour: NSColor, _ p: ArsenalPalette) -> NSColor {
            switch self {
            case .weights: return colour
            case .working: return colour.withAlphaComponent(0.6)
            case .cache: return colour.withAlphaComponent(0.34)
            case .outside: return ActivityColours.other(p)
            }
        }
    }
    struct MemoryPiece: Equatable {
        var layer: MemoryLayer
        var mb: Double
    }
    /// The bar, left to right. With MLX's counters: weights (or, when the weights were not measured apart, everything MLX
    /// holds), working memory, MLX's cache, and the rest of the process; without them, the process as one piece.
    static func memoryBar(_ m: ActivityFeed.Memory) -> [MemoryPiece] {
        guard m.source == "mlx" else { return [.init(layer: .weights, mb: m.totalMB)] }
        var pieces = [MemoryPiece(layer: .weights, mb: m.weightsMB ?? m.arraysMB ?? 0)]
        // What grew since the runner's last reading is drawn as working memory: it grew while a job ran.
        if m.weightsMB != nil || m.unsplitMB != nil { pieces.append(.init(layer: .working, mb: (m.workingMB ?? 0) + (m.unsplitMB ?? 0))) }
        pieces += [.init(layer: .cache, mb: m.cacheMB ?? 0), .init(layer: .outside, mb: outsideMLX(m))]
        return pieces
    }
    /// The process beyond MLX's parts (and beyond what grew since the runner's last reading).
    static func outsideMLX(_ m: ActivityFeed.Memory) -> Double {
        let arrays = m.weightsMB.map { $0 + (m.workingMB ?? 0) } ?? m.arraysMB ?? 0
        return max(0, m.totalMB - arrays - (m.cacheMB ?? 0) - (m.unsplitMB ?? 0))
    }
    /// The figures under the bar, always the same entries for a model (a value not known yet is a dash): weights, working
    /// memory (its cache included) and the peak; or what MLX holds, its cache and the peak; or the process and its peak.
    /// A nil layer is the peak's mark.
    struct MemoryLegendEntry: Equatable {
        var name: String
        var mb: Double?
        var layer: MemoryLayer?
    }
    static func memoryLegend(_ m: ActivityFeed.Memory) -> [MemoryLegendEntry] {
        let peak = MemoryLegendEntry(name: "Peak", mb: m.peakMB, layer: nil)
        guard m.source == "mlx" else { return [.init(name: "In use", mb: m.totalMB, layer: .weights), peak] }
        if let weights = m.weightsMB {
            return [
                .init(name: "Weights", mb: weights, layer: .weights), .init(name: "Working", mb: (m.workingMB ?? 0) + (m.cacheMB ?? 0) + (m.unsplitMB ?? 0), layer: .working), peak
            ]
        }
        return [.init(name: "In use", mb: m.arraysMB.map { $0 + (m.unsplitMB ?? 0) }, layer: .weights), .init(name: "Cache", mb: m.cacheMB, layer: .cache), peak]
    }
    /// The bar's hover: each part in words and GB, then where the figures come from. Its figures are the ones under the
    /// bar: Working and In use include what grew since the runner's last reading, and the peak is the process's own.
    static func memoryHelp(_ m: ActivityFeed.Memory) -> String {
        func g(_ mb: Double?) -> String { (gb(mb) ?? "\u{2014}") + " GB" }
        var lines: [String] = []
        let grew = m.unsplitMB.map { "Of which \(g($0)) grew during this job since the runner last reported; Arsenal sees it in the process\u{2019}s footprint." }
        if m.source == "mlx" {
            if let weights = m.weightsMB {
                lines.append("Weights \(g(weights)): what MLX held once the model had loaded.")
                lines.append(
                    "Working \(g((m.workingMB ?? 0) + (m.cacheMB ?? 0) + (m.unsplitMB ?? 0))): activations, KV cache or context and scratch \(g(m.workingMB)), and buffers MLX keeps for reuse \(g(m.cacheMB))."
                )
                if let grew { lines.append(grew) }
            } else {
                lines.append("In use \(g(m.arraysMB.map { $0 + (m.unsplitMB ?? 0) })): weights and working memory together; this runner does not measure its weights apart.")
                if let grew { lines.append(grew) }
                lines.append("Cache \(g(m.cacheMB)): buffers MLX keeps for reuse.")
            }
            lines.append("Outside MLX \(g(outsideMLX(m))): the runner's code and CPU memory.")
        } else {
            lines.append("In use \(g(m.totalMB)): this runner\u{2019}s whole process. Its engine does not report MLX\u{2019}s counters, so the parts are not known.")
        }
        let peak = "Peak \(g(m.peakMB)): the most memory the runner\u{2019}s process held during the current or last job (or the load), as macOS counts it."
        lines.append(m.peakMB == nil ? "Peak: not measured yet." : peak)
        lines.append("Measured in the runner\u{2019}s process\(m.source == "mlx" ? " with MLX\u{2019}s counters" : ""), and by Arsenal from macOS while a job runs.")
        return lines.joined(separator: "\n")
    }

    /// A legend entry: the models together (their colours side by side), the app, everything else, free.
    struct LegendEntry: Identifiable, Equatable {
        var id: String { name }
        var name: String
        var mb: Double
        var colours: [NSColor]
        var kind: MemoryPart.Kind
        var help: String?
    }
    static func legend(_ parts: [MemoryPart], models: [MemorySegment], totalMB: Double? = nil) -> [LegendEntry] {
        // A figure in GB tenths as printed. Models adds its cards' printed figures (1.1 + 14.4 + 0.4 = 15.9), not the
        // rounded total of their exact sizes (15.8).
        func tenths(_ mb: Double) -> Double { (mb * 1e6 / 1_073_741_824 * 10).rounded() }
        var entries: [LegendEntry] = []
        if !models.isEmpty {
            entries.append(
                .init(
                    name: "Models", mb: models.reduce(0) { $0 + tenths($1.mb) } / 10 * 1_073_741_824 / 1e6, colours: models.map(\.colour), kind: .model,
                    help: "The loaded models together; each card shows its own."))
        }
        for part in parts where part.kind != .model {
            entries.append(
                .init(
                    name: part.name, mb: part.mb, colours: [part.colour], kind: part.kind,
                    help: part.kind == .arsenal ? "Arsenal itself, without its models." : part.kind == .free ? freeHelp(totalMB: totalMB) : nil))
        }
        // Each figure is rounded to a tenth on its own, so Other takes up the rounding: the legend's used figures add up to
        // the header's exactly as printed (15.5 + 0.3 + 33.0 = 48.8, never 48.9).
        if let other = entries.firstIndex(where: { $0.kind == .other }), entries.allSatisfy({ $0.mb.isFinite }) {
            let used = parts.filter { $0.kind != .free }.reduce(0) { $0 + $1.mb }
            let printed = entries.filter { $0.kind != .free && $0.kind != .other }.reduce(0) { $0 + tenths($1.mb) }
            entries[other].mb = max(0, tenths(used) - printed) / 10 * 1_073_741_824 / 1e6
        }
        return entries
    }
    /// The legend as drawn: with no bar to draw (no total reported) every entry holds its place with a dash, Models
    /// too, so no figure stands beside an empty bar.
    static func shownLegend(parts: [MemoryPart], models: [MemorySegment], totalMB: Double?, p: ArsenalPalette) -> [LegendEntry] {
        guard parts.isEmpty else { return legend(parts, models: models, totalMB: totalMB) }
        return legend(placeholderParts(p), models: models, totalMB: totalMB).map { e in
            var e = e
            if e.kind == .model { e.mb = .nan }
            return e
        }
    }
    struct MemoryPart: Identifiable, Equatable {
        enum Kind: Equatable { case model, arsenal, other, free }
        var id: String
        var name: String
        var mb: Double
        var colour: NSColor
        var kind: Kind
        var fraction: Double
    }
    static func placeholderParts(_ p: ArsenalPalette) -> [MemoryPart] {
        [
            .init(id: "arsenal", name: "App", mb: .nan, colour: ActivityColours.arsenal, kind: .arsenal, fraction: 0),
            .init(id: "other", name: "Other", mb: .nan, colour: ActivityColours.other(p), kind: .other, fraction: 0),
            .init(id: "free", name: freeName, mb: .nan, colour: .clear, kind: .free, fraction: 0)
        ]
    }
    /// The bar's parts in order: each model, the app, everything else (used minus those), system free. Without a total
    /// nothing is drawn; without a used figure only the models are.
    static func memoryParts(machine: ActivityFeed.Machine, models: [MemorySegment], p: ArsenalPalette = .dark) -> [MemoryPart] {
        guard let total = machine.memoryTotalMB, total > 0 else { return [] }
        var parts = models.map { MemoryPart(id: $0.id, name: $0.name, mb: $0.mb, colour: $0.colour, kind: .model, fraction: 0) }
        if let arsenal = machine.arsenalMB { parts.append(.init(id: "arsenal", name: "App", mb: arsenal, colour: ActivityColours.arsenal, kind: .arsenal, fraction: 0)) }
        if let used = machine.memoryUsedMB {
            let counted = parts.reduce(0) { $0 + $1.mb }
            parts.append(.init(id: "other", name: "Other", mb: max(0, used - counted), colour: ActivityColours.other(p), kind: .other, fraction: 0))
            parts.append(.init(id: "free", name: freeName, mb: max(0, total - max(used, counted)), colour: .clear, kind: .free, fraction: 0))
        }
        let sum = max(total, parts.reduce(0) { $0 + $1.mb })
        for i in parts.indices { parts[i].fraction = parts[i].kind == .free ? 0 : parts[i].mb / sum }
        return parts
    }
}
