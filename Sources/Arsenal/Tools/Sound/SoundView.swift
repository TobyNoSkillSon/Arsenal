// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI

// Make a Sound, composed from the tool kit: the band (the model's pop-up of every sound and voice model in the table, its
// audio operations as pills, Show in Finder), the live figures, the options its operation declares with Generate or
// Stop at their foot, and on the right the take: its waveform with Play, Copy and Show in Finder, and this window's
// earlier takes under it. Takes are saved to ~/Music/Arsenal as they arrive.

struct SoundView: View {
    @ObservedObject var session: SoundSession
    static let size = ToolLayout.size, minSize = ToolLayout.minSize

    var body: some View {
        ToolFrame(figures: session.figures(), model: session.model?.model) {
            SessionBand(session: session)
        } controls: {
            OptionsColumn(session: session, run: RunButton(session: session, title: session.runVerb, id: "generate"))
        } result: {
            SoundResult(session: session, player: session.player)
        }
    }
}

// MARK: Result

/// The shown take: Play, its time, the waveform and what it was made from; this window's takes in a list under it (the
/// audio form of the kit's history strip).
struct SoundResult: View {
    @ObservedObject var session: SoundSession
    @ObservedObject var player: SoundPlayer
    @Environment(\.palette) private var p
    static let listHeight: CGFloat = 156

    var body: some View {
        ResultPanel(session: session, symbol: "waveform", empty: session.takes.isEmpty) {
            if !session.takes.isEmpty {
                // The take above, this window's takes under it from the first on: one panel, whose parts keep their size as
                // takes arrive (the list's place is kept, as the image strip's is).
                VStack(spacing: 0) {
                    if let take = session.current { card(take).padding(16).frame(maxHeight: .infinity) }
                    Group {
                        Rectangle().fill(p.hairline.swiftUI).frame(height: 1)
                        takesList.padding(6).frame(height: Self.listHeight)
                    }
                    .historySlot(!session.takes.isEmpty)
                }
            }
        }
    }

    private func card(_ take: SoundTake) -> some View {
        VStack(alignment: .leading, spacing: 14) {
            HStack(spacing: 12) {
                Button {
                    player.toggle(take.url)
                } label: {
                    Image(systemName: player.playing == take.url ? "stop.fill" : "play.fill").font(.system(size: 15)).foregroundStyle(Color.white)
                        .frame(width: 38, height: 38).background(p.accent.swiftUI, in: Circle()).contentShape(Circle())
                }.buttonStyle(.plain).accessibilityLabel(player.playing == take.url ? "Stop" : "Play").controlID("play")
                PlayClock(take: take, player: player).layoutPriority(1)
                // The take's voice or seed and sample rate (its words are the list row below); the words are the hover.
                Text(Self.meta(take, voices: session.voices)).font(.system(size: 12.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
                    .lineLimit(1).truncationMode(.tail).padding(.top, 2)
                    .contentShape(Rectangle()).windowHelp(take.text.isEmpty ? nil : take.text).controlID("take-title")
                Spacer(minLength: 8)
                SaveAsIcon { session.saveAs(take.url) }
                ResultIcon(symbol: "doc.on.doc", help: "Copy the take", id: "copy") { session.copy(take.url) }
            }
            WaveformView(shape: take.shape, url: take.url, player: player)
                .frame(maxWidth: .infinity, maxHeight: .infinity)
                .onDrag { NSItemProvider(contentsOf: take.url) ?? NSItemProvider() }
                .controlID("waveform")
        }
    }

    /// What the take's figures and title leave out: "seed 42 · 44.1 kHz stereo", or "Heart · 24 kHz" for a voice.
    static func meta(_ take: SoundTake, voices: [VoiceChoice]?) -> String {
        var parts: [String] = []
        if let voice = take.voice { parts.append(VoiceChoice.title(voice, voices: voices ?? [])) }
        if let seed = take.seed { parts.append("seed \(seed)") }
        parts.append(OperationForm.number((take.shape.sampleRate / 100).rounded() / 10) + " kHz" + (take.shape.channels == 2 ? " stereo" : ""))
        return parts.joined(separator: " \u{00b7} ")
    }

    private var takesList: some View {
        ScrollViewReader { proxy in
            ScrollView(.vertical) {
                VStack(spacing: 2) {
                    ForEach(session.takes.reversed()) { take in row(take).id(take.id) }
                }
            }
            .scrollIndicators(.automatic)
            .onChange(of: session.takes.count) { _, _ in if let last = session.takes.last { proxy.scrollTo(last.id, anchor: .top) } }
        }
    }

    private func row(_ take: SoundTake) -> some View {
        let on = session.current?.id == take.id
        return HStack(spacing: 10) {
            Button {
                player.toggle(take.url)
            } label: {
                Image(systemName: player.playing == take.url ? "stop.fill" : "play.fill").font(.system(size: 10)).frame(width: 24, height: 24).contentShape(Rectangle())
            }.buttonStyle(.plain).foregroundStyle(p.t1.swiftUI).accessibilityLabel(player.playing == take.url ? "Stop" : "Play").controlID("play-\(take.id)")
            Text(take.text.isEmpty ? take.operation : take.text).font(.system(size: 12.5)).lineLimit(1).frame(maxWidth: .infinity, alignment: .leading)
                .layoutPriority(1)
            // The model's name only tells takes apart when they come from more than one model.
            if Set(session.takes.map(\.model)).count > 1 {
                Text(take.modelName).font(.system(size: 11.5)).foregroundStyle(p.t3.swiftUI).lineLimit(1)
            }
            Text(AudioShape.clock(take.shape.seconds)).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI).frame(width: 40, alignment: .trailing)
        }
        .padding(.horizontal, 8).frame(height: 32)
        .background((on ? p.accent.withAlphaComponent(0.16) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 6))
        .contentShape(Rectangle())
        .onTapGesture { session.show(take.id) }
        .onDrag { NSItemProvider(contentsOf: take.url) ?? NSItemProvider() }
        .accessibilityElement(children: .contain).accessibilityLabel(take.text)
        .controlID("take-\(take.id)")
    }
}

/// "0:03 / 0:10" while playing (ticking with the playhead), else the take's length alone.
struct PlayClock: View {
    let take: SoundTake
    @ObservedObject var player: SoundPlayer
    @Environment(\.palette) private var p
    var body: some View {
        if player.playing == take.url {
            TimelineView(.periodic(from: .now, by: 0.1)) { _ in text(AudioShape.clock(player.elapsed) + " / " + AudioShape.clock(take.shape.seconds)) }
        } else {
            text(AudioShape.clock(take.shape.seconds))
        }
    }
    private func text(_ s: String) -> some View {
        Text(s).font(.system(size: 15, weight: .medium).monospacedDigit()).foregroundStyle(p.t1.swiftUI)
    }
}
