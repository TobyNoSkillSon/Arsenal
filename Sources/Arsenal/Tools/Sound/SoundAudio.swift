// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation
import SwiftUI

// A take's audio as the window shows and plays it: its length and a peak envelope read once from the file, and one
// player for the window. Playback is the window's own (AVAudioPlayer), separate from the speech queue agents use, so
// Mute in the menu does not silence a take you pressed Play on.

/// A clip's length and its loudness envelope (peaks in 0…1, one per bucket), read once when the take arrives.
struct AudioShape: Equatable {
    var seconds: Double
    var sampleRate: Double
    var channels: Int
    var peaks: [Float]

    static let buckets = 240

    /// Reads the whole file in blocks (a 380 s stereo take at 44.1 kHz is about 33 M samples; nothing is kept but the
    /// peaks). Nil when the file is not audio AVFoundation can read.
    static func read(_ url: URL, buckets: Int = AudioShape.buckets) -> AudioShape? {
        guard let file = try? AVAudioFile(forReading: url) else { return nil }
        let format = file.processingFormat
        let frames = file.length
        guard format.sampleRate > 0 else { return nil }
        let seconds = Double(frames) / format.sampleRate
        var peaks = [Float](repeating: 0, count: max(1, buckets))
        guard frames > 0 else { return AudioShape(seconds: 0, sampleRate: format.sampleRate, channels: Int(format.channelCount), peaks: peaks) }
        let block: AVAudioFrameCount = 32_768
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: block) else { return nil }
        var position: AVAudioFramePosition = 0
        while position < frames {
            do { try file.read(into: buffer, frameCount: block) } catch { break }
            let n = Int(buffer.frameLength)
            guard n > 0, let data = buffer.floatChannelData else { break }
            for c in 0..<Int(format.channelCount) {
                let samples = data[c]
                for i in 0..<n {
                    let bucket = min(peaks.count - 1, Int((Int64(i) + position) * Int64(peaks.count) / frames))
                    let v = abs(samples[i])
                    if v > peaks[bucket] { peaks[bucket] = v }
                }
            }
            position += AVAudioFramePosition(n)
        }
        return AudioShape(seconds: seconds, sampleRate: format.sampleRate, channels: Int(format.channelCount), peaks: peaks.map { min(1, $0) })
    }

    /// "0:05", "1:02", "0:00.4" for a clip under a second.
    static func clock(_ s: Double) -> String {
        guard s.isFinite, s >= 0 else { return "\u{2014}" }
        if s > 0, s < 1 { return String(format: "0:00.%d", Int((s * 10).rounded(.down))) }
        let whole = Int(s.rounded(.down))
        return "\(whole / 60):" + String(format: "%02d", whole % 60)
    }
}

/// The window's player: one take at a time; starting another stops the first.
@MainActor final class SoundPlayer: NSObject, ObservableObject, AVAudioPlayerDelegate {
    /// The take playing (nil at rest).
    @Published private(set) var playing: URL?
    private var player: AVAudioPlayer?
    /// Tests and renders play at zero volume: nothing is heard on the Mac they run on.
    nonisolated(unsafe) static var silent = false

    func play(_ url: URL, from fraction: Double = 0) {
        stop()
        guard let p = try? AVAudioPlayer(contentsOf: url) else { return }
        p.delegate = self
        if Self.silent { p.volume = 0 }
        p.currentTime = max(0, min(1, fraction)) * p.duration
        guard p.play() else { return }
        player = p; playing = url
    }
    func stop() {
        player?.stop(); player = nil
        if playing != nil { playing = nil }
    }
    func toggle(_ url: URL) {
        if playing == url { stop() } else { play(url) }
    }
    /// How far the playing take is, 0…1 (read by the waveform only while it plays).
    var fraction: Double {
        guard let p = player, p.duration > 0 else { return 0 }
        return min(1, p.currentTime / p.duration)
    }
    var elapsed: Double { player?.currentTime ?? 0 }

    nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let id = ObjectIdentifier(player)
        Task { @MainActor in
            if let current = self.player, ObjectIdentifier(current) == id { self.player = nil; self.playing = nil }
        }
    }
}

/// The take's envelope as mirrored bars; the played part in the accent. Click anywhere to play from there.
struct WaveformView: View {
    let shape: AudioShape
    let url: URL
    @ObservedObject var player: SoundPlayer
    @Environment(\.palette) private var p

    var body: some View {
        // Only a playing take redraws (its playhead); at rest the bars are drawn once.
        if player.playing == url {
            TimelineView(.animation) { _ in bars(played: player.fraction) }
        } else {
            bars(played: nil)
        }
    }

    private func bars(played: Double?) -> some View {
        GeometryReader { g in
            Canvas(rendersAsynchronously: false) { context, size in
                let n = shape.peaks.count
                guard n > 0 else { return }
                let step = size.width / CGFloat(n)
                let bar = max(1, step * 0.62)
                let mid = size.height / 2
                var before = Path(), after = Path()
                for (i, peak) in shape.peaks.enumerated() {
                    let h = max(1.5, CGFloat(peak) * (size.height - 4))
                    let x = CGFloat(i) * step + (step - bar) / 2
                    let rect = CGRect(x: x, y: mid - h / 2, width: bar, height: h)
                    if let played, (CGFloat(i) + 0.5) / CGFloat(n) <= played {
                        before.addRoundedRect(in: rect, cornerSize: CGSize(width: bar / 2, height: bar / 2))
                    } else {
                        after.addRoundedRect(in: rect, cornerSize: CGSize(width: bar / 2, height: bar / 2))
                    }
                }
                context.fill(after, with: .color((played == nil ? p.t2 : p.t3).swiftUI))
                context.fill(before, with: .color(p.accent.swiftUI))
            }
            .contentShape(Rectangle())
            .onTapGesture(coordinateSpace: .local) { location in
                player.play(url, from: Double(location.x / max(1, g.size.width)))
            }
        }
        .accessibilityElement()
        .accessibilityLabel("Waveform, \(AudioShape.clock(shape.seconds))")
        .accessibilityAddTraits(.isButton)
        .accessibilityAction { player.toggle(url) }
    }
}
