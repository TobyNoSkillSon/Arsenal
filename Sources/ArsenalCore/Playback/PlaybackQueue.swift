// SPDX-License-Identifier: MIT
import AVFoundation
import Foundation

public struct PlaybackItem: Codable, Sendable, Equatable, Identifiable {
    public let id: String
    public let path: String
    public let requester: String
    public let model: String
    public let voice: String?
    public let gainDB: Double
}
public struct PlaybackSnapshot: Codable, Sendable, Equatable {
    public let muted: Bool
    public let current: PlaybackItem?
    public let queued: Int
    public let playing: Bool
}

/// One shared output queue for MCP and the voice menu. Mute pauses without losing the queue; Stop clears it.
/// No audio engine or device is opened while muted. Generated files remain at the model's original level.
@MainActor public final class PlaybackQueue: NSObject, AVAudioPlayerDelegate {
    public static let shared = PlaybackQueue(
        preference: ArsenalPaths().support.appendingPathComponent("playback-muted.json"),
        forceMuted: ProcessInfo.processInfo.environment["ARSENAL_PLAYBACK_MUTED"] == "1")
    public private(set) var muted: Bool
    private var items: [PlaybackItem] = []
    private var player: AVAudioPlayer?
    private let outputEnabled: Bool
    private var preference: URL?
    private let meter: @Sendable (URL) async throws -> Double
    private var stopGeneration = 0
    private var subscribers: [UUID: AsyncStream<PlaybackSnapshot>.Continuation] = [:]
    public convenience init(muted: Bool = false) { self.init(muted: muted, outputEnabled: true) }
    init(muted: Bool, outputEnabled: Bool, meter: (@Sendable (URL) async throws -> Double)? = nil) {
        self.muted = muted; self.outputEnabled = outputEnabled
        self.meter = meter ?? { path in try await Task.detached { try Self.measureGain(path) }.value }
        super.init()
    }
    /// Read the choice synchronously, before the queue can accept any speech. A corrupt choice fails muted.
    public convenience init(preference: URL, forceMuted: Bool = false, outputEnabled: Bool = true) {
        let saved: Bool
        if FileManager.default.fileExists(atPath: preference.path) {
            saved = (try? JSONDecoder().decode(Bool.self, from: Data(contentsOf: preference))) ?? true
        } else {
            saved = false
        }
        self.init(muted: forceMuted || saved, outputEnabled: outputEnabled)
        self.preference = preference
    }
    public func snapshot() -> PlaybackSnapshot { .init(muted: muted, current: items.first, queued: max(0, items.count - 1), playing: player?.isPlaying == true) }
    public func events() -> AsyncStream<PlaybackSnapshot> {
        let id = UUID()
        return AsyncStream(bufferingPolicy: .bufferingNewest(1)) { continuation in
            subscribers[id] = continuation; continuation.yield(snapshot())
            continuation.onTermination = { @Sendable _ in Task { @MainActor in self.subscribers.removeValue(forKey: id) } }
        }
    }
    @discardableResult public func enqueue(path: URL, requester: String, model: String, voice: String? = nil) async throws -> PlaybackItem? {
        guard !muted else { return nil }
        let generation = stopGeneration
        guard items.count < 64 else { throw WorkerFailure(code: "busy", message: "Playback queue is full; Stop it before speaking again.") }
        let gain = try await meter(path)
        guard generation == stopGeneration, !muted else { return nil }
        guard items.count < 64 else { throw WorkerFailure(code: "busy", message: "Playback queue is full; Stop it before speaking again.") }
        let item = PlaybackItem(id: UUID().uuidString, path: path.path, requester: requester, model: model, voice: voice, gainDB: gain)
        items.append(item)
        do { try startNext() } catch { items.removeAll { $0.id == item.id }; publish(); throw error }
        publish(); return item
    }
    nonisolated static func measureGain(_ path: URL) throws -> Double {
        let file = try AVAudioFile(forReading: path)
        guard file.length > 0, file.length <= 20_000_000,
            let buffer = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: AVAudioFrameCount(file.length))
        else {
            throw WorkerFailure(code: "audio", message: "Audio is empty or too large for playback; read the returned file instead.")
        }
        try readEveryFrame(file, into: buffer)
        guard AVAudioFramePosition(buffer.frameLength) == file.length else {
            throw WorkerFailure(code: "audio", message: "Audio ended after \(buffer.frameLength) of \(file.length) frames.")
        }
        guard let channels = buffer.floatChannelData else { throw WorkerFailure(code: "audio", message: "Audio must decode to floating point PCM.") }
        let count = Int(buffer.frameLength), channelCount = Int(buffer.format.channelCount)
        var samples = [Float](); samples.reserveCapacity(count * channelCount)
        for frame in 0..<count { for channel in 0..<channelCount { samples.append(channels[channel][frame]) } }
        let measurement = PlaybackLoudness.measure(samples, rate: Int(buffer.format.sampleRate), channels: channelCount)
        return min(0, PlaybackLoudness.gainDB(loudness: measurement.integrated, truePeak: measurement.truePeak))
    }
    public func setMuted(_ value: Bool) throws {
        // Keep the previous choice if persistence fails, and report the filesystem error to the caller.
        if let preference {
            try FileManager.default.createDirectory(at: preference.deletingLastPathComponent(), withIntermediateDirectories: true)
            try JSONEncoder().encode(value).write(to: preference, options: .atomic)
        }
        muted = value
        if value { stopGeneration += 1; player?.pause() } else if let player { player.play() } else { try startNext() }
        publish()
    }
    public func stop() { stopGeneration += 1; player?.stop(); player = nil; items.removeAll(); publish() }
    private func startNext() throws {
        guard outputEnabled, !muted, player == nil, let item = items.first else { return }
        let next = try AVAudioPlayer(contentsOf: URL(fileURLWithPath: item.path))
        next.delegate = self; next.volume = Float(pow(10, item.gainDB / 20))
        guard next.play() else { throw WorkerFailure(code: "playback", message: "Audio output is unavailable; read the returned file or Mute playback.") }
        player = next
    }
    private func publish() { let state = snapshot(); for continuation in subscribers.values { continuation.yield(state) } }
    public nonisolated func audioPlayerDidFinishPlaying(_ player: AVAudioPlayer, successfully flag: Bool) {
        let identity = ObjectIdentifier(player)
        Task { @MainActor in self.advance(identity: identity) }
    }
    public nonisolated func audioPlayerDecodeErrorDidOccur(_ player: AVAudioPlayer, error: (any Error)?) {
        let identity = ObjectIdentifier(player)
        Task { @MainActor in self.advance(identity: identity) }
    }
    private func advance(identity: ObjectIdentifier) {
        guard let player, ObjectIdentifier(player) == identity else { return }
        self.player = nil; if !items.isEmpty { items.removeFirst() }
        while !items.isEmpty {
            do { try startNext(); break } catch { items.removeFirst() }
        }
        publish()
    }
}

/// Fills `buffer` with every frame of `file`. One `read(into:)` can stop short of `length` (float WAVs written
/// by libsndfile come back hundreds of frames short), so read in blocks and copy until the buffer is full.
private func readEveryFrame(_ file: AVAudioFile, into buffer: AVAudioPCMBuffer) throws {
    buffer.frameLength = 0
    guard let block = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536), let target = buffer.floatChannelData else {
        return
    }
    while file.framePosition < file.length, buffer.frameLength < buffer.frameCapacity {
        let room = AVAudioFramePosition(buffer.frameCapacity - buffer.frameLength)
        try file.read(into: block, frameCount: AVAudioFrameCount(min(65_536, file.length - file.framePosition, room)))
        guard block.frameLength > 0, let source = block.floatChannelData else { break }
        for channel in 0..<Int(buffer.format.channelCount) {
            (target[channel] + Int(buffer.frameLength)).update(from: source[channel], count: Int(block.frameLength))
        }
        buffer.frameLength += block.frameLength
    }
}
