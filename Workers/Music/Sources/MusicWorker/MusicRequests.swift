// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import ArsenalWire

enum MusicRequests {
    /// The host's per-job artifact limit: files[] artifacts written to the job directory are bounded on disk per job,
    /// not by the 64 MiB inline response cap. Mirrors the host constant; keep the two equal.
    static let artifactLimit: Int64 = 4 * 1024 * 1024 * 1024

    /// Upper bound on the audio files a request can write, checked before any generation starts: stereo 48 kHz,
    /// 16-bit (32-bit for wav32) PCM per second of the longest song the engine may write, plus 1% and 64 KiB per file
    /// for container and FLAC/ALAC frame overhead. AAC is far smaller; the bound stays conservative for it.
    static func outputBound(_ request: SongRequest, format: AudioFileFormat, maxSeconds: Double) -> Int64 {
        let seconds = Swift.min(request.duration ?? maxSeconds, maxSeconds)
        let perSecond: Double = format == .wav32 ? 384_000 : 192_000
        let pcm = Int64((seconds * perSecond).rounded(.up))
        let perSong = pcm + pcm / 100 + 65_536
        return perSong * Int64(request.batchSize)
    }

    static func decode(_ input: JSONObject, maxSeconds: Double) throws -> SongRequest {
        guard let prompt = input["prompt"]?.string, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, prompt.count <= 10000 else {
            throw MusicSupport.failure("A nonempty music style prompt is required.")
        }
        var fields = input
        fields["caption"] = .string(prompt); fields.removeValue(forKey: "prompt"); fields.removeValue(forKey: "out_dir"); fields.removeValue(forKey: "format")
        for key in ["source_audio", "reference_audio"] {
            if let file = fields[key] {
                guard let path = file.object?["path"]?.string, !path.isEmpty else { throw MusicSupport.failure(key + " requires a file path.") }; fields[key] = .string(path)
            }
        }
        let request = try JSONDecoder().decode(SongRequest.self, from: WireJSON.encode(fields))
        guard (1...16).contains(request.batchSize), request.steps >= 1, request.steps <= 200, request.lyrics.count <= 100000 else {
            throw MusicSupport.failure("Music batch, step or lyric bounds exceeded.")
        }
        guard let format = AudioFileFormat(rawValue: input["format"]?.string ?? "wav") else { throw MusicSupport.failure("Unsupported music audio format.") }
        let bound = outputBound(request, format: format, maxSeconds: maxSeconds)
        guard bound <= artifactLimit else {
            let seconds = Int(Swift.min(request.duration ?? maxSeconds, maxSeconds).rounded(.up))
            throw MusicSupport.failure(
                "\(request.batchSize) songs of up to \(seconds) s as \(format.rawValue) need up to \(bound / 1_048_576) MiB, over the "
                    + "\(artifactLimit / 1_048_576) MiB per-request output limit. Ask for fewer or shorter songs, or flac or m4a.")
        }
        return request
    }
    static func files(_ songs: [GeneratedSong], input: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        guard let directory = input["out_dir"]?.string, let format = AudioFileFormat(rawValue: input["format"]?.string ?? "wav"), !songs.isEmpty else {
            throw MusicSupport.failure("Music output requires out_dir and a supported audio format.")
        }
        let root = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var files: [JSONValue] = []
        for (index, song) in songs.enumerated() {
            try cancellation.check()
            guard song.audio.frameCount > 0, song.audio.channels.count == 2, song.audio.channels.allSatisfy({ $0.count == song.audio.frameCount && $0.allSatisfy(\.isFinite) })
            else { throw MusicSupport.failure("Music model returned empty or non-finite stereo audio.") }
            let name = "song-\(index)-" + UUID().uuidString
            let url = root.appendingPathComponent(name + "." + format.fileExtension)
            try AudioFiles.write(song.audio, to: url, format: format)
            try cancellation.check()
            let mime: String = format == .flac ? "audio/flac" : (format == .m4a || format == .alac ? "audio/mp4" : "audio/wav")
            files.append(.object(try NativeMedia.file(url, root: root, mime: mime, id: "audio-\(index)")))
            if input["include_score"]?.bool == true, let score = song.plan?.score {
                let url = root.appendingPathComponent(name + ".abc"); try Data(score.utf8).write(to: url)
                files.append(.object(try NativeMedia.file(url, root: root, mime: "text/plain", id: "score-\(index)")))
            }
            if input["include_metadata"]?.bool == true {
                let metadata = root.appendingPathComponent(name + ".json")
                let encoder = JSONEncoder(); encoder.outputFormatting = [.sortedKeys]
                struct Receipt: Encodable { let request: SongRequest; let plan: SongPlan?; let audioCodes: [Int]?; let truncated: Bool }
                try encoder.encode(Receipt(request: song.resolved, plan: song.plan, audioCodes: song.audioCodes, truncated: song.truncated)).write(to: metadata)
                files.append(.object(try NativeMedia.file(metadata, root: root, mime: "application/json", id: "metadata-\(index)")))
            }
        }
        return ["files": .array(files)]
    }
}
