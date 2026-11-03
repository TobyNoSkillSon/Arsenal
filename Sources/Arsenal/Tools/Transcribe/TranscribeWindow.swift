// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// Transcribe: one window for every model that declares `transcribe` or `diarize`. A clip dropped, chosen or pasted;
// the operation's own options (language, word times, speakers, translate to English, words to expect); the transcript
// with each speaker in a colour and each line's time, which plays the clip from there; Copy and Save as text or
// subtitles. Live: the microphone streamed to the model, a line per turn (a turn ends where an installed end-of-turn
// model says, else at a pause); with a diarizer that streams (`diarize_session`), who spoke when as it grows.

@MainActor final class TranscribeSession: CapabilitySession {
    let live = LiveTranscription()
    override func outputText(_ o: Outcome) -> String? {
        let s = labelled(TranscriptSegment.list(o.reply)); return TranscriptSegment.text(s, speakers: speakers(s, o))
    }
    override var outputNoun: String { "Transcript" }
    /// Speakers are labelled only while Label speakers is on, so the result never says "2 speakers" beside a switch that
    /// is off (a form without the switch shows them as the model answered).
    var labelsSpeakers: Bool { form?.fields.contains { $0.key == "diarize" } != true || value("diarize") == .toggle(true) }
    func labelled(_ segments: [TranscriptSegment]) -> [TranscriptSegment] {
        labelsSpeakers
            ? segments
            : segments.map { s in
                var s = s; s.speaker = nil; return s
            }
    }
    func speakers(_ segments: [TranscriptSegment], _ o: Outcome) -> [String] { labelsSpeakers ? TranscriptSegment.speakers(segments, reply: o.reply) : [] }
    /// Installed models that decide where a turn ends (`end_of_turn`), for live mode.
    var turnModels: [String] = []
    /// Live mode is offered only when the app may use the microphone (tests turn it on).
    var liveAvailable = MicrophoneSource.declared
    @Published var liveMode = false

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .transcribe, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }

    override func titles(_ op: OperationDescriptor) -> [String: String] { ["prompt": "Words to expect", "task": "Output", "diarize": "Label speakers"] }
    /// What comes out, in words: the clip's own language (what the model does unless asked) or an English translation.
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = super.makeForm(op)
        form.update("task") { f in
            f.choiceTitles = ["transcribe": "Original language", "translate": "English translation"]
            if f.defaultValue == nil { f.defaultValue = "transcribe" }
        }
        return form
    }
    /// The transcript, or only who spoke when.
    override func pillTitle(_ op: OperationDescriptor) -> String { op.name == "diarize" ? "Speaker timing" : op.name == "transcribe" ? "Transcript" : super.pillTitle(op) }
    override var resultNoun: String { "transcript" }
    override var idleLine: String { descriptor?.name == "diarize" ? "Finding speakers" : "Transcribing" }
    override func runTitle(_ op: OperationDescriptor?) -> String { op?.name == "diarize" ? "Find Speakers" : "Transcribe" }

    override func check(_ outcome: Outcome) throws {
        if TranscriptSegment.list(outcome.reply).isEmpty, outcome.reply["text"]?.string?.isEmpty ?? true, outcome.operation == "transcribe" {
            throw Undelivered(.init(message: "\(outcome.modelName) heard no speech in \(outcome.inputs.first?.lastPathComponent ?? "the clip").", action: nil))
        }
    }

    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        let length = outcome?.reply["duration"]?.number ?? outcome?.inputs.first.flatMap { AudioCache.shape($0)?.seconds }
        let speed = outcome.flatMap { o in length.map { $0 / max(0.001, o.seconds) } }
        return [
            Figure(title: "Length", value: FigureText.seconds(length), reference: nil, help: "How long the clip plays.", width: 84),
            Figure(
                title: "vs real time", value: speed.map { String(format: $0 < 10 ? "%.1f" : "%.0f", $0) + "\u{00d7}" }, reference: nil,
                help: "Clip length divided by time: above 1\u{00d7} the model transcribes faster than the clip plays.", width: 84)
        ]
    }

    // MARK: Live

    /// Live mode is offered for a model that transcribes or streams speakers.
    var offersLive: Bool { operations.contains { ["transcribe", "transcribe_session", "diarize_session"].contains($0.name) } }
    /// The Realtime socket streams a diarizer's session only when the model has no transcription session of its own.
    var liveDiarizes: Bool { operations.contains { $0.name == "diarize_session" } && !operations.contains { $0.name == "transcribe_session" } }

    func startLive() {
        guard let client = client(), let model, !running else { return }
        let language = form?.fields.first {
            if case .language = $0.control { return true }; return false
        }.map { value($0.key).text }
        let diarize = liveDiarizes, turnModel = diarize ? nil : turnModels.first
        // Load before listening, so not having the memory for it shows the same warning and Load Anyway as every other load.
        let prepare: () async throws -> Void = {
            for id in [model.id] + (turnModel.map { [$0] } ?? []) {
                _ = try await client.send("POST", "/arsenal/v1/load", json: ["id": id], timeout: 3600)
            }
        }
        live.start(port: client.port, model: model.id, language: language, turnModel: turnModel, diarize: diarize, prepare: prepare)
    }
    /// Stops listening; the session is kept once its last turn has arrived.
    func stopLive() {
        guard let model else { live.stop(); return }
        live.stop { [weak self] live in self?.keep(live, model: model) }
    }
    private func keep(_ live: LiveTranscription, model: ModelChoice) {
        var reply: JSONObject = [:]
        var text = live.text
        if let timeline = live.timeline, let segments = timeline["segments"], segments.array?.isEmpty == false {
            reply["segments"] = segments; reply["speakers"] = timeline["speakers"] ?? .array([])
            let list = TranscriptSegment.list(reply)
            if text.isEmpty { text = TranscriptSegment.text(list, speakers: TranscriptSegment.speakers(list, reply: reply)) }
        }
        guard !text.isEmpty else { return }
        reply["text"] = .string(text)
        append(
            Outcome(
                model: model.id, modelName: model.name, operation: "live", operationTitle: "Live", inputs: [], text: text, values: [:], input: [:],
                reply: reply, files: [], seconds: live.started.map { Date().timeIntervalSince($0) } ?? 0))
    }
    override func close() {
        live.close()
        super.close()
    }
}

struct TranscribeView: View {
    @ObservedObject var session: TranscribeSession
    @ObservedObject var live: LiveTranscription

    init(session: TranscribeSession) { self.session = session; live = session.live }

    var body: some View {
        ToolFrame(figures: session.figures(), model: session.model?.model) {
            SessionBand(session: session, accessory: liveButton)
        } controls: {
            if session.liveMode {
                LiveColumn(session: session, live: live)
            } else {
                OptionsColumn(session: session, run: RunButton(session: session, title: session.runTitle(session.descriptor), id: "run"), leadFills: !session.subjectFields.isEmpty)
                {
                    SubjectLead(session: session, fills: session.optionFields.isEmpty)
                }
            }
        } result: {
            if session.liveMode, live.listening || live.stopping || session.current == nil {
                VStack(spacing: 12) {
                    LiveTranscript(live: live)
                    // Where the other windows keep their history, so every panel ends at the same line.
                    Color.clear.frame(height: ImageStrip.height)
                }
            } else {
                TranscriptResult(session: session)
            }
        }
    }

    /// The microphone: live transcription, offered when the app may use it and the model transcribes or streams speakers.
    private var liveButton: AnyView? {
        guard session.liveAvailable, session.offersLive else { return nil }
        return AnyView(
            // On while live mode is on (accent ground, as a chosen pill), never greyed by it: a click leaves live mode,
            // stopping a listen first.
            BandButton(
                symbol: session.liveMode ? "mic.fill" : "mic", help: session.liveMode ? "Transcribe a file" : "Transcribe live from the microphone",
                enabled: !session.running, on: session.liveMode
            ) {
                if live.listening { session.stopLive() }
                session.liveMode.toggle()
            }
            .controlID("live"))
    }
}

/// Live mode's column: the one button that starts and stops listening, the input's level while it listens.
struct LiveColumn: View {
    @ObservedObject var session: TranscribeSession
    @ObservedObject var live: LiveTranscription
    @Environment(\.palette) private var p

    var body: some View {
        VStack(spacing: 18) {
            Spacer(minLength: 0)
            Button {
                if live.listening { session.stopLive() } else { session.startLive() }
            } label: {
                ZStack {
                    Circle().strokeBorder(p.accent.withAlphaComponent(live.listening ? 0.35 : 0).swiftUI, lineWidth: 6 + CGFloat(live.level) * 14)
                    Circle().fill((live.listening ? p.red : p.accent).swiftUI).padding(14)
                    Image(systemName: live.listening ? "stop.fill" : "mic.fill").font(.system(size: 26, weight: .semibold)).foregroundStyle(Color.white)
                }
                .frame(width: 112, height: 112).contentShape(Circle())
            }
            .buttonStyle(.plain).disabled(session.model == nil || session.blocked != nil || live.stopping)
            .accessibilityLabel(live.listening ? "Stop listening" : "Start listening").controlID(live.listening ? "live-stop" : "live-start")
            if let started = live.started, live.listening {
                TimelineView(.periodic(from: started, by: 1)) { context in
                    Text(TranscriptSegment.time(context.date.timeIntervalSince(started))).font(.system(size: 15, weight: .medium).monospacedDigit())
                }
            }
            Spacer(minLength: 0)
            if let problem = live.problem {
                ProblemLine(problem: .init(message: problem, action: nil)) { _ in }
            } else {
                SessionProblem(session: session)
            }
        }
        .frame(maxWidth: .infinity)
    }
}

/// What live mode has heard: a line per turn, the turn in progress fainter; from a diarizer, who spoke when so far.
struct LiveTranscript: View {
    @ObservedObject var live: LiveTranscription
    @Environment(\.palette) private var p
    var body: some View {
        let segments = live.timeline.map(TranscriptSegment.list) ?? []
        let speakers = live.timeline.map { TranscriptSegment.speakers(segments, reply: $0) } ?? []
        ZStack {
            RoundedRectangle(cornerRadius: 10).fill(p.sidebar.swiftUI)
            if live.lines.isEmpty, live.partial.isEmpty, segments.isEmpty {
                Image(systemName: "mic").font(.system(size: 40, weight: .light)).foregroundStyle(p.t4.swiftUI).accessibilityHidden(true)
            } else {
                ScrollViewReader { proxy in
                    ScrollView(.vertical) {
                        VStack(alignment: .leading, spacing: 10) {
                            if !segments.isEmpty {
                                // Each snapshot replaces the last whole: earlier stretches can still grow or merge.
                                SpeakerTimeline(segments: segments, speakers: speakers, length: segments.map(\.end).max() ?? 0)
                                ForEach(segments) { s in
                                    HStack(spacing: 10) {
                                        Circle().fill(s.speaker.map { SpatialColours.colour($0, in: speakers) } ?? p.accent.swiftUI).frame(width: 8, height: 8)
                                        Text(s.speaker.map { TranscriptSegment.speakerName($0, in: speakers) } ?? "Speaker").font(.system(size: 14, weight: .medium))
                                        Text(TranscriptSegment.time(s.start) + " \u{2013} " + TranscriptSegment.time(s.end))
                                            .font(.system(size: 13).monospacedDigit()).foregroundStyle(p.t3.swiftUI)
                                    }
                                }
                            }
                            ForEach(Array(live.lines.enumerated()), id: \.offset) { _, line in
                                Text(line).font(.system(size: 14)).lineSpacing(2).textSelection(.enabled).fixedSize(horizontal: false, vertical: true)
                            }
                            if !live.partial.isEmpty { Text(live.partial).font(.system(size: 14)).foregroundStyle(p.t3.swiftUI) }
                            Color.clear.frame(height: 1).id("end")
                        }.padding(16).frame(maxWidth: .infinity, alignment: .leading)
                    }
                    .onChange(of: live.lines.count) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
                    .onChange(of: segments.count) { _, _ in proxy.scrollTo("end", anchor: .bottom) }
                }
            }
        }
    }
}

/// The shown transcript: language and speakers, who spoke when, the lines; Copy, Save as text or subtitles.
struct TranscriptResult: View {
    @ObservedObject var session: TranscribeSession
    @Environment(\.palette) private var p

    var body: some View {
        OutcomePanel(
            session: session, what: "transcript",
            copyText: { o in
                let s = session.labelled(TranscriptSegment.list(o.reply)); return TranscriptSegment.text(s, speakers: session.speakers(s, o))
            },
            actions: { o in actions(o) }, trailing: { o in AnyView(saves(o)) },
            content: { o in content(o) })
    }

    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        let segments = session.labelled(TranscriptSegment.list(o.reply)), speakers = session.speakers(segments, o)
        if let language = LanguageList.detected(o.reply["language"]?.string) {
            Text(LanguageList.name(language)).font(.system(size: 12, weight: .medium)).foregroundStyle(p.t2.swiftUI)
        }
        if speakers.count > 0 {
            Label(speakers.count == 1 ? "1 speaker" : "\(speakers.count) speakers", systemImage: "person.2").font(.system(size: 12)).foregroundStyle(p.t2.swiftUI)
        }
    }

    @ViewBuilder private func saves(_ o: Outcome) -> some View {
        let segments = session.labelled(TranscriptSegment.list(o.reply)), speakers = session.speakers(segments, o)
        ResultIcon(symbol: "square.and.arrow.down", help: "Save as text", id: "save-text", enabled: !session.running) {
            _ = session.save(text: TranscriptSegment.text(segments, speakers: speakers), name: OutcomeHistory.title(o), ext: "txt")
        }
        if segments.contains(where: \.timed) {
            ResultIcon(symbol: "captions.bubble", help: "Save as subtitles (SRT)", id: "save-srt", enabled: !session.running) {
                _ = session.save(text: TranscriptSegment.srt(segments, speakers: speakers), name: OutcomeHistory.title(o), ext: "srt")
            }
        }
    }

    private func content(_ o: Outcome) -> some View {
        let segments = session.labelled(TranscriptSegment.list(o.reply)), speakers = session.speakers(segments, o)
        let length = o.reply["duration"]?.number ?? o.inputs.first.flatMap { AudioCache.shape($0)?.seconds } ?? segments.map(\.end).max() ?? 0
        return VStack(alignment: .leading, spacing: 10) {
            if speakers.count > 1, length > 0 {
                SpeakerTimeline(segments: segments, speakers: speakers, length: length).padding(.horizontal, 16)
            }
            TranscriptList(segments: segments, speakers: speakers, audio: o.inputs.first, player: session.player)
        }
        .padding(.bottom, 8)
    }
}
