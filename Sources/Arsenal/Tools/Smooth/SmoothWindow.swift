// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// Smooth Video: one window for every model that declares `interpolate_video`. A video, or two frames to make the
// frames between; the factor, the frame rate to reach or the positions between the frames, as the model takes them;
// the smoothed video playing (the original one click away), or the made frames in order. A job: its progress shows,
// Stop cancels it. Videos are saved to ~/Movies/Arsenal.

@MainActor final class SmoothSession: CapabilitySession {
    @Published var showsOriginal = false

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .smooth, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["factor": "Factor", "target_fps": "Frame rate", "t": "Positions", "frames": "Frames"] }
    override var resultNoun: String { "video" }
    /// The factor reads as one: "4×".
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = super.makeForm(op)
        form.update("factor") { $0.unit = "\u{00d7}" }
        return form
    }
    override func groups(_ op: OperationDescriptor) -> [OptionGroup] {
        [OptionGroup(id: "rate", options: [.init(title: "Factor", keys: ["factor"]), .init(title: "Frame rate", keys: ["target_fps"])])]
    }
    /// Positions between two frames mean something only for two frames.
    override func hiddenOutsideGroups(_ key: String) -> Bool { key == "t" && value("frames").files.isEmpty }
    override var idleLine: String { "Smoothing" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Smooth" }
    override func check(_ outcome: Outcome) throws {
        guard !outcome.files.isEmpty else { throw Undelivered(.init(message: "\(outcome.modelName) returned no video or frames.", action: nil)) }
    }
    /// The result's frame rate is read from its file once it arrives.
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        try super.deliver(result, op: op, model: model, input: input)
        if let video = current?.file("video/") {
            Task {
                _ = await VideoInfo.load(video.url); objectWillChange.send()
            }
        }
    }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        let rate = outcome?.file("video/").flatMap { VideoInfo.cachedRate($0.url) }
        return [Figure(title: "Frame rate", value: rate.map { String(format: "%.0f fps", $0) }, reference: nil, help: "Frames per second of the result.", width: 84)]
    }
}

struct SmoothView: View {
    @ObservedObject var session: SmoothSession
    var body: some View {
        CapabilityView(session: session) {
            OutcomePanel(
                session: session, what: "video", actions: { o in actions(o) },
                content: { o in
                    Group {
                        if let video = o.file("video/") {
                            VideoResult(url: session.showsOriginal ? (o.inputs.first ?? video.url) : video.url)
                                .task(id: video.url) { _ = await VideoInfo.load(video.url) }
                        } else {
                            let frames = o.files.filter(\.isImage)
                            ScrollView(.vertical) {
                                LazyVGrid(columns: [GridItem(.adaptive(minimum: 150), spacing: 8)], spacing: 8) {
                                    ForEach(o.inputs.prefix(1) + frames.map(\.url) + o.inputs.dropFirst(), id: \.self) { url in
                                        Thumbnail(url: url, fit: true).frame(height: 110).frame(maxWidth: .infinity)
                                            .clipShape(RoundedRectangle(cornerRadius: 6)).opacity(o.inputs.contains(url) ? 0.6 : 1)
                                            .onDrag { NSItemProvider(contentsOf: url) ?? NSItemProvider() }
                                    }
                                }
                            }
                        }
                    }
                    .padding(.horizontal, 16).padding(.bottom, 16)
                })
        }
    }
    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        if o.file("video/") != nil, let first = o.inputs.first, FileAccept(mimes: ["video/*"]).accepts(first) {
            Pill(title: "Smoothed", on: !session.showsOriginal, enabled: true, help: "The smoothed video") { session.showsOriginal = false }.controlID("show-result")
            Pill(title: "Original", on: session.showsOriginal, enabled: true, help: "The video as it was") { session.showsOriginal = true }.controlID("show-original")
        }
    }
}
