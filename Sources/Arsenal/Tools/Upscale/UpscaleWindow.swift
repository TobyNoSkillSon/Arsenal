// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// Upscale: one window for every model that declares `upscale_image` or `upscale_video`. A picture or a video; the
// scale or the height to reach, a prompt and seed when the model takes them; the result over the original under a
// handle (a picture) or playing (a video), with both sizes. Long runs are jobs: their progress shows and Stop cancels
// them. Pictures are saved to ~/Pictures/Arsenal, videos to ~/Movies/Arsenal.

@MainActor final class UpscaleSession: CapabilitySession {
    /// For a video: the result, or the original, on show.
    @Published var showsOriginal = false

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .upscale, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["scale": "Scale", "target_height": "Height", "prompt": "Prompt"] }
    override var resultNoun: String { "upscale" }
    override func groups(_ op: OperationDescriptor) -> [OptionGroup] {
        [OptionGroup(id: "size", options: [.init(title: "Scale", keys: ["scale"]), .init(title: "Height", keys: ["target_height"])])]
    }
    override func pillTitle(_ op: OperationDescriptor) -> String { op.name == "upscale_video" ? "Video" : op.name == "upscale_image" ? "Image" : super.pillTitle(op) }
    override var idleLine: String { "Upscaling" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Upscale" }
    /// A video dropped while the picture operation is chosen switches to the video one (and back).
    override func setSubject(_ urls: [URL]) {
        if let first = urls.first, !running {
            let video = FileAccept(mimes: ["video/*"]).accepts(first)
            let wanted = video ? "upscale_video" : "upscale_image"
            if operation != wanted, operations.contains(where: { $0.name == wanted }) { select(wanted) }
        }
        super.setSubject(urls)
    }
    override var subjectAccepts: [FileAccept] {
        // Both kinds of file are taken when the model offers both operations: the drop picks the operation.
        let both = operations.map(\.name).contains("upscale_image") && operations.map(\.name).contains("upscale_video")
        return both ? [FileAccept(mimes: ["image/png", "image/jpeg"]), FileAccept(mimes: ["video/*"])] : super.subjectAccepts
    }
    override func check(_ outcome: Outcome) throws {
        guard outcome.file("image/") != nil || outcome.file("video/") != nil else {
            throw Undelivered(.init(message: "\(outcome.modelName) returned nothing to show.", action: nil))
        }
    }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        [Figure(title: "Size", value: outcome.flatMap(resultSize)?.title, reference: nil, help: "The result's size in pixels.", width: 120)]
    }
    func resultSize(_ o: Outcome) -> PixelSize? {
        o.file("image/").flatMap { f in f.width.flatMap { w in f.height.map { PixelSize(width: w, height: $0) } } ?? PixelSize.read(f.url) }
            ?? o.file("video/").flatMap { VideoInfo.cachedSize($0.url) }
    }
}

struct UpscaleView: View {
    @ObservedObject var session: UpscaleSession
    @Environment(\.palette) private var p
    var body: some View {
        CapabilityView(session: session) {
            OutcomePanel(
                session: session, what: "result", actions: { o in actions(o) },
                content: { o in
                    Group {
                        if let image = o.file("image/"), let original = o.inputs.first {
                            BeforeAfter(
                                before: original, after: image.url, afterSize: image.width.flatMap { w in image.height.map { PixelSize(width: w, height: $0) } },
                                afterTitle: "Upscaled"
                            )
                        } else if let video = o.file("video/") {
                            VideoResult(url: session.showsOriginal ? (o.inputs.first ?? video.url) : video.url)
                        }
                    }
                    .padding(.horizontal, 16).padding(.bottom, 16)
                })
        }
    }
    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        if o.file("video/") != nil, !o.inputs.isEmpty {
            Pill(title: "Upscaled", on: !session.showsOriginal, enabled: true, help: "The upscaled video") { session.showsOriginal = false }.controlID("show-result")
            Pill(title: "Original", on: session.showsOriginal, enabled: true, help: "The video as it was") { session.showsOriginal = true }.controlID("show-original")
        }
        if let input = o.inputs.first, let before = ToolFiles.isImageFile(input) ? PixelSize.read(input) : VideoInfo.cachedSize(input) {
            // Both sizes, the result's last: the header names the change, not only the original.
            let after = session.resultSize(o)
            Text(before.title + (after.map { " \u{2192} " + $0.title } ?? "")).font(.system(size: 11.5).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
        }
    }
}
