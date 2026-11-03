// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// Find Objects: one window for every model that declares `detect` or `ground_screen`. A picture or a screenshot
// (pasted with ⌘V); labels to find, a phrase or an instruction, and the threshold, as the model takes them; every
// detection as a box (or a point) in its label's colour with its score, over the picture, and listed beside it. Screen
// grounding answers in screenshot pixels: they are converted with the reply's own image size, never forwarded as if
// they were normalized.

@MainActor final class FindSession: CapabilitySession {
    /// The detection picked in the list (drawn heavier).
    @Published var chosenDetection: Int?

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .find, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func mainKeys(_ op: OperationDescriptor) -> [String: String] {
        op.name == "ground_screen" ? ["instruction": "What to find on the screen"] : [:]
    }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["labels": "Find", "phrase": "Phrase", "instruction": "Instruction", "threshold": "Confidence"] }
    override var resultNoun: String { "search" }
    override func helps(_ op: OperationDescriptor) -> [String: String] {
        ["threshold": "How sure the model must be to show a find: lower shows more (and more mistakes), higher shows only sure ones."]
    }
    override func groups(_ op: OperationDescriptor) -> [OptionGroup] {
        op.name == "detect"
            ? [
                OptionGroup(
                    id: "what", options: [.init(title: "Labels", keys: ["labels"]), .init(title: "Phrase", keys: ["phrase"]), .init(title: "Instruction", keys: ["instruction"])])
            ]
            : []
    }
    override func pillTitle(_ op: OperationDescriptor) -> String { op.name == "ground_screen" ? "Screen" : op.name == "detect" ? "Objects" : super.pillTitle(op) }
    override func historyPicture(_ o: Outcome) -> URL? { o.inputs.first }
    override var idleLine: String { "Looking" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Find" }
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        try super.deliver(result, op: op, model: model, input: input)
        chosenDetection = nil
    }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        [Figure(title: "Found", value: outcome.map { "\(detections($0).count)" }, reference: nil, help: "Detections at or above the threshold.", width: 64)]
    }
    func detections(_ o: Outcome) -> [Detection] {
        Detection.list(o.reply, imageSize: o.inputs.first.flatMap(PixelSize.read))
    }
}

struct FindView: View {
    @ObservedObject var session: FindSession
    @Environment(\.palette) private var p

    var body: some View {
        CapabilityView(session: session) {
            OutcomePanel(
                session: session, what: "detections", copyText: { o in OperationForm.pretty(.object(o.reply)) ?? "" },
                content: { o in
                    let detections = session.detections(o)
                    let pixels = o.inputs.first.flatMap(PixelSize.read)
                    // The list beside the picture where the panel is wide; under it where beside would shrink the
                    // picture until its labels collide.
                    GeometryReader { g in
                        if g.size.width >= Self.sideBySide {
                            HStack(alignment: .top, spacing: 12) {
                                picture(o, detections)
                                list(detections, pixels: pixels).frame(width: 180)
                            }
                        } else {
                            VStack(alignment: .leading, spacing: 10) {
                                picture(o, detections)
                                list(detections, pixels: pixels).frame(height: min(120, CGFloat(max(1, detections.count)) * 28 + 4))
                            }
                        }
                    }
                    .padding(.horizontal, 16).padding(.bottom, 16)
                })
        }
    }

    /// The panel width from which the list sits beside the picture (at the window's default size, beside Copy's gutter).
    static let sideBySide: CGFloat = 580

    @ViewBuilder private func picture(_ o: Outcome, _ detections: [Detection]) -> some View {
        if let picture = o.inputs.first {
            let pixels = PixelSize.read(picture).map { CGSize(width: $0.width, height: $0.height) } ?? CGSize(width: 1, height: 1)
            GeometryReader { g in
                let box = Letterbox(container: g.size, pixels: pixels, top: true)
                ZStack(alignment: .topLeading) {
                    Thumbnail(url: picture, fit: true).frame(width: box.rect.width, height: box.rect.height).clipShape(RoundedRectangle(cornerRadius: 6))
                        .offset(x: box.rect.minX, y: box.rect.minY)
                    DetectionOverlay(detections: detections, box: box, chosen: session.chosenDetection)
                }
            }
        }
    }

    private func list(_ detections: [Detection], pixels: PixelSize?) -> some View {
        let labels = detections.map(\.label).reduce(into: [String]()) { if !$0.contains($1) { $0.append($1) } }
        return ScrollView(.vertical) {
            VStack(alignment: .leading, spacing: 2) {
                if detections.isEmpty {
                    Text("Nothing found").font(.system(size: 12.5)).foregroundStyle(p.t2.swiftUI).padding(.top, 4)
                }
                ForEach(detections) { d in
                    Button {
                        session.chosenDetection = session.chosenDetection == d.id ? nil : d.id
                    } label: {
                        HStack(spacing: 7) {
                            Circle().fill(SpatialColours.colour(d.label, in: labels)).frame(width: 8, height: 8)
                            Text(d.label.isEmpty ? "Object" : d.label).font(.system(size: 12.5)).lineLimit(1)
                            Spacer(minLength: 4)
                            Text("\(Int((d.score * 100).rounded()))%").font(.system(size: 12).monospacedDigit()).foregroundStyle(p.t2.swiftUI)
                        }
                        .padding(.horizontal, 8).frame(height: 26)
                        .background((session.chosenDetection == d.id ? p.accent.withAlphaComponent(0.16) : .clear).swiftUI, in: RoundedRectangle(cornerRadius: 6))
                        .contentShape(Rectangle())
                    }
                    .buttonStyle(.plain)
                    .windowHelp(Self.place(d, pixels: pixels))
                    .controlID("detection-\(d.id)")
                }
            }
        }
    }

    /// Where a detection is, in the picture's pixels ("x 120–480, y 40–310" or "x 812, y 44").
    static func place(_ d: Detection, pixels: PixelSize?) -> String {
        let w = Double(pixels?.width ?? 1), h = Double(pixels?.height ?? 1)
        func px(_ v: Double) -> Int { Int(v.rounded()) }
        if let b = d.box { return "x \(px(b.minX * w))\u{2013}\(px(b.maxX * w)), y \(px(b.minY * h))\u{2013}\(px(b.maxY * h))" }
        if let pt = d.point { return "x \(px(pt.x * w)), y \(px(pt.y * h))" }
        return d.label
    }
}
