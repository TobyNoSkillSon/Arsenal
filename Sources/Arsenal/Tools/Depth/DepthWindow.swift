// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI

// Depth: one window for every model that declares `estimate_depth`. A picture (or several views of one scene, when
// the model takes them); relative or metric depth, points and camera, as the model takes them; the depth map in
// colour (near warm, far cool) with its units, the picture beside it under a handle, and any other file the model
// returned (a point cloud) to reveal. Units come from the reply (`depth_unit`): relative depth says near and far,
// metric depth says metres.

@MainActor final class DepthSession: CapabilitySession {
    @Published var colour = true
    @Published var compare = false
    /// The view on show when the model returned several.
    @Published var viewIndex = 0

    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .depth, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override var resultNoun: String { "depth map" }
    override var idleLine: String { "Estimating depth" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Make Depth Map" }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["points": "Point cloud", "intrinsics": "Camera parameters"] }
    /// "Camera parameters" says what comes back, never camera input or access: the hover names the values.
    override func helps(_ op: OperationDescriptor) -> [String: String] {
        ["intrinsics": "Also returns the camera\u{2019}s focal length and image centre, in pixels."]
    }
    override func check(_ outcome: Outcome) throws {
        guard outcome.files.contains(where: \.isImage) else { throw Undelivered(.init(message: "\(outcome.modelName) returned no depth map Arsenal can show.", action: nil)) }
    }
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        try super.deliver(result, op: op, model: model, input: input)
        viewIndex = 0
    }

    /// Depth maps, one per view.
    func maps(_ o: Outcome) -> [ResultFile] { o.files.filter(\.isImage) }
    /// Files that are not pictures (point clouds, raw depth).
    func others(_ o: Outcome) -> [ResultFile] { o.files.filter { !$0.isImage } }
    /// Metric depth is in metres; relative depth only orders near and far.
    static func metric(_ o: Outcome) -> Bool { o.reply["depth_unit"]?.string == "meters" }
    /// The focal length in pixels, when the reply has the camera.
    static func focal(_ o: Outcome) -> Double? {
        o.reply["intrinsics"]?.array?.first?.array?.first?.number ?? o.reply["views"]?.array?.first?["intrinsics"]?.array?.first?.array?.first?.number
    }
    /// "0.4 m" … "12.0 m" from the map's own range fields when the reply gives them; else near and far.
    static func legend(_ o: Outcome, map: ResultFile) -> (String, String) {
        guard metric(o) else { return ("Near", "Far") }
        let extras = Dictionary(map.extras.map { ($0.key, $0.value) }, uniquingKeysWith: { a, _ in a })
        let lo = extras["depth_min"].flatMap(Double.init) ?? o.reply["depth_min"]?.number
        let hi = extras["depth_max"].flatMap(Double.init) ?? o.reply["depth_max"]?.number
        return (lo.map { String(format: "%.1f m", $0) } ?? "Near", hi.map { String(format: "%.1f m", $0) } ?? "Far")
    }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        let focal = outcome.flatMap(Self.focal)
        return focal.map {
            [Figure(title: "Focal length", value: String(format: "%.0f px", $0), reference: nil, help: "The camera's focal length the model estimated.", width: 96)]
        } ?? []
    }
}

struct DepthView: View {
    @ObservedObject var session: DepthSession
    @Environment(\.palette) private var p

    var body: some View {
        CapabilityView(session: session) {
            OutcomePanel(
                session: session, what: "depth map", actions: { o in actions(o) },
                content: { o in
                    let maps = session.maps(o)
                    if let map = maps.indices.contains(session.viewIndex) ? maps[session.viewIndex] : maps.first {
                        VStack(spacing: 10) {
                            picture(o, map: map)
                            let (near, far) = DepthSession.legend(o, map: map)
                            DepthLegend(near: near, far: far).frame(maxWidth: 360)
                            if !session.others(o).isEmpty { others(session.others(o)) }
                        }
                        .padding(.horizontal, 16).padding(.bottom, 14)
                    }
                })
        }
    }

    @ViewBuilder private func picture(_ o: Outcome, map: ResultFile) -> some View {
        let nearIsHigh = !DepthSession.metric(o)
        if let rendered = DepthImage.render(map.url, colour: session.colour, nearIsHigh: nearIsHigh) {
            let shown = session.scratch.appendingPathComponent("depth-\(map.url.deletingPathExtension().lastPathComponent)-\(session.colour ? "colour" : "grey").png")
            let written = Self.write(rendered.image, to: shown)
            let source = o.inputs.indices.contains(session.viewIndex) ? o.inputs[session.viewIndex] : o.inputs.first
            if session.compare, let source, let written {
                BeforeAfter(before: source, after: written, afterSize: PixelSize.read(source), beforeTitle: "Image", afterTitle: "Depth")
            } else {
                Image(nsImage: rendered.image).resizable().scaledToFit().clipShape(RoundedRectangle(cornerRadius: 6))
                    .frame(maxWidth: .infinity, maxHeight: .infinity)
                    .onDrag { NSItemProvider(contentsOf: map.url) ?? NSItemProvider() }
                    .accessibilityLabel("Depth map")
            }
        }
    }

    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        Pill(title: "Colour", on: session.colour, enabled: !session.running, help: "Near warm, far cool") { session.colour = true }.controlID("depth-colour")
        Pill(title: "Grey", on: !session.colour, enabled: !session.running, help: "The map as the model wrote it") { session.colour = false }.controlID("depth-grey")
        if !o.inputs.isEmpty {
            BandButton(symbol: "square.split.2x1", help: session.compare ? "Show the depth map alone" : "Compare with the picture", enabled: !session.running) {
                session.compare.toggle()
            }.controlID("depth-compare")
        }
        let maps = session.maps(o)
        if maps.count > 1 {
            Spacer(minLength: 6)
            ForEach(maps.indices, id: \.self) { i in
                Pill(title: "\(i + 1)", on: session.viewIndex == i, enabled: !session.running, help: "View \(i + 1)") { session.viewIndex = i }.controlID("view-\(i)")
            }
        }
    }

    private func others(_ files: [ResultFile]) -> some View {
        HStack(spacing: 8) {
            ForEach(files) { f in
                Button {
                    ToolFiles.reveal(f.url, folder: f.url.deletingLastPathComponent())
                } label: {
                    Label(f.label, systemImage: "cube.transparent").font(.system(size: 12, weight: .medium)).padding(.horizontal, 8).frame(height: 24)
                        .background(p.field.swiftUI, in: RoundedRectangle(cornerRadius: 6)).contentShape(Rectangle())
                }.buttonStyle(.plain).windowHelp("Show \(f.url.lastPathComponent) in Finder").controlID("file-\(f.id)")
            }
            Spacer(minLength: 0)
        }
    }

    /// The rendered map as a file (the compare view reads files).
    @MainActor static func write(_ image: NSImage, to url: URL) -> URL? {
        if FileManager.default.fileExists(atPath: url.path) { return url }
        guard let tiff = image.tiffRepresentation, let png = NSBitmapImageRep(data: tiff)?.representation(using: .png, properties: [:]) else { return nil }
        try? FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        return (try? png.write(to: url)).map { url }
    }
}
