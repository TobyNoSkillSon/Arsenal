// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SceneKit
import SwiftUI

// Make 3D: one window for every model that declares `image_to_3d`. A picture; resolution, texture size, simplification
// and seed, as the model takes them; the model in a view the pointer turns and zooms, Open (the Mac's own viewer for
// the file) and Show in Finder. A job: its progress shows, Stop cancels it. Models are saved to ~/Documents/Arsenal.

@MainActor final class Make3DSession: CapabilitySession {
    init(models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL? = nil, scratch: URL? = nil) {
        super.init(tool: .make3d, models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    override func titles(_ op: OperationDescriptor) -> [String: String] { ["resolution": "Detail", "texture_size": "Texture size", "simplify": "Simplify"] }
    /// Detail and texture size are chosen from the sizes that make sense, not typed: the grid's steps of 512, the
    /// texture's powers of two, within what the model declares.
    override func makeForm(_ op: OperationDescriptor) -> ToolForm {
        var form = super.makeForm(op)
        form.offer("resolution", [512, 1024, 1536, 2048]) { String($0) }
        form.offer("texture_size", [512, 1024, 2048, 4096]) { "\($0) px" }
        return form
    }
    override var resultNoun: String { "model" }
    override var idleLine: String { "Making the model" }
    override func runTitle(_ op: OperationDescriptor?) -> String { "Make 3D" }
    /// A 3D file the window can show: it is read before the result is kept, so a broken one is a stated failure.
    override func check(_ outcome: Outcome) throws {
        guard let file = Self.model(outcome) else { throw Undelivered(.init(message: "\(outcome.modelName) returned no 3D model.", action: nil)) }
        if file.url.pathExtension.lowercased() == "glb" {
            do { _ = try GLB.scene(file.url) } catch {
                throw Undelivered(
                    .init(
                        message: "\(outcome.modelName) returned a 3D file Arsenal cannot show: \(error.localizedDescription) It is saved as \(file.url.lastPathComponent).",
                        action: nil))
            }
        }
    }
    /// Triangles in a GLB, counted once per file (the figures redraw while a run goes).
    private static let counts = MediaCache<URL, Int>(limit: MediaCaches.mb) { _ in 64 }
    static func triangles(_ url: URL) -> Int? {
        if let n = counts[url] { return n }
        guard url.pathExtension.lowercased() == "glb", let scene = try? GLB.scene(url) else { return nil }
        var n = 0
        scene.rootNode.enumerateHierarchy { node, _ in n += node.geometry?.elements.reduce(0) { $0 + $1.primitiveCount } ?? 0 }
        counts.set(n, for: url)
        return n
    }
    /// Gaussians in a splat PLY, read from its header (`element vertex N`).
    private static let splatCounts = MediaCache<URL, Int>(limit: MediaCaches.mb) { _ in 64 }
    static func gaussians(_ url: URL) -> Int? {
        if let n = splatCounts[url] { return n }
        guard url.pathExtension.lowercased() == "ply", let handle = try? FileHandle(forReadingFrom: url) else { return nil }
        defer { try? handle.close() }
        guard let head = try? handle.read(upToCount: 4096), let end = head.range(of: Data("end_header".utf8)) else { return nil }
        let header = String(decoding: head[..<end.lowerBound], as: UTF8.self)
        guard header.contains("property float f_dc_0") else { return nil }
        guard let n = header.split(separator: "\n").first(where: { $0.hasPrefix("element vertex ") }).flatMap({ Int($0.dropFirst(15)) }) else { return nil }
        splatCounts.set(n, for: url)
        return n
    }
    /// A splat has no triangles: the window shows the model's own render of it.
    static func splatPreview(_ o: Outcome) -> ResultFile? {
        guard let file = model(o), gaussians(file.url) != nil else { return nil }
        return o.files.first { $0.isImage }
    }
    static func model(_ o: Outcome) -> ResultFile? { o.file("model/") ?? o.files.first { ["glb", "gltf", "usdz", "obj", "ply"].contains($0.url.pathExtension.lowercased()) } }
    override var shownFile: URL? { current.flatMap(Self.model)?.url ?? super.shownFile }
    override func extraFigures(_ outcome: Outcome?) -> [Figure] {
        if let file = outcome.flatMap(Self.model), let count = Self.gaussians(file.url) {
            return [Figure(title: "Gaussians", value: count.grouped, reference: nil, help: "Gaussians in the splat.", width: 96)]
        }
        let triangles = outcome.flatMap(Self.model).flatMap { Self.triangles($0.url) }
        return [Figure(title: "Triangles", value: triangles.map { $0.grouped }, reference: nil, help: "Triangles in the model.", width: 96)]
    }
}

struct Make3DView: View {
    @ObservedObject var session: Make3DSession
    var body: some View {
        CapabilityView(session: session) {
            OutcomePanel(
                session: session, what: "model", actions: { _ in EmptyView() }, trailing: { o in AnyView(actions(o)) },
                content: { o in
                    Group {
                        if let file = Make3DSession.model(o), file.url.pathExtension.lowercased() == "glb" {
                            ModelPreview(url: file.url).accessibilityLabel("3D model").controlID("model-preview")
                                .onDrag { NSItemProvider(contentsOf: file.url) ?? NSItemProvider() }
                        } else if let preview = Make3DSession.splatPreview(o) {
                            FittedImage(url: preview.url, size: PixelSize.read(preview.url), label: "3D splat", backdrop: false).controlID("splat-preview")
                                .onDrag { NSItemProvider(contentsOf: Make3DSession.model(o)?.url) ?? NSItemProvider() }
                        } else if let picture = o.inputs.first {
                            FittedImage(url: picture, size: PixelSize.read(picture), label: picture.lastPathComponent)
                        }
                    }
                    .padding(.horizontal, 16).padding(.bottom, 16)
                })
        }
    }
    @ViewBuilder private func actions(_ o: Outcome) -> some View {
        if let file = Make3DSession.model(o) {
            ResultIcon(symbol: "arrow.up.forward.app", help: "Open \(file.url.lastPathComponent)", id: "open-model") { NSWorkspace.shared.open(file.url) }
        }
    }
}
