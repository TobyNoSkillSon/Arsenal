// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore

// Make an Image, as the tool kit composes it: one window for every model that declares the image tool, the chosen
// model's operations that make images with their options from their schemas (prompt, size, steps, seed, how many, reference
// images: whatever the operation declares), each result saved to ~/Pictures/Arsenal as it arrives. The model's settings
// start the options. Step progress and memory come from the activity stream (C3) while a run goes; the time is the
// window's own clock, compared with the reference measured for the same size and steps.

struct ImageResult: Identifiable, Equatable {
    let id = UUID()
    var url: URL
    var prompt: String
    var width: Int
    var height: Int
    var steps: Int?
    var seed: Int?
    var seconds: Double
}

@MainActor final class ImageSession: ToolSession {
    @Published private(set) var results: [ImageResult] = []
    @Published var shown: UUID?
    /// The reference figures measured for the chosen model as it runs now (its dtype and mode).
    @Published private(set) var reference: ReferenceState = .loading
    private var referencing: Task<Void, Never>?

    static var picturesDirectory: URL { ToolFiles.picturesDirectory }

    override init(
        models: [ModelChoice], selected: String, client: @escaping () -> ArsenalClient?, inbox: URL, outputDirectory: URL = ImageSession.picturesDirectory,
        scratch: URL? = nil
    ) {
        super.init(models: models, selected: selected, client: client, inbox: inbox, outputDirectory: outputDirectory, scratch: scratch)
    }
    /// One model's window (tests and renders).
    convenience init(model: ToolModel, client: @escaping () -> ArsenalClient?, outputDirectory: URL = ImageSession.picturesDirectory) {
        self.init(
            models: [ModelChoice(model: model)], selected: model.id, client: client,
            inbox: FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-inbox", isDirectory: true), outputDirectory: outputDirectory)
    }

    override func offers(_ op: OperationDescriptor) -> Bool { op.output == "image" }
    override var preferredOperations: [String] { ["generate"] }
    override var resultNoun: String { "image" }
    override var taskTitle: String { "Make an Image" }
    override var runVerb: String { descriptor.map { $0.name == "generate" ? "Make Image" : OperationPills.title($0.title) } ?? "Make Image" }
    override func noOperations(_ name: String) -> String { "\(name) declares no operation that makes an image." }

    var current: ImageResult? { results.first { $0.id == shown } ?? results.last }
    override var shownSeed: Int? { current?.seed }
    override var shownFile: URL? { current?.url }

    /// A Reload with another dtype or mode changes the figures' reference.
    override func update(models: [ModelChoice]) {
        let before = model?.model
        super.update(models: models)
        if let now = model?.model, let before, now.id == before.id, now.dtype != before.dtype || now.mode != before.mode { readReference() }
    }

    override func didPrepare() { readReference() }

    private func readReference() {
        referencing?.cancel()
        guard let model = model?.model else { return }
        reference = .loading
        referencing = Task {
            guard let client = client() else { reference = .failed(ChatSession.noAPI); return }
            let found: ReferenceState
            do {
                found = try await client.reference(model.id, dtype: model.dtype, mode: model.mode).map { .found($0) } ?? .none
            } catch let e as ToolError where e.status == 404 {
                found = .none
            } catch {
                found = .failed(error.localizedDescription)
            }
            guard !Task.isCancelled, selected == model.id else { return }
            reference = found
        }
    }

    override func close() {
        referencing?.cancel(); referencing = nil
        super.close()
    }

    /// Each image saved as `<date> <time> <model> <seed>.png`.
    override func deliver(_ result: OperationResult, op: OperationDescriptor, model: ModelChoice, input: JSONObject) throws {
        let images = result.files.filter(\.isImage)
        guard !images.isEmpty else { throw Undelivered(.init(message: "The model answered without an image.", action: nil)) }
        let seed = input["seed"]?.number.flatMap(\.roundedInt)
        let asked = Self.size(input)
        let prompt = form?.main.flatMap { input[$0.key]?.string } ?? ""
        for file in images {
            let url = try ToolFiles.save(file.url, into: outputDirectory, model: model.id, tag: seed.map(String.init) ?? "", what: "image")
            let size = PixelSize.read(url)
            results.append(
                ImageResult(
                    url: url, prompt: prompt, width: asked?.width ?? size?.width ?? file.width ?? 0, height: asked?.height ?? size?.height ?? file.height ?? 0,
                    steps: input["steps"]?.number?.roundedInt, seed: seed, seconds: result.seconds))
        }
        // The first of a batch is shown; the rest are in the strip.
        shown = results[results.count - images.count].id
    }

    /// The size a request asked for: width and height, else its resolution.
    static func size(_ input: JSONObject) -> ImageSize? {
        if let w = input["width"]?.number?.roundedInt, let h = input["height"]?.number?.roundedInt { return ImageSize(width: w, height: h) }
        return input["resolution"]?.string.flatMap(ImageSize.init(resolution:))
    }

    /// The size and steps the options hold now.
    private var chosen: (size: ImageSize?, steps: Int?) {
        let size = form?.fields.first {
            if case .size = $0.control { return true }; return false
        }.flatMap { ImageSize(resolution: value($0.key).text) }
        return (size, Int(value("steps").text))
    }

    // MARK: Figures

    /// Time is this window's own clock from Generate to the result; steps and memory come from the activity stream (C3).
    override func figures() -> [Figure] {
        let r = run, ref = comparable
        let done = r?.done == true
        let seconds = done ? r?.seconds(now: Date()) : nil
        let asked = r?.input["steps"]?.number?.roundedInt
        let total = r?.total ?? (done ? asked : nil)
        let step = done ? total : r?.step
        let perStep = seconds.flatMap { s in total.flatMap { $0 > 0 ? s / Double($0) : nil } }
        let refSteps = ref?.workload.flatMap { w -> Int? in
            if case .image(_, _, let n) = w { return n }; return nil
        }
        let refSeconds = ref?.value(["seconds_per_image", "seconds"])
        return [
            Figure(
                title: "Time", value: FigureText.seconds(seconds), reference: FigureText.seconds(refSeconds),
                help: "From Generate to the saved image, as this window measured it.", width: 90),
            Figure(
                title: "Per step", value: FigureText.seconds(perStep, decimals: 3),
                reference: FigureText.seconds(refSeconds.flatMap { s in refSteps.map { s / Double($0) } }, decimals: 3),
                help: "Time divided by the denoising steps.", width: 96),
            Figure(
                title: "Steps", value: step.map { s in total.map { "\(s) of \($0)" } ?? "\(s)" } ?? (running ? asked.map { "0 of \($0)" } : nil),
                reference: refSteps.map { "\($0) of \($0)" },
                help: "Denoising steps done.", width: 84),
            memoryFigure(reference: FigureText.gigabytes(mb: ref?.value(["peak_mb"]) ?? ref?.value(["peak_ram", "peak_ram_gb"]).map { $0 * 1000 }))
        ]
    }

    /// The chosen model has reference figures (for some size and steps).
    var hasReference: Bool {
        if case .found = reference { return true }; return false
    }
    private var comparable: ToolReference? {
        if case .compared(let r) = referenceLine { return r } else { return nil }
    }
    /// The image suite makes 512 × 512 images in 4 steps; other sizes and step counts have nothing to compare with.
    var referenceLine: ReferenceLine? {
        switch reference {
        case .loading: return nil
        case .failed: return .unavailable("Reference unavailable")
        // Nothing measured: no row at all, rather than a line saying so.
        case .none: return nil
        case .found(let r):
            guard case .image(let w, let h, let n)? = r.workload else { return .unavailable("No reference for this size") }
            let size = run.flatMap { Self.size($0.input) } ?? chosen.size
            let count = run?.input["steps"]?.number?.roundedInt ?? chosen.steps
            guard let size, let count else { return .unavailable("No reference for this size") }
            return size.width == w && size.height == h && count == n ? .compared(r) : .unavailable("No reference for \(size.width) \u{00d7} \(size.height) in \(count) steps")
        }
    }

    /// Render harness and tests: a state without the API.
    func preview(
        operations: [OperationDescriptor], selected op: String? = nil, values: [String: FieldValue] = [:], results: [ImageResult] = [], run: ToolRun? = nil,
        reference: ReferenceState = .none, problem: OperationProblem? = nil, settings: [String: JSONValue] = [:]
    ) {
        previewOptions(operations: operations, selected: op, values: values, settings: settings)
        self.results = results; self.run = run; self.reference = reference; self.problem = problem; shown = results.last?.id
    }
}
