// SPDX-License-Identifier: MIT
// FLUX 3 Image Klein on the flux worker: a model folder whose `architecture` is `flux3-image-klein` loads Flux3Klein's
// pipeline instead of FLUX.2 Klein 4B's. Ops: generate (text to image) and edit (prompt plus 1–10 reference images).
import CryptoKit
import Flux2Klein
import Flux3Klein
import Foundation
import MLX

enum Flux3Serving {
    static let architecture = "flux3-image-klein"
    static let maxPixels = 4 * 1024 * 1024
}

/// What the runner needs from the FLUX 3 engine; the real one is `Flux3Pipeline`, tests use a fake.
protocol Flux3Engine: AnyObject {
    func image(_ request: Flux3Request, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int)
}

extension Flux3Pipeline: Flux3Engine {
    func image(_ request: Flux3Request, step: (Int, Int) throws -> Void) throws -> (pixels: [UInt8], width: Int, height: Int) {
        let out = try generate(request, step: step)
        return (out.pixels, out.width, out.height)
    }
}

/// The request contract of the add-on's ops/generate.input.json and ops/edit.input.json, rechecked at the worker.
struct Flux3ImageRequest: Equatable {
    var prompt: String, width: Int?, height: Int?, steps: Int, n: Int, seed: UInt32, images: [URL], outDir: URL

    static func parse(_ r: JSONObject, operation: String, support: URL, randomSeed: () -> UInt32 = { UInt32.random(in: 0...UInt32.max) }) throws
        -> Flux3ImageRequest
    {
        func invalid(_ message: String) -> WorkerFailure { ImageRequest.invalid(message) }
        guard ["generate", "edit"].contains(operation) else { throw WorkerFailure(code: "unsupported_op", message: "Unsupported operation.") }
        guard let prompt = r["prompt"]?.string, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, prompt.count <= 32768 else {
            throw invalid("prompt must contain 1 to 32768 characters.")
        }
        for key in ["guidance", "negative_prompt"] where r[key] != nil {
            throw invalid("FLUX 3 Image Klein is distilled and runs without guidance or a negative prompt.")
        }
        if r["resolution"] != nil { throw invalid("Use width and height (multiples of 16) for FLUX 3 Image Klein.") }
        func optional(_ key: String, _ range: ClosedRange<Int>) throws -> Int? {
            r[key] == nil ? nil : try ImageRequest.integer(r, key, default: 0, range)
        }
        var width = try optional("width", 64...4096), height = try optional("height", 64...4096)
        for side in [width, height].compactMap({ $0 }) where side % 16 != 0 { throw invalid("Width and height must be multiples of 16.") }
        if operation == "generate" { width = width ?? 1024; height = height ?? 1024 }
        if let width, let height, width * height > Flux3Serving.maxPixels {
            throw invalid("Width × height must be at most 4 megapixels (2048 × 2048); choose a smaller size.")
        }
        let steps = try ImageRequest.integer(r, "steps", default: Flux3Schedule.defaultSteps, 1...100)
        let n = try ImageRequest.integer(r, "n", default: 1, 1...4)
        let seed = try ImageRequest.integer(r, "seed", default: Int(randomSeed()), 0...Int(UInt32.max))
        var images: [URL] = []
        if let list = r["images"] {
            guard operation == "edit" else { throw invalid("Reference images belong to edit; generate takes a prompt only.") }
            guard let items = list.array, (1...10).contains(items.count) else { throw invalid("images must list 1 to 10 reference images.") }
            images = try items.map { item in
                guard let path = item["path"]?.string, path.hasPrefix("/") else { throw invalid("Each reference image needs an absolute path.") }
                guard FileManager.default.isReadableFile(atPath: path) else { throw invalid("Reference image \(URL(fileURLWithPath: path).lastPathComponent) is not readable.") }
                return URL(fileURLWithPath: path)
            }
        } else if operation == "edit" {
            throw invalid("edit needs 1 to 10 reference images.")
        }
        guard let out = r["out_dir"]?.string, out.hasPrefix("/") else { throw invalid("out_dir must be an absolute supervisor job directory.") }
        let dir = ImageRequest.resolved(URL(fileURLWithPath: out)), jobs = ImageRequest.resolved(support.appendingPathComponent("jobs"))
        guard dir.path.hasPrefix(jobs.path + "/") else { throw invalid("Image output must be inside an Arsenal job directory.") }
        return Flux3ImageRequest(prompt: prompt, width: width, height: height, steps: steps, n: n, seed: UInt32(seed), images: images, outDir: dir)
    }
}

extension Runner {
    func loadFlux3(_ root: URL, dtype: String, cancelled: @escaping () -> Bool) throws -> any Flux3Engine {
        guard dtype == "bf16" else {
            throw WorkerFailure(code: "unsupported_dtype", message: "FLUX 3 Image Klein runs in its release precision, BF16.")
        }
        if let missing = Flux3Weights.requiredFiles.first(where: { !FileManager.default.fileExists(atPath: root.appendingPathComponent($0).path) }) {
            throw WorkerFailure(code: "missing_weights", message: "The local snapshot is incomplete: \(missing); run Get again.")
        }
        do { return try flux3Loader(root, cancelled) } catch is CancellationError { throw KleinRequestCancelled() }
    }

    func generateFlux3(_ r: JSONObject, operation: String, engine: any Flux3Engine) throws {
        let id = r["id"] ?? .null, number = id.number ?? -1
        let request = try Flux3ImageRequest.parse(r, operation: operation, support: supportDirectory)
        // References are decoded, prepared and the output size resolved before any GPU work: a reference too small to
        // condition on, or a size past 4 megapixels, is the caller's error.
        var planned = Flux3Request(prompt: request.prompt)
        planned.width = request.width; planned.height = request.height; planned.steps = request.steps
        do {
            planned.references = try request.images.map { try RGBImage.read($0) }
            planned = try Flux3Pipeline.plan(planned)
        } catch let error as Flux3Error { throw ImageRequest.invalid(error.message) }
        try FileManager.default.createDirectory(at: request.outDir, withIntermediateDirectories: true)
        var files: [JSONValue] = [], seeds: [JSONValue] = [], written: [URL] = []
        let start = Date(), total = request.steps * request.n
        do {
            for i in 0..<request.n {
                if jobs.cancelled(number) { throw KleinRequestCancelled() }
                let seed = UInt32(truncatingIfNeeded: UInt64(request.seed) + UInt64(i))
                var job = planned
                job.seed = UInt64(seed)
                let out = try engine.image(job) { step, steps in
                    let done = i * steps + step
                    try? self.wire.send([
                        "id": id, "event": "progress", "phase": "step", "step": .number(Double(done)), "total": .number(Double(total)),
                        "fraction": .number(Double(done) / Double(total))
                    ])
                    if self.jobs.cancelled(number) { throw KleinRequestCancelled() }
                }
                if jobs.cancelled(number) { throw KleinRequestCancelled() }
                let file = request.outDir.appendingPathComponent("image-\(i + 1)-\(String(format: "%08x", seed)).png")
                try PNG.write(out.pixels, width: out.width, height: out.height, to: file)
                written.append(file)
                let data = try Data(contentsOf: file)
                files.append([
                    "id": .string("image-\(i + 1)"), "path": .string(file.path), "mime": "image/png", "width": .number(Double(out.width)),
                    "height": .number(Double(out.height)), "bytes": .number(Double(data.count)),
                    "sha256": .string(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
                ])
                seeds.append(.number(Double(seed)))
            }
        } catch {
            for file in written { try? FileManager.default.removeItem(at: file) }
            throw error
        }
        try wire.reply(id, ["files": .array(files), "seed": .number(Double(request.seed)), "seeds": .array(seeds), "seconds": .number(Date().timeIntervalSince(start))])
    }
}
