// SPDX-License-Identifier: MIT
// Lab checks without the model: tokens, rendered prompts and processed pictures, for comparison with the vendor stack.
//   d1-check tokens  <weights>  < JSON lines {"text": ...}                 -> {"ids": [...]}
//   d1-check prompts <weights>  < JSON lines {"state": ..., "questions": {...}, "pictures": n}
//                                -> {"prefix", "suffixes", "prefix_ids", "suffix_ids", "render_ids"}
//   d1-check pixels  <weights> <picture> <out.bin>  -> pieces (uint8 RGB, feature order) into out.bin; layout on stdout
import D1Engine
import Foundation
import MLX

func fail(_ message: String) -> Never {
    FileHandle.standardError.write(Data((message + "\n").utf8))
    exit(1)
}

func json(_ v: PyJSON) -> String { v.compact() }
func ids(_ a: [Int]) -> PyJSON { .array(a.map { .scalar(String($0)) }) }

func question(_ q: PyJSON) -> Question {
    let kind = QuestionKind(rawValue: q["type"]?.string ?? "") ?? .choice
    let instructions = q["instructions"] ?? .string("")
    switch kind {
    case .choice: return Question(kind: .choice, instructions: instructions, options: q["criteria"]?.fields ?? [])
    case .score: return Question(kind: .score, instructions: instructions, levels: q["criteria"]?.array ?? [])
    case .noul: return Question(kind: .noul, instructions: instructions, noulCriteria: q["criteria"].flatMap { $0.isNull ? nil : $0 })
    }
}

let args = CommandLine.arguments
guard args.count >= 3 else { fail("usage: d1-check tokens|prompts|pixels <weights> ...") }
let weights = URL(fileURLWithPath: args[2])
let tokenizer: LFMTokenizer
do { tokenizer = try LFMTokenizer(url: weights.appendingPathComponent("tokenizer.json")) } catch { fail("\(error)") }
let bos = "<|startoftext|>"

switch args[1] {
case "tokens":
    while let line = readLine(strippingNewline: true) {
        guard let v = try? PyJSONParser.parse(Data(line.utf8)), let text = v["text"]?.string else { fail("bad line") }
        print(json(.object([("ids", ids(tokenizer.encode(text)))])))
    }
case "prompts":
    let verbalizer = Verbalizer(tokenizer: tokenizer)
    while let line = readLine(strippingNewline: true) {
        guard let v = try? PyJSONParser.parse(Data(line.utf8)), let qs = v["questions"]?.fields else { fail("bad line") }
        let state: PyJSON? = v["state"].flatMap { $0.isNull ? nil : $0 }
        let markup = PictureProcessor.markup(v["pictures"]?.int ?? 0)
        let prefix = Prompt.prefix(state, bos: bos, images: markup)
        do {
            let suffixes = try qs.map { try verbalizer.suffix(question($0.1)) }
            print(
                json(
                    .object([
                        ("prefix", .string(prefix)), ("suffixes", .array(suffixes.map { .string($0) })), ("prefix_ids", ids(tokenizer.encode(prefix))),
                        ("suffix_ids", .array(suffixes.map { ids(tokenizer.encode($0)) })), ("render_ids", ids(tokenizer.encode(prefix + suffixes[0])))
                    ])))
        } catch {
            print(json(.object([("error", .string("\(error)"))])))
        }
    }
case "pixels":
    guard args.count == 5 else { fail("usage: d1-check pixels <weights> <picture> <out.bin>") }
    do {
        let processor = try PictureProcessor(config: weights.appendingPathComponent("processor_config.json"))
        let picture = processor.cap(try Picture.load(URL(fileURLWithPath: args[3])))
        let layout = processor.layout(picture)
        let pieces = processor.pieces(picture, layout)
        var data = Data()
        for p in pieces { data.append(contentsOf: p.pixels) }
        try data.write(to: URL(fileURLWithPath: args[4]))
        print(
            json(
                .object([
                    ("capped", .array([.scalar(String(picture.height)), .scalar(String(picture.width))])), ("rows", .scalar(String(layout.rows))),
                    ("cols", .scalar(String(layout.cols))), ("size", .array([.scalar(String(layout.size.height)), .scalar(String(layout.size.width))])),
                    ("pieces", .array(pieces.map { .array([.scalar(String($0.height)), .scalar(String($0.width))]) })),
                    ("expanded", .string(try processor.expand("<image>", [layout])))
                ])))
    } catch {
        fail("\(error)")
    }
case "features":
    // d1-check features <weights> <picture> <out.bin>: projected picture features, float32 (rows x 2048).
    guard args.count == 5 else { fail("usage: d1-check features <weights> <picture> <out.bin>") }
    do {
        let processor = try PictureProcessor(config: weights.appendingPathComponent("processor_config.json"))
        let store = try SafetensorsFile(url: weights.appendingPathComponent("model.safetensors"))
        let tower = try VisionTower(store: store, processor: processor)
        let picture = processor.cap(try Picture.load(URL(fileURLWithPath: args[3])))
        let features = tower.features([(picture, processor.layout(picture))]).asType(.float32)
        let values = features.asArray(Float.self)
        try values.withUnsafeBufferPointer { try Data(buffer: $0).write(to: URL(fileURLWithPath: args[4])) }
        print(features.shape)
    } catch {
        fail("\(error)")
    }
case "forward":
    // d1-check forward <weights> <reps> < one JSON line {"trunk": [...], "branches": [[...], ...]}: ms per forward.
    guard args.count == 4, let reps = Int(args[3]), let line = readLine(), let v = try? PyJSONParser.parse(Data(line.utf8)) else { fail("usage") }
    do {
        let store = try SafetensorsFile(url: weights.appendingPathComponent("model.safetensors"))
        let model = try D1Model(weights: weights, store: store)
        let trunk = v["trunk"]!.array!.map { $0.int! }
        let branches = v["branches"]!.array!.map { $0.array!.map { $0.int! } }
        var times: [Double] = []
        for _ in 0..<reps {
            let t = ProcessInfo.processInfo.systemUptime
            var layout = Layout()
            layout.add(trunk: trunk, branches: branches)
            let logits = model.logits(model.hidden(layout), [1, 2, 3])
            eval(logits)
            times.append((ProcessInfo.processInfo.systemUptime - t) * 1000)
        }
        print(times.map { String(format: "%.1f", $0) }.joined(separator: " "))
    } catch {
        fail("\(error)")
    }
default:
    fail("unknown command \(args[1])")
}
