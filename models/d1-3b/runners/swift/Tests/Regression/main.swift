// SPDX-License-Identifier: MIT
// Pure CPU regression checks; compiled with just the tokenizer, prompt, JSON and picture sources (no MLX).
import Darwin
import Foundation

func check(_ condition: @autoclosure () -> Bool, _ message: String) {
    if !condition() { fatalError(message) }
}
let fixtures = URL(fileURLWithPath: CommandLine.arguments[1])
let scratch = URL(fileURLWithPath: CommandLine.arguments[2])
try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
let chars = LFMTokenizer.byteLevelChars()
var vocab = Dictionary(uniqueKeysWithValues: chars.enumerated().map { (String($0.element), $0.offset) })
func mapped(_ s: String) -> String { String(String.UnicodeScalarView(s.utf8.map { chars[Int($0)] })) }
vocab[mapped("Å")] = 256
vocab[mapped("Å")] = 257
// Equal-rank overlapping merges and recursively growing merges, compared to the old leftmost scan below.
let mergeWords = [("a", "a"), ("aa", "a"), ("a", "aa"), ("aaa", "aaa"), ("b", "a"), ("ba", "aaa")]
for (a, b) in mergeWords where vocab[a + b] == nil { vocab[a + b] = vocab.count }
let raw: [String: Any] = [
    "model": [
        "type": "BPE", "byte_fallback": false, "ignore_merges": true,
        "vocab": vocab, "merges": mergeWords.map { [$0.0, $0.1] }
    ], "normalizer": NSNull(),
    "pre_tokenizer": [
        "type": "Sequence",
        "pretokenizers": [
            ["type": "Split", "behavior": "Isolated", "invert": false, "pattern": ["Regex": LFMTokenizer.pattern]],
            ["type": "ByteLevel", "add_prefix_space": false, "use_regex": false]
        ]
    ], "added_tokens": []
]
let tokenizerURL = scratch.appendingPathComponent("tokenizer.json")
try JSONSerialization.data(withJSONObject: raw).write(to: tokenizerURL)
let tokenizer = try LFMTokenizer(url: tokenizerURL)
let v = Verbalizer(tokenizer: tokenizer)
let first = try v.aliases(["Å", "Å"])
let second = try v.aliases(["Å", "Å"])
check(first.map(\.1) == [256, 257], "single-cache canonical collision")
check(second.map(\.1) == [257, 256], "alias-cache canonical collision")
check(Array(second[0].0.utf8) == Array("Å".utf8), "cached code spelling changed")
let q = Question(kind: .choice, instructions: .string("Which?"), options: [("Å", PyJSON.scalar("null")), ("Å", PyJSON.scalar("null"))])
let readout = try v.readoutIDs(q)
check(readout == [[257], [256]], "readout assigned to wrong option")
print("Unicode: both byte spellings, reversed aliases and readout IDs pass")

func oldBPE(_ text: String) -> [Int] {
    if let id = vocab[text] { return [id] }
    var parts = text.utf8.map(Int.init)
    while parts.count > 1 {
        var best = Int.max, at = -1, id = 0
        for k in 0..<(parts.count - 1) {
            for (rank, pair) in mergeWords.enumerated() where vocab[pair.0] == parts[k] && vocab[pair.1] == parts[k + 1] && rank < best {
                best = rank; at = k; id = vocab[pair.0 + pair.1]!
            }
        }
        if at < 0 { break }
        parts[at] = id; parts.remove(at: at + 1)
    }
    return parts
}
for n in 1...256 {
    for text in [String(repeating: "a", count: n), "b" + String(repeating: "a", count: n)] {
        check(tokenizer.encode(text) == oldBPE(text), "BPE leftmost merge order changed at \(n)")
    }
}
print("BPE: 512 overlapping/tied/recursive merge strings match original scan")
for mode in [0, 2, 4, 6] {
    let picture = try Picture.load(fixtures.appendingPathComponent("png16-\(mode).png"))
    let want = Array(try Data(contentsOf: fixtures.appendingPathComponent("png16-\(mode).rgb")))
    check(picture.pixels == want, "PNG16 mode \(mode): \(picture.pixels) != \(want)")
}
print("PNG16: grayscale clamps; RGB, gray-alpha and RGBA take high byte (all 8 sample values)")
let config: [String: Any] = [
    "image_processor": [
        "image_processor_type": "Lfm2VlImageProcessor", "downsample_factor": 2,
        "encoder_patch_size": 16, "tile_size": 512, "min_image_tokens": 64, "max_image_tokens": 256, "min_tiles": 2,
        "max_tiles": 10, "use_thumbnail": true, "do_image_splitting": true, "max_pixels_tolerance": 2.0, "resample": 3,
        "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
    ]
]
let configURL = scratch.appendingPathComponent("processor.json")
try JSONSerialization.data(withJSONObject: config).write(to: configURL)
let processor = try PictureProcessor(config: configURL)
func peakBytes() -> UInt64 {
    var info = rusage_info_v4()
    let rc = withUnsafeMutablePointer(to: &info) { ptr in
        ptr.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) { proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0) }
    }
    check(rc == 0, "rusage failed")
    return info.ri_lifetime_max_phys_footprint
}
let startPeak = peakBytes()
var budget = PictureBudget()
var kept: [[Picture]] = []
var rejected = 0
for _ in 0..<256 { // largest wire batch, ten pictures per item, retained even beyond the worker's 32-item groups
    do {
        let pictures = try processor.loadCapped(Array(repeating: fixtures.appendingPathComponent("large.png"), count: 10), budget: &budget)
        check(pictures.allSatisfy { $0.width * $0.height <= PictureProcessor.maxPixels }, "original source retained")
        kept.append(pictures)
    } catch { rejected += 1; check("\(error)".contains("decoded-picture memory limit"), "unclean memory error") }
}
let peak = peakBytes()
check(kept.count == 4 && rejected == 252, "budget did not reject per item transactionally")
check(budget.retainedBytes <= PictureBudget.retainedLimit, "retained bytes over limit")
check(peak - startPeak < UInt64(PictureBudget.workingLimit), "peak decoded working memory exceeds limit: \(peak - startPeak)")
var bigBudget = PictureBudget()
for _ in 0..<256 {
    do {
        _ = try processor.loadCapped(Array(repeating: fixtures.appendingPathComponent("maximum-header.png"), count: 10), budget: &bigBudget)
        fatalError("maximum permitted source was decoded")
    } catch { check("\(error)".contains("decoded-picture memory limit"), "header not rejected before decode") }
}
check(bigBudget.retainedBytes == 0, "failed item consumed budget")
print(
    "Memory: 256×10 batch, \(kept.count) retained items, \(rejected) clean item errors, capped bytes \(budget.retainedBytes), peak delta \(peak - startPeak) bytes; 256 maximum-source items rejected before decode"
)
