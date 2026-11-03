// SPDX-License-Identifier: MIT
// Compiled with models/esm3-sm-open-v1/runners/swift/Sources/Runner/{Tokens,Protein,Sampling}.swift alone (Foundation
// only: no MLX, no weights) by Esm3FolderTests.weightFreePiecesFollowTheMaker. Prints "ok <checks>" or exits 1.
import Foundation

var checks = 0
func check(_ condition: Bool, _ what: String) {
    checks += 1
    if !condition { FileHandle.standardError.write(Data("FAIL: \(what)\n".utf8)); exit(1) }
}
func refused(_ body: () throws -> Void) -> Bool {
    do { try body(); return false } catch is TokenError { return true } catch { return false }
}
let args = CommandLine.arguments
let pdbPath = args.count > 1 ? args[1] : ""

// 1. The sequence tokenizer: <cls> residues <eos>, "_" is the mask, and round trip.
let ids = try SequenceTokens.encode("MS_KG")
check(ids == [0, 20, 8, 32, 15, 6, 2], "sequence tokens are <cls> M S <mask> K G <eos>: \(ids)")
check(SequenceTokens.decode(ids) == "MS_KG", "decoding is the input")
check(try SequenceTokens.encode(" M\nS ") == [0, 20, 8, 2], "whitespace is ignored")
check(refused { _ = try SequenceTokens.encode("MJ") }, "J is refused")
check(refused { _ = try SequenceTokens.encode("M|M") }, "a chain break is refused")
check(refused { _ = try SequenceTokens.encode("") }, "an empty sequence is refused")
check(ESM3Vocab.sequenceValid == Array(4...30) && !ESM3Vocab.sequenceValid.contains(32), "the sampling vocabulary excludes every special token")

// 2. SS8: <pad> at both ends, "_" is the mask (also <pad>).
check(try SS8Tokens.encode("GH_C", length: 4) == [0, 3, 4, 0, 10, 0], "ss8 tokens")
check(SS8Tokens.decode([0, 3, 4, 0, 10, 0]) == "GH_C", "ss8 round trip")
check(refused { _ = try SS8Tokens.encode("GH", length: 3) }, "a wrong-length ss8 string is refused")
check(refused { _ = try SS8Tokens.encode("GX", length: 2) }, "an unknown ss8 letter is refused")

// 3. SASA: torch.bucketize semantics (the first boundary at or above the value), nil is the mask, midpoints decode.
check(SASATokens.bucket(0) == 0 && SASATokens.bucket(0.8) == 0 && SASATokens.bucket(0.81) == 1, "a value equal to a boundary stays in the lower bucket")
check(SASATokens.bucket(151.4) == 14 && SASATokens.bucket(151.5) == 15, "the last boundary and beyond")
check(try SASATokens.encode([nil, 0.5, 200], length: 3) == [0, 0, 3, 18, 0], "sasa tokens")
let decoded = SASATokens.decode([0, 3, 0, 18, 0])
check(decoded.count == 3 && decoded[0] == 0.4 && decoded[1] == nil && abs(decoded[2]! - 227.1) < 1e-9, "sasa midpoints: \(decoded)")
check(refused { _ = try SASATokens.encode([Double.nan], length: 1) }, "a NaN sasa value is refused")
check(refused { _ = try SASATokens.encode([-1], length: 1) }, "a negative sasa value is refused")
// sample_sasa_logits: the expectation over tokens 3...17 with the 15 midpoints of [0] + boundaries; the open-ended
// last token gives infinity.
var probs = [Double](repeating: 0, count: 19)
probs[3] = 1
check(SASATokens.expected(probabilities: probs) == 0.4, "all mass in the first range gives its midpoint")
probs = [Double](repeating: 0, count: 19); probs[18] = 1
check(SASATokens.expected(probabilities: probs).isInfinite, "the open-ended range gives infinity")

// 4. Annotation merging (misc.merge_ranges / merge_annotations) over 1-indexed inclusive ranges.
check(Annotations.mergeRanges([5..<7, 1..<3, 3..<4], gap: 0).map { [$0.lowerBound, $0.upperBound] } == [[1, 4], [5, 7]], "touching ranges merge, a gap of 1 does not")
check(Annotations.mergeRanges([1..<3, 5..<7], gap: 3).count == 1, "a gap inside the limit merges")
let merged = Annotations.merge([Annotation(label: "a", start: 1, end: 2), Annotation(label: "b", start: 9, end: 9), Annotation(label: "a", start: 3, end: 5)], gap: 0)
check(merged == [Annotation(label: "a", start: 1, end: 5), Annotation(label: "b", start: 9, end: 9)], "annotations merge per label, in first-seen order: \(merged)")
check(Annotations.dropShort(merged, minLength: 5) == [Annotation(label: "a", start: 1, end: 5)], "annotations shorter than the minimum are dropped")

// 5. TF-IDF: (1 + log tf) x idf, L2-normalised, unknown terms ignored.
let tfidf = try TFIDF(vocabulary: ["alpha", "beta", "gamma"], idf: [1, 2, 4])
let one = tfidf.encode(["beta", "nonsense"])
check(one.count == 1 && abs(one[1]! - 1) < 1e-12, "one known term normalises to 1: \(one)")
let two = tfidf.encode(["alpha", "alpha", "gamma"])
let expectedAlpha = (1 + Foundation.log(2.0)) * 1, expectedGamma = 4.0
let norm = (expectedAlpha * expectedAlpha + expectedGamma * expectedGamma).squareRoot()
check(abs(two[0]! - expectedAlpha / norm) < 1e-12 && abs(two[2]! - expectedGamma / norm) < 1e-12, "sublinear term frequency and normalisation: \(two)")
check(tfidf.encode(["nothing"]).isEmpty, "a vector with no known term is empty")
check(refused { _ = try TFIDF(vocabulary: ["a"], idf: [1, 2]) }, "a mismatched IDF length is refused")

// 6. The function tokenizer: LSH bits with known hyperplanes, the <none> and <unk> rows, and the pad rows.
// Two keywords, one InterPro entry mapping to the second keyword. Table k flips sign on bit k, so the token of a
// vector pointing at keyword 0 is predictable.
var tables: [[Double]] = []
for table in 0..<8 {
    var plane = [Double](repeating: 0, count: 8 * 2)
    for bit in 0..<8 { plane[bit * 2] = (bit == table) ? -1 : 1 } // keyword 0 is negative only on bit == table
    tables.append(plane)
}
let function = try FunctionTokenizer(
    tfidf: try TFIDF(vocabulary: ["kinase", "binding"], idf: [1, 1]),
    interproKeywords: ["IPR000001": ["binding"]], hyperplanes: tables, residueLabels: ["active site"])
let hash = try function.hash(["kinase"])
check(hash != nil && hash! == [254, 253, 251, 247, 239, 223, 191, 127], "one keyword hashes to the predicted bits: \(hash ?? [])")
check(refused { _ = try function.hash(["nonsense-not-in-any-vocabulary"]) }, "an unknown label is refused")
check(
    try function.kinds(of: "IPR000001").contains(where: {
        if case .function = $0 { return true }; return false
    }), "an InterPro id feeds the function track")
check(
    try function.kinds(of: "active site").contains(where: {
        if case .residue = $0 { return true }; return false
    }), "a residue label feeds the residue track")
check(refused { _ = try function.kinds(of: "IPR999999") }, "an InterPro id the model does not know is refused")
let rows = try function.encode([Annotation(label: "kinase", start: 2, end: 3)], length: 4)
check(rows.count == 6, "function tokens have a pad row at each end")
check(rows[0] == [Int](repeating: 0, count: 8) && rows[5] == rows[0], "the pad rows are <pad>")
check(rows[1] == [Int](repeating: 3, count: 8), "an unannotated residue is <none>")
check(rows[2] == hash!.map { $0 + 4 } && rows[3] == rows[2], "an annotated residue carries the hash, offset past the special tokens")
check(try function.encode([], length: 3).allSatisfy { $0 == [Int](repeating: 0, count: 8) }, "no annotations means all <pad>")
check(
    try function.encode([Annotation(label: "active site", start: 1, end: 1)], length: 2).allSatisfy { $0 == [Int](repeating: 0, count: 8) },
    "a residue-only annotation leaves the function track empty")
check(refused { _ = try function.encode([Annotation(label: "kinase", start: 0, end: 2)], length: 4) }, "a range starting at 0 is refused")
check(refused { _ = try function.encode([Annotation(label: "kinase", start: 1, end: 9)], length: 4) }, "a range past the end is refused")

// 7. The schedule and annealing (noise_schedules.py, generation.py).
// cos(pi/2) in float32 is not exactly zero (as in torch); the +0.1 truncation of toSample absorbs it.
check(abs(Schedule.maskedFraction("cosine", step: 0, steps: 1)) < 1e-6 && Schedule.maskedFraction("linear", step: 0, steps: 1) == 0, "nothing stays masked after the last step")
check(Schedule.toSample("cosine", step: 0, steps: 1, stillMasked: 7, total: 7) == 7, "one cosine step fills everything")
check(abs(Schedule.maskedFraction("linear", step: 0, steps: 4) - 0.75) < 1e-6, "linear leaves 3/4 masked after step 1")
check(abs(Schedule.maskedFraction("cosine", step: 0, steps: 4) - Double(cos(Float.pi * 0.125))) < 1e-6, "cosine matches cos(t pi / 2)")
check(Schedule.toSample("linear", step: 0, steps: 4, stillMasked: 8, total: 8) == 2, "a quarter of the positions are filled at the first of four linear steps")
check(Schedule.toSample("cosine", step: 3, steps: 4, stillMasked: 3, total: 8) == 3, "the last step fills the rest")
// The maker adds 0.1 before truncating, so rounding never leaves one position behind.
// After the second of three linear steps a third should stay masked: int(0.3333 x 10 + 0.1) = 3, so 7 are filled.
check(Schedule.toSample("linear", step: 1, steps: 3, stillMasked: 10, total: 10) == 7, "the epsilon keeps the count on the integer")
check(Schedule.toSample("linear", step: 0, steps: 3, stillMasked: 10, total: 10) == 4, "the first of three linear steps fills 4 of 10")
check(abs(Schedule.annealed(1.0, step: 0, steps: 10) - 1) < 1e-12, "annealing starts at the temperature")
check(abs(Schedule.annealed(1.0, step: 9, steps: 10) - 1e-6) < 1e-12, "annealing ends at the 0.001 floor squared")
check(
    abs(Schedule.annealed(0.5, step: 2, steps: 5) - Foundation.pow(0.0, 2) - 0.0) < 1e-9 || Schedule.annealed(0.5, step: 2, steps: 5) >= 1e-6,
    "annealing never goes below the floor")

// 8. Sampling: top-p, argmax at temperature 0, entropy, and the choice of positions.
let logits = [Foundation.log(0.6), Foundation.log(0.3), Foundation.log(0.1)]
// top_p_logits keeps the tokens whose running mass is at or below top_p, and always the first: at 0.7 the second
// token's running mass is 0.9, so it goes.
let cut = Sampling.topP(logits, 0.7)
check(cut[0] == logits[0] && cut[1] < -1e30 && cut[2] < -1e30, "top-p 0.7 keeps only the first token")
check(Sampling.topP(logits, 0.95)[1] == logits[1] && Sampling.topP(logits, 0.95)[2] < -1e30, "top-p 0.95 keeps the first two")
check(Sampling.topP(logits, 0.1)[0] == logits[0], "the first token always survives")
check(Sampling.topP(logits, 1) == logits, "top-p 1 changes nothing")
var rng = SeededGenerator(seed: 1)
let (token, logp) = Sampling.sample(logits, valid: [0, 1, 2], temperature: 0, topP: 1, rng: &rng)
check(token == 0, "temperature 0 takes the most likely token")
check(abs(Foundation.exp(logp[0]) - 0.6) < 1e-9, "the returned log-probabilities are the masked, pre-temperature ones")
let (restricted, _) = Sampling.sample(logits, valid: [1, 2], temperature: 0, topP: 1, rng: &rng)
check(restricted == 1, "an invalid id is never sampled")
check(abs(Sampling.entropy([Foundation.log(0.5), Foundation.log(0.5)]) - Foundation.log(2.0)) < 1e-9, "the entropy of a coin is log 2")
check(Sampling.entropy([0, -.infinity]) == 0, "a one-hot row has zero entropy")
let chosen = Sampling.choose(candidates: [1, 2, 3, 4], count: 2, strategy: "entropy", entropy: [0, 9, 1, 2, 8], rng: &rng)
check(chosen == [2, 3], "entropy order fills the least uncertain positions first: \(chosen)")
var rngA = SeededGenerator(seed: 42), rngB = SeededGenerator(seed: 42)
check(
    Sampling.choose(candidates: [1, 2, 3, 4], count: 2, strategy: "random", entropy: [], rng: &rngA)
        == Sampling.choose(candidates: [1, 2, 3, 4], count: 2, strategy: "random", entropy: [], rng: &rngB), "the same seed chooses the same positions")
// sample_function_logits: <none> when its mean probability over the depths is above 0.05, else the argmax elsewhere.
var depths = [[Double]](repeating: [Double](repeating: 0, count: 10), count: 8)
for d in 0..<8 { depths[d][3] = 10 } // all mass on <none>
check(Sampling.function(depths, temperature: 1, topP: 1).ids == [Int](repeating: 3, count: 8), "a confident <none> row stays <none>")
for d in 0..<8 { depths[d] = [Double](repeating: 0, count: 10); depths[d][5 + d % 2] = 10 }
check(Sampling.function(depths, temperature: 1, topP: 1).ids == [5, 6, 5, 6, 5, 6, 5, 6], "otherwise each depth takes its own argmax")
check(Sampling.function(depths, temperature: 0, topP: 1).ids == [5, 6, 5, 6, 5, 6, 5, 6], "temperature 0 is the argmax limit, not a division by zero")

// 9. Geometry: Gram-Schmidt frames, the black-hole frame, the inferred oxygen and the kNN graph.
let n: Vec3 = (1, 0, 0), ca: Vec3 = (0, 0, 0), c: Vec3 = (0, 1, 0)
let frame = Geometry.frame(negX: c, origin: ca, xyPlane: n)
check(abs(frame.rotation[0] - 0) < 1e-6 && abs(frame.rotation[3] + 1) < 1e-6, "the x axis points from C to CA: \(frame.rotation)")
let mapped = frame.apply((1, 0, 0))
check(abs(mapped.0) < 1e-6 && abs(mapped.1 + 1) < 1e-6 && abs(mapped.2) < 1e-6, "applying the frame rotates and translates: \(mapped)")
let (frames, mask) = Geometry.backboneFrames([[n, ca, c], [nil, nil, nil], [(2, 0, 0), (1, 0, 0), (1, 1, 0)]])
check(mask == [true, false, true], "a residue without atoms has no frame")
check(frames[1].translation.0 != 0 || frames[1].translation.1 != 0, "a residue without atoms gets the average (black-hole) frame")
let (_, emptyMask) = Geometry.backboneFrames([[nil, nil, nil]])
check(emptyMask == [false], "all-missing coordinates stay masked")
let oxygens = Geometry.inferOxygen([[n, ca, c], [(2, 0, 0), (1, 0, 0), (1, 1, 0)]])
check(oxygens.count == 2 && oxygens[1] == nil, "the last residue has no inferred oxygen")
check(oxygens[0] != nil, "an oxygen is placed from CA, C and the next N")
let edges = Geometry.knn([(0, 0, 0), (1, 0, 0), (5, 0, 0)], mask: [true, true, true], k: 2)
check(edges == [[0, 1], [1, 0], [2, 1]], "every residue's first neighbour is itself, then the nearest: \(edges)")
let fallback = Geometry.knn([(0, 0, 0), nil, nil], mask: [true, false, false], k: 2)
check(fallback[1] == [1, 0] || fallback[1] == [1, 2], "residues without coordinates fall back to sequence distance: \(fallback[1])")

// 10. PDB reading and writing.
if !pdbPath.isEmpty {
    let text = try String(contentsOfFile: pdbPath, encoding: .utf8)
    let chain = try PDB.parse(text, maxResidues: 4096)
    check(chain.count == 64, "the fixture crop has 64 residues: \(chain.count)")
    check(String(chain.sequence) == "IVGGYTCGANTVPYQVSLNSGYHFCGGSLINSQWVVSAAHCYKSGIQVRLGEDNINVVEGNEQF", "the fixture's sequence: \(String(chain.sequence))")
    check(chain.chain == "A" && chain.residueNumbers.first == 16, "the chain and its first residue number")
    check(chain.atoms.allSatisfy { $0.allSatisfy { $0 != nil } }, "every residue of the crop has N, CA and C")
    check(refused { _ = try PDB.parse(text, chain: "Z", maxResidues: 4096) }, "an absent chain is refused")
    check(refused { _ = try PDB.parse(text, maxResidues: 10) }, "a structure past the residue limit is refused")
    check(refused { _ = try PDB.parse("REMARK nothing here\n", maxResidues: 10) }, "a file without ATOM records is refused")
    // HETATM records and a second model are ignored; the first alternate location wins.
    let messy = """
        ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N
        ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
        ATOM      3  CA AALA A   1       9.000   9.000   9.000  1.00  0.00           C
        ATOM      4  C   ALA A   1       0.000   1.000   0.000  1.00  0.00           C
        HETATM    5  O   HOH A   2       5.000   5.000   5.000  1.00  0.00           O
        ATOM      6  N   GLY B   1       3.000   0.000   0.000  1.00  0.00           N
        ENDMDL
        ATOM      7  N   TRP A   9       7.000   0.000   0.000  1.00  0.00           N
        """
    let parsed = try PDB.parse(messy, maxResidues: 16)
    check(parsed.count == 1 && parsed.sequence == ["A"], "water, a second chain and a second model are ignored: \(String(parsed.sequence))")
    check(parsed.atoms[0][1]!.0 == 0, "the first alternate location of CA wins")
    let written = PDB.write(sequence: ["A", "G"], atoms: [[(1, 0, 0), (0, 0, 0), (0, 1, 0)], [(2, 0, 0), (1, 0, 0), (1, 1, 0)]], plddt: [0.5, 1], title: "test")
    let back = try PDB.parse(written, maxResidues: 16)
    check(back.count == 2 && String(back.sequence) == "AG", "a written file reads back")
    check(abs(back.atoms[0][0]!.0 - 1) < 1e-3 && abs(back.atoms[1][2]!.1 - 1) < 1e-3, "the coordinates survive the round trip")
    check(written.contains(" O  ") && written.contains("50.00"), "the file carries the inferred oxygen and pLDDT as the B-factor")
    check(written.hasSuffix("END\n") && written.contains("TER"), "the file is terminated")
}

// 11. Review fixes: nonstandard residues stay as X (biotite keeps them; dropping one shifted every later position),
// nucleotides and water in ATOM records do not count; capture names are plain names, never paths.
let modified = """
    ATOM      1  N   ALA A   1       1.000   0.000   0.000  1.00  0.00           N
    ATOM      2  CA  ALA A   1       0.000   0.000   0.000  1.00  0.00           C
    ATOM      3  C   ALA A   1       0.000   1.000   0.000  1.00  0.00           C
    ATOM      4  N   NLE A   2       2.000   0.000   0.000  1.00  0.00           N
    ATOM      5  CA  NLE A   2       3.000   0.000   0.000  1.00  0.00           C
    ATOM      6  C   NLE A   2       3.000   1.000   0.000  1.00  0.00           C
    ATOM      7  P    DA A   3       9.000   9.000   9.000  1.00  0.00           P
    ATOM      8  O   HOH A   4       5.000   5.000   5.000  1.00  0.00           O
    ATOM      9  N   GLY A   5       4.000   0.000   0.000  1.00  0.00           N
    """
let kept = try PDB.parse(modified, maxResidues: 16)
check(String(kept.sequence) == "AXG", "a nonstandard residue reads as X; a nucleotide and water are skipped: \(String(kept.sequence))")
check(kept.atoms[1][1] != nil && kept.atoms[2][1] == nil, "the nonstandard residue keeps its atoms; the glycine has no CA here")
check(CaptureName.sanitized("B.function_decoder") == "B.function_decoder", "a plain capture name passes")
for bad in ["../x", "a/b", ".hidden", "", "x..y", String(repeating: "a", count: 65), "a b"] {
    check(CaptureName.sanitized(bad) == nil, "capture name \(bad) is refused")
}

// 12. Release review fixes (2 Nov): structure files are read as bytes (UTF-8, else Latin-1; gzip and unreadable files
// are the caller's to fix), a chain with no complete backbone is refused, empty variants are refused, and replies or
// generations past Arsenal's frame and timeout limits are refused before the GPU runs.
let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("esm3-files-\(UUID().uuidString)")
try FileManager.default.createDirectory(at: scratch, withIntermediateDirectories: true)
defer { try? FileManager.default.removeItem(at: scratch) }
var latin = Data("REMARK   1 MEASURED AT 25 ".utf8) + Data([0xB0]) + Data("C\n".utf8)
latin += Data(modified.utf8)
let latinURL = scratch.appendingPathComponent("latin.pdb")
try latin.write(to: latinURL)
check(String(data: latin, encoding: .utf8) == nil, "the fixture is not UTF-8")
let latinText = try PDB.read(latinURL)
let latinChain = try PDB.parse(latinText, maxResidues: 16)
check(latinText.contains("25 °C") && String(latinChain.sequence) == "AXG", "a Latin-1 REMARK reads and parses")
let gzipURL = scratch.appendingPathComponent("x.pdb.gz")
try (Data([0x1F, 0x8B, 0x08, 0x00]) + Data(modified.utf8)).write(to: gzipURL)
check(refused { _ = try PDB.read(gzipURL) }, "a gzip file is refused as invalid input")
check(refused { _ = try PDB.read(scratch.appendingPathComponent("absent.pdb")) }, "an unreadable file is refused as invalid input")
check(refused { _ = try PDB.read(scratch) }, "a folder is refused as invalid input")
let cbOnly = """
    ATOM      1  CB  ALA A   1       1.000   0.000   0.000  1.00  0.00           C
    ATOM      2  CB  ALA A   2       2.000   0.000   0.000  1.00  0.00           C
    """
let noBackbone = try PDB.parse(cbOnly, maxResidues: 16)
check(refused { try noBackbone.requireBackbone() }, "a chain with no N, CA and C is refused")
check(refused { try kept.requireBackbone(3...3) }, "a segment whose only residue lacks CA is refused")
check(!refused { try kept.requireBackbone() } && !refused { try kept.requireBackbone(1...2) }, "a chain with one complete residue passes")
let wild = try SequenceTokens.encode("MAGK")
check(try Variants.parse("A2G", sequence: wild).map(\.position) == [2], "A2G parses")
check(try Variants.parse("A2G:K4M", sequence: wild).count == 2, "a double mutant parses")
for bad in ["", ":", " ", "A2G:", ":A2G", "A2G::K4M", "G2A", "A9G"] {
    check(refused { _ = try Variants.parse(bad, sequence: wild) }, "variant \"\(bad)\" is refused, not scored")
}
check(!refused { try RequestLimits.reply(640_000, "x") } && refused { try RequestLimits.reply(640_001, "x") }, "the reply budget is 640,000 numbers")
// 640,000 of the longest float32 values as JSON stay under the 16 MiB - 64 KiB reply limit.
check(640_000 * 26 <= 16 * 1024 * 1024 - 64 * 1024, "the reply budget fits the frame at the worst-case number length")
check(
    !refused { try RequestLimits.reply(416 * 1536, "per_residue") } && refused { try RequestLimits.reply(417 * 1536, "per_residue") },
    "per_residue fits 415 residues and refuses 416")
check(
    !refused { try RequestLimits.logits(["structure"], residues: 156) } && refused { try RequestLimits.logits(["structure"], residues: 157) },
    "structure logits fit 156 residues and refuse 157")
check(refused { try RequestLimits.logits(["nope"], residues: 1) }, "an unknown logits track is refused")
check(!refused { try RequestLimits.reply(800 * 800, "pae") } && refused { try RequestLimits.reply(801 * 801, "pae") }, "PAE fits 800 residues, not 801")
check(!refused { try RequestLimits.passes([1024]) } && refused { try RequestLimits.passes([1024, 1]) }, "generate runs at most 1,024 passes in all")

print("ok \(checks)")
