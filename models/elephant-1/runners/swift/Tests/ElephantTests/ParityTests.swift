// SPDX-License-Identifier: MIT
// Heavy: the MLX port at FP16 against the maker's capture (OneJev qev @81ce62f + serve.patch on MPS, bfloat16 and
// float16; lab/evidence/elephant-1/maker-2026-11-02). Bounds were frozen before the port first ran
// (lab/evidence/elephant-1/bounds-2026-11-02.json). Run with
//   ARSENAL_HEAVY_TESTS=1 ELEPHANT_WEIGHTS=<FP16 snapshot from the convert op> ELEPHANT_EVIDENCE=<maker capture dir>
//   [ELEPHANT_PARITY_OUT=<json>] scripts/test.sh --heavy --filter ElephantTests
import ElephantEngine
import Foundation
import MLX
import Testing

struct MakerReference: Decodable {
    struct Case: Decodable { let id: String; let prefix_ids: [Int]; let suffix_ids: [[Int]]; let slot_logits: [[Double]] }
    let cases: [Case]
}

extension Trait where Self == ConditionTrait {
    static var heavy: Self { .enabled(if: ProcessInfo.processInfo.environment["ARSENAL_HEAVY_TESTS"] == "1") }
}

@Suite(.serialized) struct ParityTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    @Test(.heavy) func fp16PortMatchesTheMaker() throws {
        let env = ProcessInfo.processInfo.environment
        let weights = URL(fileURLWithPath: try #require(env["ELEPHANT_WEIGHTS"]))
        let evidence = URL(fileURLWithPath: try #require(env["ELEPHANT_EVIDENCE"]))
        func reference(_ dtype: String) throws -> [MakerReference.Case] {
            try JSONDecoder().decode(MakerReference.self, from: Data(contentsOf: evidence.appendingPathComponent("reference-\(dtype).json"))).cases
        }
        let bf16 = try reference("bfloat16"), fp16 = try reference("float16")
        let cases = try makerFixture()
        #expect(cases.count == bf16.count && cases.count == fp16.count)
        let engine = try ElephantEngine(weights: weights)
        #expect(engine.model.dtype == .float16 && engine.maxSlots == 256)

        var report: [String: Any] = [:]
        var worstBF16 = 0.0, worstFP16 = 0.0, flips = 0, branches = 0
        var perCase: [[String: Any]] = []
        for (c, (rb, rf)) in zip(cases, zip(bf16, fp16)) {
            let (state, questions) = try parsed(c)
            let plan = try engine.branches(questions.map(\.1))
            let layout = try engine.layout(state, plan.rendered)
            // Counts before values: every branch's token ids are the maker's.
            #expect(layout.prefix == rb.prefix_ids, "\(c.id) prefix")
            #expect(layout.suffixes == rb.suffix_ids, "\(c.id) suffixes")
            let results = try engine.run(layout, plan.rendered)
            #expect(results.count == rb.slot_logits.count)
            var caseWorst = 0.0
            for (branch, (lb, lf)) in zip(results, zip(rb.slot_logits, rf.slot_logits)) {
                let p = Answers.softmax(branch.logits), pb = Answers.softmax(lb), pf = Answers.softmax(lf)
                #expect(p.count == pb.count)
                let db = zip(p, pb).map { Swift.abs($0 - $1) }.max()!, df = zip(p, pf).map { Swift.abs($0 - $1) }.max()!
                worstBF16 = Swift.max(worstBF16, db); worstFP16 = Swift.max(worstFP16, df); caseWorst = Swift.max(caseWorst, Swift.max(db, df))
                let a = p.firstIndex(of: p.max()!)!
                if a != pb.firstIndex(of: pb.max()!)! || a != pf.firstIndex(of: pf.max()!)! { flips += 1 }
                branches += 1
            }
            perCase.append([
                "id": c.id, "prefix_tokens": layout.prefix.count, "max_abs_dp": caseWorst,
                "probabilities": results.map { Answers.softmax($0.logits).map { Answers.round6($0) } }
            ])
        }
        #expect(branches == 15)
        #expect(flips == 0)
        #expect(worstBF16 <= 0.025 && worstFP16 <= 0.025)
        report["answers"] = ["branches": branches, "argmax_flips": flips, "max_abs_dp_vs_maker_bf16": worstBF16, "max_abs_dp_vs_maker_fp16": worstFP16, "cases": perCase]

        // Components: first case, first branch; last token of the prefix pass (pass0) and the suffix pass (pass1).
        let components = try #require(
            try JSONSerialization.jsonObject(with: Data(contentsOf: evidence.appendingPathComponent("components-float16.json"))) as? [String: [Double]])
        let first = cases[0]
        let (state, questions) = try parsed(first)
        let plan = try engine.branches([questions[0].1])
        let layout = try engine.layout(state, plan.rendered)
        var pass = -1, captured: [String: [Float]] = [:]
        _ = try engine.run(layout, plan.rendered) { name, value in
            if name == "embed" { pass += 1 }
            captured["pass\(pass).\(name)"] = value.asType(.float32).asArray(Float.self)
        }
        #expect(captured.count == components.count && Set(captured.keys) == Set(components.keys))
        var worstCosine = 1.0, worstName = "", rows: [[String: Any]] = []
        var maxMagnitude: Float = 0
        for name in components.keys.sorted() {
            let a = captured[name]!.map(Double.init), b = components[name]!
            maxMagnitude = Swift.max(maxMagnitude, captured[name]!.map(Swift.abs).max()!)
            let dot = zip(a, b).reduce(0) { $0 + $1.0 * $1.1 }
            let na = a.reduce(0) { $0 + $1 * $1 }.squareRoot(), nb = b.reduce(0) { $0 + $1 * $1 }.squareRoot()
            let diff = zip(a, b).reduce(0) { $0 + ($1.0 - $1.1) * ($1.0 - $1.1) }.squareRoot()
            let cosine = dot / (na * nb)
            rows.append(["name": name, "cosine": cosine, "relative_l2": diff / nb])
            if cosine < worstCosine { worstCosine = cosine; worstName = name }
            let finite = captured[name]!.allSatisfy { $0.isFinite }
            #expect(finite, "\(name) finite")
        }
        #expect(worstCosine >= 0.999, "worst \(worstName) \(worstCosine)")
        report["components"] = ["count": rows.count, "worst_cosine": worstCosine, "worst": worstName, "max_abs_last_token": maxMagnitude, "rows": rows]
        report["peak_mlx_mb"] = Double(Memory.peakMemory) / 1e6
        if let out = env["ELEPHANT_PARITY_OUT"] {
            try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys]).write(to: URL(fileURLWithPath: out))
        }
    }
}
