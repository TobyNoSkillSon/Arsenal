// SPDX-License-Identifier: MIT
// The maker's answer assembly (OneJev qev/answers.py and calibrate.softmax @81ce62f, Apache-2.0; the confidence
// formulas are TypeSafe AI's system-one-adapter-python, MIT).
import Foundation

public enum Answers {
    /// `calibrate.softmax` in float64 with an optional temperature.
    public static func softmax(_ logits: [Double], temperature: Double = 1) -> [Double] {
        let z = logits.map { $0 / temperature }
        let m = z.max() ?? 0
        let w = z.map { Foundation.exp($0 - m) }
        let s = w.reduce(0, +)
        return w.map { $0 / s }
    }

    static func normalize(_ p: [Double]) -> [Double] {
        let total = p.reduce(0, +)
        return total <= 0 ? p.map { _ in 1 / Double(p.count) } : p.map { $0 / total }
    }

    /// Python's `round(x, 6)`: correctly rounded to six decimals, ties to even on the exact binary value.
    public static func round6(_ x: Double) -> Double { Double(String(format: "%.6f", x)) ?? x }

    public static func choiceConfidence(_ probs: [Double]) -> Double {
        if probs.count == 1 { return 1 }
        let p = normalize(probs), uniform = 1 / Double(p.count)
        return (p.max()! - uniform) / (1 - uniform)
    }

    public static func scoreConfidence(_ probs: [Double]) -> Double {
        if probs.count == 1 { return 1 }
        let p = normalize(probs), k = p.count
        let mode = argmax(p)
        var distance = 0.0, uniformMAD = 0.0
        let center = Double(k - 1) / 2
        for (i, pi) in p.enumerated() {
            distance += pi * Double(Swift.abs(i - mode))
            uniformMAD += Swift.abs(Double(i) - center)
        }
        uniformMAD /= Double(k)
        return Swift.max(0, 1 - distance / uniformMAD)
    }

    /// Python's `max(range(k), key=p.__getitem__)`: the first index of the largest value.
    static func argmax(_ p: [Double]) -> Int {
        var best = 0
        for i in p.indices where p[i] > p[best] { best = i }
        return best
    }

    static func number(_ d: Double) -> PyJSON { .scalar(PyJSON.floatRepr(d)) }

    public static func noul(_ pYes: Double) -> PyJSON {
        .object([("type", .string("noul")), ("noul", number(round6(Swift.min(1, Swift.max(0, pYes)))))])
    }

    public static func choice(_ options: [String], _ probs: [Double]) -> PyJSON {
        let p = normalize(probs)
        // pydantic's dict keeps the last value for a repeated name; names are unique after validation.
        let shares: [(String, PyJSON)] = zip(options, p).map { ($0, number(round6($1))) }
        let fields: [(String, PyJSON)] = [
            ("type", .string("choice")), ("choice", .string(options[argmax(p)])), ("probabilities", .object(shares)),
            ("confidence", number(round6(choiceConfidence(p))))
        ]
        return .object(fields)
    }

    public static func score(_ levels: [PyJSON?], _ probs: [Double]) -> PyJSON {
        let p = normalize(probs)
        var expected = 0.0
        for (i, pi) in p.enumerated() { expected += Double(i) * pi }
        let legend: [(String, PyJSON)] = levels.enumerated().map { (String($0.offset), $0.element ?? PyJSON.scalar("null")) }
        let shares: [(String, PyJSON)] = p.enumerated().map { (String($0.offset), number(round6($0.element))) }
        let fields: [(String, PyJSON)] = [
            ("type", .string("score")), ("score", number(round6(expected))), ("legend", .object(legend)),
            ("probabilities", .object(shares)), ("confidence", number(round6(scoreConfidence(p))))
        ]
        return .object(fields)
    }
}
