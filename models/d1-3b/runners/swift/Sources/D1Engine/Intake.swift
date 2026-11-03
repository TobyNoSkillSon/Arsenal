// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision

/// Only the lab operation installs this observer. Each block consumes the original saved input.
final class D1ComponentSession {
    var inputs: [String: MLXArray]
    let capture: Bool
    var validShapes = true
    var used = Set<String>()
    var outputs: [String: MLXArray] = [:]
    let recorder = ComponentCapture()
    init(inputs: [String: MLXArray], capture: Bool) { self.inputs = inputs; self.capture = capture }
    func input(_ key: String, _ value: MLXArray) -> MLXArray {
        used.insert(key)
        if capture { eval(value); inputs[key] = value }
        if let saved = inputs[key], saved.shape != value.shape || saved.dtype != value.dtype { validShapes = false }
        // Shape/key validation happens before a replay is allowed to return evidence.
        return inputs[key] ?? value
    }
    /// Fail before executing readout when the exact reference input is absent or incompatible.
    func readout(_ key: String, hidden: MLXArray, compute: (MLXArray) -> MLXArray) throws -> MLXArray {
        guard capture || inputs[key].map({ $0.shape == hidden.shape && $0.dtype == hidden.dtype }) == true else {
            throw JSONError("Intake readout input missing or shape/dtype differs: \(key)")
        }
        return compute(input(key, hidden))
    }
    func output(_ key: String, _ value: MLXArray) {
        eval(value); outputs[key] = value; recorder.observe(key, value)
    }
}

public extension SystemOne {
    func intakeComponents(fixture: URL, output: URL, capture: Bool) throws {
        let archive = fixture.appendingPathComponent("inputs.safetensors")
        let session = D1ComponentSession(inputs: capture ? [:] : try TensorArchive(archive).arrays(), capture: capture)
        model.intake = session
        let vision = try visionTower(); vision.intake = session
        defer { model.intake = nil; vision.intake = nil }
        // Both a chain and the default shared-state tree, using real tokenization and readout IDs.
        let questions = [
            Question(kind: .noul, instructions: .string("Does the customer ask for a refund?")),
            Question(kind: .choice, instructions: .string("What is requested?"), options: [("refund", .string("A refund")), ("delivery", .string("A delivery update"))])
        ]
        for (index, state) in ["The customer asks for a refund.", "The package has not arrived. Please check the delivery."].enumerated() {
            model.intakePrefix = "case-\(index)/"
            var layout = Layout()
            let trunk = tokenizer.encode(Prompt.prefix(.string(state), bos: bos))
            let branches = try questions.map { tokenizer.encode(try verbalizer.suffix($0)) }
            if index == 0 { layout.add(trunk: trunk + branches[0], branches: []) } else { layout.add(trunk: trunk, branches: branches) }
            let ids = MLXArray(layout.ids)
            let saved = session.input("case-\(index)/ids", ids)
            guard saved.shape == ids.shape, saved.asArray(Int32.self) == layout.ids else { throw JSONError("Intake token IDs changed") }
            let hidden = model.hidden(layout)
            let selected = index == 0 ? [questions[0]] : questions
            let groups = try selected.map { try verbalizer.readoutIDs($0) }
            let tokens = Array(Set(groups.flatMap { $0.flatMap { $0 } })).sorted()
            let logits = try session.readout("case-\(index)/readout/input", hidden: hidden) { model.logits($0, tokens) }
            session.output("case-\(index)/readout", logits)
        }
        // A deterministic small patch grid exercises every SigLIP block, position interpolation and projector.
        let patches = MLXArray((0..<(16 * 768)).map { Float($0 % 251) / 125 - 1 }, [1, 16, 768]).asType(.bfloat16)
        _ = vision.intakeForward(session.input("vision/patches", patches))
        guard session.validShapes, Set(session.inputs.keys) == session.used else { throw JSONError("Intake fixture keys/count differ") }
        if capture {
            guard !FileManager.default.fileExists(atPath: fixture.path) else { throw JSONError("Fixture already exists") }
            try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
            try save(arrays: session.inputs, url: archive)
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: archive.path)
        }
        try session.recorder.write(session.outputs, to: output)
        model.intake = nil; vision.intake = nil
        var probabilities: [String: MLXArray] = [:]
        for (i, state) in ["The customer asks for a refund.", "The package has not arrived. Please check the delivery."].enumerated() {
            let selected = i == 0 ? [questions[0]] : questions
            let results = try run([Request(state: .string(state), questions: selected)])
            guard results.count == 1 else { throw JSONError("Probability case count mismatch") }
            let result = try results[0].get()
            for (j, values) in result.probabilities.enumerated() { probabilities["case-\(i)/question-\(j)"] = MLXArray(values.map(Float.init)) }
        }
        try save(arrays: probabilities, url: output.appendingPathComponent("probabilities.safetensors"))
    }
}
