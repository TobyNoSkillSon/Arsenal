// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision

/// Lab-only component observer for Arsenal intake (`measurement_components`). Installed only for that operation:
/// every component consumes its original saved 16-bit input, so a candidate's error never feeds the next component.
final class LayaIntakeSession {
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
        return inputs[key] ?? value
    }
    func output(_ key: String, _ value: MLXArray) { eval(value); outputs[key] = value }
}

public extension LayaModel {
    /// The intake fixture: a one-row English yes/no decision and a one-row two-option choice, each unpadded and unmasked
    /// (one row fills its length), through every encoder layer, head layer and the scorer. The supervisor sends the two
    /// states in the model's supported languages (Multilingual: the second is German; English-only models: English).
    internal static let intakeQuestions = [
        Question(id: "refund", kind: .noul, instructions: "Does the customer ask for a refund?", criteria: []),
        Question(id: "request", kind: .choice, instructions: "What is requested?", criteria: [("refund", "A refund"), ("delivery", "A delivery update")])
    ]
    internal static let intakeStates = ["The customer asks for a refund.", "The package has not arrived. Please check the delivery."]

    /// Component keys per case: embedding, every encoder layer, every head layer, scorer.
    static func intakeComponentKeys(layers: Int, headLayers: Int) -> Set<String> {
        var keys = Set<String>()
        for c in 0..<2 {
            keys.insert("case-\(c)/embedding"); keys.insert("case-\(c)/scorer")
            for i in 0..<layers { keys.insert("case-\(c)/layer-\(i)") }
            for i in 0..<headLayers { keys.insert("case-\(c)/head-\(i)") }
        }
        return keys
    }

    func intakeComponents(fixture: URL, output: URL, capture: Bool, states: [String]? = nil) throws {
        let states = states ?? Self.intakeStates
        guard states.count == 2, states.allSatisfy({ !$0.isEmpty }) else { throw LayaError.invalid("Intake needs two case states") }
        let archive = fixture.appendingPathComponent("inputs.safetensors")
        let session = LayaIntakeSession(inputs: capture ? [:] : try TensorArchive(archive).arrays(), capture: capture)
        for (index, state) in states.enumerated() {
            let row = try prepare(Item(text: state), Self.intakeQuestions[index])
            let ids = MLXArray(row.ids.map(Int32.init))
            let saved = session.input("case-\(index)/ids", ids)
            guard saved.shape == ids.shape, saved.asArray(Int32.self) == row.ids.map(Int32.init) else { throw LayaError.invalid("Intake token IDs changed") }
            _ = intakeForward(row, session: session, prefix: "case-\(index)/")
        }
        guard session.validShapes, Set(session.inputs.keys) == session.used,
            Set(session.outputs.keys) == Self.intakeComponentKeys(layers: intakeLayers, headLayers: intakeHeadLayers)
        else { throw LayaError.invalid("Intake fixture keys/count differ") }
        if capture {
            guard !FileManager.default.fileExists(atPath: fixture.path) else { throw LayaError.invalid("Fixture already exists") }
            try FileManager.default.createDirectory(at: fixture, withIntermediateDirectories: true)
            try save(arrays: session.inputs, url: archive)
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: archive.path)
        }
        try session.recorder.write(session.outputs, to: output)
        // Served-path decision probabilities (production padding/chunking for one item), unrounded.
        var probabilities: [String: MLXArray] = [:]
        let cases: [(String, [Question])] = [(states[0], [Self.intakeQuestions[0]]), (states[1], Self.intakeQuestions)]
        for (i, (state, questions)) in cases.enumerated() {
            for (j, question) in questions.enumerated() {
                let row = try prepare(Item(text: state), question)
                let values = try logits([row])[0]
                probabilities["case-\(i)/question-\(j)"] = MLXArray(calibratedProbabilities(values, optionCount: row.markers.count, question: question))
            }
        }
        try save(arrays: probabilities, url: output.appendingPathComponent("probabilities.safetensors"))
    }
}
