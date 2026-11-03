// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

func replicateSoundParams(_ input: JSONObject) throws -> JSONObject {
    guard let prompt = input["prompt"]?.string, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw HTTPFailure(422, "prompt must be a nonempty string.") }
    var params: JSONObject = ["prompt": .string(prompt)]
    if let value = input["duration"] {
        guard let duration = value.number, duration.isFinite, duration > 0, duration <= 380 else {
            throw HTTPFailure(422, "duration must be greater than zero and at most 380 seconds.")
        }
        params["duration"] = value
    }
    for key in ["seed", "steps", "negative_prompt", "guidance", "lyrics", "language", "instrumental"] { params[key] = input[key] }
    if let seed = params["seed"] { _ = try jsonInteger(seed, name: "seed", range: 0...Int(UInt32.max)) }
    if let steps = params["steps"] { _ = try jsonInteger(steps, name: "steps", range: 1...100) }
    if let negative = params["negative_prompt"], negative.string == nil { throw HTTPFailure(422, "negative_prompt must be a string.") }
    return params
}
