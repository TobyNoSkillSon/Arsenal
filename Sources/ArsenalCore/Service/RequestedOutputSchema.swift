// SPDX-License-Identifier: MIT
import Foundation

/// Request-specific output schemas are enforced after inference on every transport, in addition to the operation's declared result schema.
public enum RequestedOutputSchema {
    public static func validate(_ result: JSONObject, operation: String, input: JSONObject) throws {
        let schema = operation == "chat" ? input["response_format"]?["json_schema"]?["schema"]?.object : operation == "extract" ? input["json_schema"]?.object : nil
        guard let schema else { return }
        let value: JSONValue
        if operation == "extract" {
            guard let data = result["data"] else { throw WorkerFailure(code: "invalid_output", message: "Extraction returned no structured data.") }; value = data
        } else {
            guard let text = result["text"]?.string, let parsed = try? JSONDecoder().decode(JSONValue.self, from: Data(text.utf8)) else {
                throw WorkerFailure(code: "invalid_output", message: "Model did not return JSON for the requested output schema.")
            }; value = parsed
        }
        do { try OperationSchemaValidator.validate(value, schema: .object(schema)) } catch {
            throw WorkerFailure(code: "invalid_output", message: "Model output failed the requested schema: " + error.localizedDescription)
        }
    }
}
