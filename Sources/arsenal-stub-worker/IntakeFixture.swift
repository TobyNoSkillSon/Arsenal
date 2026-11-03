// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// Tiny protocol fixture: output dtype comes from the loaded archive, not the requested Load label.
final class IntakeFixture {
    let fp16Passes: Bool
    var dtype = "BF16"
    init(fp16Passes: Bool) { self.fp16Passes = fp16Passes }

    func handle(_ op: String, request: JSONObject) throws -> JSONObject? {
        switch op {
        case "convert", "measurement_source":
            let source = URL(fileURLWithPath: request["src_dir"]!.string!)
            // Both half casts must consume the pinned source, not a previously converted snapshot.
            guard !FileManager.default.fileExists(atPath: source.appendingPathComponent("arsenal-snapshot.json").path) else {
                return ["ok": false, "error": ["code": "fixture_source", "message": "Half trial was derived from the selected base"]]
            }
            let half = op == "measurement_source" || request["dtype"] == "bf16" ? "BF16" : "F16"
            let dst = request["dst_dir"]!.string!
            let metadata: JSONObject = ["dtype": .string(half), "source_is_original": true, "recipe": request["recipe"] ?? .null]
            let info = try WireJSON.encode(metadata)
            return [
                "files": .array([
                    try file(dst + "/weights.safetensors", Self.tensor(keys: ["weight"], dtype: half, value: 1)),
                    try file(dst + "/manifest.json", info), try file(dst + "/component-coverage.json", info)
                ])
            ]
        case "load":
            let weights = URL(fileURLWithPath: request["weights_dir"]!.string!)
            let data = try Data(contentsOf: weights.appendingPathComponent("weights.safetensors"))
            let size = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
            let header = try WireJSON.decode(data.subdata(in: 8..<(8 + Int(size))))
            dtype = header["weight"]!["dtype"]!.string!
            guard request["dtype"]?.string == (dtype == "BF16" ? "bf16" : "fp16") else {
                return ["ok": false, "error": ["code": "fixture_dtype", "message": "Load dtype disagrees with actual half weights"]]
            }
            if FileManager.default.fileExists(atPath: weights.appendingPathComponent("arsenal-snapshot.json").path), request["dtype"] == "bf16" {
                let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"]!
                _ = try file(support + "/fixture-baseline-load.json", WireJSON.encode(["requested": .object(request), "actual_dtype": .string(dtype)]))
                return ["ok": false, "error": ["code": "fixture_stop", "message": "Intake fixture reached selected BF16 serving baseline"]]
            }
            return ["engine": "stub"]
        case "measurement_components":
            let output = request["output_dir"]!.string!, fixture = request["fixture_dir"]!.string!
            let keys: [String]
            if kind == "enhancement" {
                keys = ["encoder", "magnitude", "phase"] + (0..<30).flatMap { ["block-\($0)/time", "block-\($0)/frequency"] }
            } else {
                keys = (0..<2).flatMap { c in ["case-\(c)/embedding", "case-\(c)/scorer"] + (0..<28).map { "case-\(c)/layer-\($0)" } }
            }
            let value: Float = dtype == "F16" && !fp16Passes ? -1 : 1
            _ = try file(output + "/outputs.safetensors", Self.tensor(keys: keys, dtype: dtype, value: value))
            let traces = Dictionary(
                uniqueKeysWithValues: keys.map { ($0 + "/output", JSONValue.object(["shape": [1], "count": 1, "finite": true, "max_abs": 1, "clamp_count": 0])) })
            _ = try file(output + "/activations.json", WireJSON.encode(traces))
            if request["capture"] == true { _ = try file(fixture + "/inputs.safetensors", Self.tensor(keys: ["input"], dtype: "F16", value: 1)) }
            return [:]
        default: return nil
        }
    }

    static func tensor(keys: [String], dtype: String, value: Float) throws -> Data {
        var header: JSONObject = [:], payload = Data()
        for key in keys {
            let offset = payload.count
            var bits = (dtype == "BF16" ? UInt16(value.bitPattern >> 16) : Float16(value).bitPattern).littleEndian
            withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
            header[key] = ["dtype": .string(dtype), "shape": [1], "data_offsets": [.number(Double(offset)), .number(Double(payload.count))]]
        }
        let json = try WireJSON.encode(header)
        var length = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(json); data.append(payload)
        return data
    }
}
