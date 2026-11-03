// SPDX-License-Identifier: MIT
import Foundation
import MLX
import ModelPrecision

extension BiRefNet {
    /// Save exact inputs at both Swin scales, the squeeze boundary and the decoder boundary.
    /// Every candidate replays these inputs independently, never its own upstream activations.
    public func intakeComponents(fixture: URL, output: URL, capture: Bool) throws {
        let recorder = ComponentCapture()
        var saved = try ComponentCapture.inputs(at: fixture, create: capture) {
            var values: [String: MLXArray] = [:]
            for (caseIndex, size) in [128, 64].enumerated() {
                let x = MLXArray((0..<(size * size * 3)).map { Float16(0.7 * sin(Double($0) * 0.019) + 0.3 * cos(Double($0) * 0.007)) }, [1,size,size,3])
                values["patch-\(caseIndex)"] = x
                let patch = bb.patchEmbed(x)
                var tokens = patch.reshaped([1,-1,patch.dim(3)]), h = patch.dim(1), w = patch.dim(2)
                for (i, layer) in bb.layers.enumerated() {
                    values["stage-\(caseIndex)-\(i)"] = tokens
                    let result = layer(tokens, H: h, W: w)
                    tokens = result.down; h = result.downH; w = result.downW
                }
                if caseIndex == 0 {
                    var features = forwardEnc(x)
                    values["squeeze"] = features[4]
                    for block in squeezeModule { features[4] = block(features[4]) }
                    for (i, feature) in features.enumerated() { values["decoder-\(i)"] = feature }
                }
            }
            return values
        }
        let keys = Set(["patch-0", "patch-1", "squeeze"] + (0..<5).map { "decoder-\($0)" } + (0..<2).flatMap { c in (0..<4).map { "stage-\(c)-\($0)" } })
        guard Set(saved.keys) == keys, saved.values.allSatisfy({ $0.dtype == .float16 }) else { throw LoadError.inventory }
        BiRefNetTrace.recorder = recorder
        defer { BiRefNetTrace.recorder = nil }
        var outputs: [String: MLXArray] = [:]
        func begin(_ key: String) { BiRefNetTrace.context = key; BiRefNetTrace.sequence = 0 }
        for c in 0..<2 {
            begin("patch-\(c)")
            outputs["patch-\(c)"] = bb.patchEmbed(saved["patch-\(c)"]!)
            var h = (c == 0 ? 128 : 64) / 4
            for i in 0..<4 {
                let key = "stage-\(c)-\(i)"; begin(key + "/out")
                let result = bb.layers[i](saved[key]!, H: h, W: h)
                outputs[key + "/out"] = result.out; outputs[key + "/down"] = result.down
                for (name, trace) in recorder.traces where name.hasPrefix(key + "/out/") {
                    recorder.traces[key + "/down/" + name.dropFirst((key + "/out/").count)] = trace
                }
                h = result.downH
            }
        }
        begin("squeeze")
        var squeeze = saved["squeeze"]!
        for block in squeezeModule { squeeze = block(squeeze) }
        outputs["squeeze"] = squeeze
        begin("decoder")
        outputs["decoder"] = decoder((0..<5).map { saved["decoder-\($0)"]! })
        guard Set(outputs.keys) == Self.intakeKeys, outputs.values.allSatisfy({ $0.dtype == .float16 }) else { throw LoadError.inventory }
        // Persist and enforce shape/count identities before any value comparison.
        let plan: [String: Any] = ["schema": 1, "resolution": 128, "scales": [128,64], "input_shapes": saved.mapValues(\.shape), "output_shapes": outputs.mapValues(\.shape)]
        if capture {
            try JSONSerialization.data(withJSONObject: plan, options: [.sortedKeys]).write(to: fixture.appendingPathComponent("fixture-plan.json"))
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: fixture.appendingPathComponent("fixture-plan.json").path)
        } else {
            let recorded = try JSONSerialization.jsonObject(with: Data(contentsOf: fixture.appendingPathComponent("fixture-plan.json"))) as! NSDictionary
            guard recorded == plan as NSDictionary else { throw LoadError.inventory }
        }
        try recorder.write(outputs, to: output)
        saved.removeAll()
    }
    public static var intakeKeys: Set<String> {
        Set(["patch-0", "patch-1", "squeeze", "decoder"] + (0..<2).flatMap { c in (0..<4).flatMap { ["stage-\(c)-\($0)/out", "stage-\(c)-\($0)/down"] } })
    }
}
