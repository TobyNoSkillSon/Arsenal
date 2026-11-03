// SPDX-License-Identifier: MIT
import Foundation
import GeometryWorker

// Lab-only layout export uses the exact loader map and never constructs an MLX array.
if ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1", CommandLine.arguments.count == 4, CommandLine.arguments[1] == "--header-map" {
    let source = URL(fileURLWithPath: CommandLine.arguments[2])
    guard let header = try JSONSerialization.jsonObject(with: Data(contentsOf: source)) as? [String: Any] else { throw CocoaError(.fileReadCorruptFile) }
    let moge = CommandLine.arguments[3] == "moge3"
    var tensors: [String: Any] = [:], ignored: [String] = []
    for (key, entry) in header where key != "__metadata__" {
        guard let name = GeometryWeights.mappedName(key, moge: moge) else { ignored.append(key); continue }
        guard let tensor = entry as? [String: Any], let shape = tensor["shape"] as? [Int] else { throw CocoaError(.fileReadCorruptFile) }
        var rule: [String: Any] = ["name": name]
        if let permutation = GeometryWeights.permutation(name, shape: shape, moge: moge) { rule["permutation"] = permutation }
        tensors[key] = rule
    }
    let data = try JSONSerialization.data(withJSONObject: ["ignore": ignored.sorted(), "tensors": tensors], options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
} else if ProcessInfo.processInfo.environment["ARSENAL_NATIVE_LAB"] == "1", CommandLine.arguments.count == 3, CommandLine.arguments[1] == "--trellis-spec" {
    let data = try JSONSerialization.data(withJSONObject: GeometryWorker.trellisShapeInventory(lab: CommandLine.arguments[2] == "lab"), options: [.prettyPrinted, .sortedKeys])
    FileHandle.standardOutput.write(data)
} else {
    GeometryWorker.run()
}
