// SPDX-License-Identifier: MIT
import Foundation
import MLX
extension BiRefNet {
 public func checkComponents(fixture: URL, output: URL) throws {
  let saved = try MLX.loadArrays(url: fixture.appendingPathComponent("inputs.safetensors"))
  let data = try Data(contentsOf: fixture.appendingPathComponent("fixture-plan.json"))
  let plan = try JSONSerialization.jsonObject(with: data) as! [String: Any]
  let calls = plan["calls"] as! [[String: Any]]
  guard calls.count == 12 else { throw LoadError.inventory }
  var outputs: [String: MLXArray] = [:]
  for call in calls {
   let key = call["key"] as! String, name = call["module"] as! String
   let args = (call["args"] as! [String: Any])["tuple"] as! [[String: Any]]
   func tensor(_ arg: [String: Any]) -> MLXArray {
    let x = saved[arg["tensor"] as! String]!
    return x.ndim == 4 ? x.transposed(0,2,3,1) : x
   }
   let values: [MLXArray]
   if name == "bb.patch_embed" { values = [bb.patchEmbed(tensor(args[0]))] }
   else if name.hasPrefix("bb.layers.") {
    let i = Int(name.split(separator: ".").last!)!
    let v = bb.layers[i](tensor(args[0]), H: args[1]["scalar"] as! Int, W: args[2]["scalar"] as! Int)
    values = [v.out, v.down]
   } else if name == "squeeze_module" {
    var v = tensor(args[0]); for block in squeezeModule { v = block(v) }; values = [v]
   } else if name == "decoder" {
    values = [decoder((args[0]["tuple"] as! [[String: Any]]).map(tensor))]
   } else { throw LoadError.inventory }
   for (i,v) in values.enumerated() {
    let y = v.ndim == 4 ? v.transposed(0,3,1,2) : v
    guard y.dtype == .float16 else { throw LoadError.precision }
    eval(y); outputs["\(key)/out-\(i)"] = y
   }
  }
  guard outputs.count == 20 else { throw LoadError.inventory }
  try MLX.save(arrays: outputs, url: output)
 }
}
