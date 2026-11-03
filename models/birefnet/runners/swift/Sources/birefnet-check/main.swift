import Foundation
import MLX
import MLXBiRefNet
let args = CommandLine.arguments
if let lib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { GPU.metallib = URL(fileURLWithPath: lib) }
Memory.memoryLimit = 12 * 1024 * 1024 * 1024
Memory.cacheLimit = 256 * 1024 * 1024
let model = BiRefNet()
try model.loadWeights(url: URL(fileURLWithPath: args[1]))
try model.checkComponents(fixture: URL(fileURLWithPath: args[2]), output: URL(fileURLWithPath: args[3]))
print("20 GPU half component outputs saved")
