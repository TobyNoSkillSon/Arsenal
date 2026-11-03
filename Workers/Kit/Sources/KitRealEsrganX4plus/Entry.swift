// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitRealEsrganX4plus {
    public static let modelID = "real-esrgan-x4plus"
    public static func run() -> Never { Wire().serve() }
}
