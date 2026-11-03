// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitT0Gamma {
    public static let modelID = "t0-gamma"
    public static func run() -> Never { Wire().serve() }
}
