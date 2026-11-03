// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitVjepa2VitlFpc64256 {
    public static let modelID = "vjepa2-vitl-fpc64-256"
    public static func run() -> Never { Wire().serve() }
}
