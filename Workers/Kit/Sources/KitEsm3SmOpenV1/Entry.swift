// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitEsm3SmOpenV1 {
    public static let modelID = "esm3-sm-open-v1"
    public static func run() -> Never { Wire().serve() }
}
