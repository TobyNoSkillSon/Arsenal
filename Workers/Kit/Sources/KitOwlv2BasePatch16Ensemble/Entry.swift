// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitOwlv2BasePatch16Ensemble {
    public static let modelID = "owlv2-base-patch16-ensemble"
    public static func run() -> Never { Wire().serve() }
}
