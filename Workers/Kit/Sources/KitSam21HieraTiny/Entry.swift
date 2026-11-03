// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitSam21HieraTiny {
    public static let modelID = "sam2.1-hiera-tiny"
    public static func run() -> Never { Wire().serve() }
}
