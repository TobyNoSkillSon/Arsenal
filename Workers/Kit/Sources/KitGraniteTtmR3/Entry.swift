// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitGraniteTtmR3 {
    public static let modelID = "granite-ttm-r3"
    public static func run() -> Never { Wire().serve() }
}
