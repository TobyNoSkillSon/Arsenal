// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitChronos2 {
    public static let modelID = "chronos-2"
    public static func run() -> Never { Wire().serve() }
}
