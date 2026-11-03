// SPDX-License-Identifier: MIT
// The bundled entry for this kit model: it replaces the folder runner's main.swift; Kit.swift, Wire.swift and the
// model files are the runner's own, unchanged.
public enum KitEomtCocoPanopticLarge640 {
    public static let modelID = "eomt-coco-panoptic-large-640"
    public static func run() -> Never { Wire().serve() }
}
