// SPDX-License-Identifier: MIT
import IntegrationKit
import MLXTrellis
public enum GeometryWorker {
    public static func trellisShapeInventory(lab: Bool) -> [String: [String: [Int]]] { TrellisShapeSpec(lab: lab).components }
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(["trellis2", "Trellis2ImageTo3DPipeline"], conversion: TrellisAdapter.baseConversion) {
                TrellisAdapter()
            },
            NativeRegistration(["idol"], conversion: IdolAdapter.conversion) {
                IdolAdapter()
            },
            NativeRegistration(["moge3"], conversion: NativeConversion(recipe: "cast:geometry-moge-v1", eligible: GeometryWeights.storageEligible(moge: true))) {
                DepthAdapter("moge3")
            },
            NativeRegistration(["da3small", "da3-small"], conversion: NativeConversion(recipe: "cast:geometry-da3-v1", eligible: GeometryWeights.storageEligible(moge: false))) {
                DepthAdapter("da3small")
            },
            NativeRegistration(["da3mono", "da3mono-large"], conversion: NativeConversion(recipe: "cast:geometry-da3-v1", eligible: GeometryWeights.storageEligible(moge: false))) {
                DepthAdapter("da3mono")
            }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(family: "geometry", kinds: ["geometry", "perception"], operations: ["estimate_depth", "image_to_3d"], registrations: registrations)
    }
}
