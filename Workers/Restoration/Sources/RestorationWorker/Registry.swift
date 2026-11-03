// SPDX-License-Identifier: MIT
import IntegrationKit

/// Owned by the restoration integrator. Register adapters here; no changes to the HTTP API, pins or native dispatcher.
public enum RestorationWorker {
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(["rife"], conversion: NativeConversion(recipe: "cast:restoration-rife", eligible: RIFEAdapter.eligible)) { RIFEAdapter() },
            NativeRegistration(["seedvr2"], conversion: NativeConversion(recipe: "native:seedvr2", eligible: SeedVR2Adapter.eligible)) { SeedVR2Adapter() },
            NativeRegistration(["ddcolor"]) { DDColorAdapter() },
            NativeRegistration(["lama"], conversion: LaMaAdapter.conversion) { LaMaAdapter() },
            NativeRegistration(["gimm_vfi_r"], conversion: GIMMAdapter.conversion) { GIMMAdapter() }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(
            family: "restoration", kinds: ["restoration"], operations: ["upscale_image", "upscale_video", "interpolate_video", "colorize", "inpaint"], registrations: registrations)
    }
}
