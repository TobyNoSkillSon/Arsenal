// SPDX-License-Identifier: MIT
import IntegrationKit
import PerceptionSupport
import MLXTIPS
import MLXRFDETR
import MLXLocateAnything
import SAM31
public enum PerceptionWorker {
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(["tipsv2"], conversion: NativeConversion(recipe: "cast:tipsv2-affine-v1", eligible: PerceptionIO.eligible)) { NativeTIPS() },
            NativeRegistration(
                ["rf_detr"],
                conversion: NativeConversion(
                    recipe: "cast:rf-detr-nano-affine-v1",
                    eligible: { name, shape in PerceptionIO.eligible(name, shape) && !name.contains("query_feat") && !name.contains("refpoint_embed") })
            ) { NativeRFDETR() },
            NativeRegistration(["locateanything"], conversion: NativeLocateAnything.conversion(recipe: "cast:locateanything-affine-v1")) { NativeLocateAnything() },
            NativeRegistration(["sam3.1_video"], conversion: NativeSAM31.conversion) { NativeSAM31() }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(
            family: "perception", kinds: ["perception", "embeddings"],
            operations: ["detect", "ground_screen", "segment", "segment_session", "track_video", "video_session", "embed"], registrations: registrations)
    }
}
