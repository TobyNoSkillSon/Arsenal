// SPDX-License-Identifier: MIT
import IntegrationKit

public enum LanguageWorker {
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(
                ["qwen3_5", "qwen3_vl", "qwen2_5_vl", "lfm2_vl", "lfm2-vl", "gemma4", "hunyuan_v1_dense", "paddleocr_vl"],
                conversion: LanguageNative.conversion(recipe: "cast:language-half-g64-v3")
            ) { LanguageNative() },
            // v2 packed every eligible checkpoint matrix, including ones Load drops. For checkpoints that carry no dropped
            // matrix (UI-Venus 2 9B, UI-Venus 1.5 2B, TeleOCR) v2 and v3 write identical snapshots, so those folders keep
            // v2 and their installed tiers stay valid. Conversion only: no model type resolves here.
            NativeRegistration([], conversion: LanguageNative.conversion(recipe: "cast:language-half-g64-v2")) { LanguageNative() }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(
            family: "language", kinds: ["chat", "perception"],
            operations: ["chat", "count_tokens", "describe", "tag", "translate", "read_document", "extract", "ground_screen"], registrations: registrations)
    }
}
