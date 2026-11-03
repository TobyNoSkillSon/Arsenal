// SPDX-License-Identifier: MIT
import IntegrationKit
import MLX
import QwenImageNative

/// The qwenimage family: Qwen-Image-2.1 and its Turbo checkpoint. Get normalizes the pinned source to the served half
/// (`cast:native-half-v1`: the base model's FP32 VAE becomes BF16). Both folders ship BF16 only; no quantized tier is
/// derived.
public enum QwenImageWorker {
    public static var registrations: [NativeRegistration] {
        [NativeRegistration([QwenImageAdapter.id]) { QwenImageAdapter() }]
    }
    /// The base format both folders serve (their [precision] declaration).
    public static let servingHalf: DType = .bfloat16
    public static func run() {
        NativeFamilyWorker.run(family: "qwenimage", kinds: ["images"], operations: ["generate", "edit"], registrations: registrations)
    }
}
