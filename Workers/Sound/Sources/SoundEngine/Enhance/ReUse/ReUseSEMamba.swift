// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import Foundation
import MLX
import MLXNN

// SAFETY: The non-Sendable MLX model is confined to MambaEnhancer.
extension SEMamba: @unchecked Sendable {}

class SEMamba: Module {
    let denseEncoder: ReUseDenseEncoder
    let tfMamba: [ReUseTFMambaBlock]
    let maskDecoder: ReUseMagDecoder
    let phaseDecoder: ReUsePhaseDecoder
    var intake: ReUseIntakeSession?

    init(numBlocks: Int = 30, mode: ReUseScanMode = .standard, kernels: ReUseKernels = .metal) {
        self.denseEncoder = ReUseDenseEncoder()
        self.tfMamba = (0..<numBlocks).map { _ in
            ReUseTFMambaBlock(hidFeature: 64, mode: mode, kernels: kernels)
        }
        self.maskDecoder = ReUseMagDecoder(hidFeature: 64, outputChannel: 1)
        self.phaseDecoder = ReUsePhaseDecoder(hidFeature: 64, outputChannel: 1)
        super.init()
    }

    func callAsFunction(_ mag: MLXArray, _ pha: MLXArray) -> (MLXArray, MLXArray) {
        // Convert `[B, F, T]` inputs to MLX's `[B, T, F, C]` layout.
        let mag = mag.transposed(0, 2, 1)
        let pha = pha.transposed(0, 2, 1)
        
        var x = MLX.stacked([mag, pha], axis: -1)
        
        let (_, t, f, _) = (x.shape[0], x.shape[1], x.shape[2], x.shape[3])
        
        // x: [B, T, F, C]
        // PyTorch F.pad(x, (0, 2, 0, 2)) on [B, 2, T, F] pads F by (0,2) and T by (0,2)
        // In MLX [B, T, F, 2], Axis 1 is T, Axis 2 is F.
        x = MLX.padded(x, widths: [IntOrPair(0), IntOrPair((0, 2)), IntOrPair((0, 2)), IntOrPair(0)])
        
        x = denseEncoder(intake?.input("encoder", x) ?? x)
        intake?.output("encoder", x)
        for (i, block) in tfMamba.enumerated() {
            x = block.intakeForward(x, session: intake, prefix: "block-\(i)/")
        }

        var denoisedMag = maskDecoder(intake?.input("magnitude", x) ?? x)
        var denoisedPha = phaseDecoder(intake?.input("phase", x) ?? x)
        
        // Remove model padding, then restore `[B, F, T]` outputs.
        denoisedMag = denoisedMag[0..., ..<t, ..<f, 0...]
        denoisedPha = denoisedPha[0..., ..<t, ..<f, 0...]
        
        denoisedMag = denoisedMag.squeezed(axis: -1).transposed(0, 2, 1)
        denoisedPha = denoisedPha.squeezed(axis: -1).transposed(0, 2, 1)
        
        intake?.output("magnitude", denoisedMag); intake?.output("phase", denoisedPha)
        return (denoisedMag, denoisedPha)
    }
}
