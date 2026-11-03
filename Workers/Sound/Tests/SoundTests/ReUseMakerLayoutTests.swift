// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import SoundEngine

/// RE-USE's Get reads NVIDIA's own PyTorch checkpoint and writes the port's canonical file. The names and layout below
/// are the ones the earlier MLX conversion used; every maker tensor must map to exactly one of them.
struct ReUseMakerLayoutTests {
    @Test func makerNamesBecomeThePortNames() throws {
        let cases = [
            "dense_encoder.dense_conv_1.0.weight": "denseEncoder.conv1.layers.0.weight",
            "dense_encoder.dense_conv_2.2.weight": "denseEncoder.conv2.layers.2.weight",
            "dense_encoder.dense_block.dense_block.3.1.bias": "denseEncoder.denseBlock.layers.3.layers.1.bias",
            "mask_decoder.up_conv1.0.conv.weight": "maskDecoder.upConv1.layers.0.conv.weight",
            "mask_decoder.final_conv.bias": "maskDecoder.finalConv.bias",
            "phase_decoder.phase_conv_i.weight": "phaseDecoder.phaseConvI.weight",
            "phase_decoder.phase_conv_r.bias": "phaseDecoder.phaseConvR.bias",
            "TSMamba.29.freq_mamba.backward_blocks.dt_proj.weight": "tfMamba.29.freqMamba.backward.dtProj.weight",
            "TSMamba.0.time_mamba.forward_blocks.A_log": "tfMamba.0.timeMamba.forward.A_log",
            "TSMamba.7.time_mamba.output_proj.bias": "tfMamba.7.timeMamba.outputProj.bias",
            "TSMamba.7.time_mamba.forward_blocks.x_proj.weight": "tfMamba.7.timeMamba.forward.xProj.weight"
        ]
        for (maker, port) in cases { #expect(try ReUseMakerLayout.rename(maker) == port, "\(maker)") }
        #expect(throws: (any Error).self) { try ReUseMakerLayout.rename("unknown.weight") }
        #expect(throws: (any Error).self) { try ReUseMakerLayout.names(["mask_decoder.final_conv.bias", "maskDecoder.finalConv.bias"]) }
    }

    @Test func convolutionsMoveToChannelsLast() {
        // (O, I, H, W) -> (O, H, W, I) and (O, I, K) -> (O, K, I); matrices and vectors keep their layout.
        #expect(ReUseMakerLayout.channelsLastAxes(rank: 4) == [0, 2, 3, 1])
        #expect(ReUseMakerLayout.channelsLastAxes(rank: 3) == [0, 2, 1])
        #expect(ReUseMakerLayout.channelsLastAxes(rank: 2) == nil && ReUseMakerLayout.channelsLastAxes(rank: 1) == nil)
    }
}
