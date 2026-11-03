// SPDX-License-Identifier: MIT
import Foundation
@testable import RestorationWorker
import IntegrationKit
import MLX
import MLXNN
import SeedVR2Native
import ArsenalWire
import Testing

/// Frame counts through the causal VAE (upstream: 4n+1 frames in, 4n+1 frames out) and the spatial tiling bound.
@Suite(.serialized)
struct VAEFrameTests {
    private func vae() throws -> SeedVR2VAE {
        NativeRuntime.prepare()
        let vae = SeedVR2VAE(blockOut: [64, 64, 64, 64])
        try NativeParameters.initializeRandom(vae)
        vae.train(false)
        return vae
    }

    @Test(.heavy) func fiveFramesEncodeToTwoLatentsAndDecodeToFive() throws {
        let vae = try vae()
        for (frames, latents) in [(1, 1), (5, 2), (9, 3)] {
            let video = MLXRandom.uniform(Float(-1)..<Float(1), [1, 3, frames, 32, 32], key: MLXRandom.key(UInt64(frames))).asType(.float16)
            let latent = vae.encode(video)
            #expect(latent.shape == [1, 16, latents, 4, 4])
            let decoded = vae.decode(latent)
            eval(decoded)
            #expect(decoded.shape == [1, 3, frames, 32, 32])
            #expect(SeedVR2VAE.decodedFrames(latentFrames: latents) == frames)
            #expect(SeedVR2Adapter.latentFrames(frames) == latents)
        }
    }

    @Test(.heavy) func tiledEncodeAndDecodeKeepShapes() throws {
        let vae = try vae()
        let video = MLXRandom.uniform(Float(-1)..<Float(1), [1, 3, 5, 192, 160], key: MLXRandom.key(11)).asType(.float16)
        let whole = vae.encode(video), tiled = vae.encode(video, tile: 128)
        #expect(tiled.shape == whole.shape)
        let decoded = vae.decode(tiled, tile: 128)
        eval(decoded)
        #expect(decoded.shape == [1, 3, 5, 192, 160])
        #expect(MLX.all(MLX.isFinite(decoded)).item(Bool.self))
    }

    @Test func tilePlanCoversEveryPixelWithPositiveWeight() {
        #expect(SeedVR2VAE.tileStarts(24, tile: 16, overlap: 8) == [0, 8])
        #expect(SeedVR2VAE.tileStarts(40, tile: 16, overlap: 8) == [0, 8, 16, 24])
        #expect(SeedVR2VAE.tileStarts(10, tile: 16, overlap: 8) == [0])
        let ramp = SeedVR2VAE.edgeRamp(16, ramp: 4, leading: true, trailing: true)
        #expect(ramp.count == 16 && ramp.allSatisfy { $0 > 0 && $0 <= 1 })
        #expect(ramp[0] < ramp[3] && ramp[15] < ramp[12] && ramp[8] == 1)
    }

    @Test func chunkAndTileBudgets() {
        #expect(SeedVR2Adapter.chunkFrames(outputPixels: 1280 * 720) == 21)
        #expect(SeedVR2Adapter.chunkFrames(outputPixels: 1920 * 1080) == 9)
        #expect(SeedVR2Adapter.chunkFrames(outputPixels: 3840 * 2160) == 5)
        #expect(SeedVR2Adapter.vaeTile(frames: 1) == 1408)
        #expect(SeedVR2Adapter.vaeTile(frames: 5) == 640)
        #expect(SeedVR2Adapter.vaeTile(frames: 21) == 256)
        #expect(SeedVR2Adapter.noise(seed: 7, height: 720, width: 1280, frames: 21).shape == [1, 16, 6, 90, 160])
    }

    @Test(.heavy) func pixelsRoundToTheNearestLevel() throws {
        NativeRuntime.prepare()
        let pixels = try MediaBridge.pixels(MLXArray([Float(0.6), 0.999, 0.001], [1, 1, 1, 3]).asType(.float16))
        #expect(Array(pixels.bytes.prefix(3)) == [153, 255, 0])
    }

    /// Progress the service shows (Activity, tool windows, job records) reads step/total/label.
    @Test func progressEventsCarryStepTotalAndLabel() throws {
        var events: [JSONObject] = []
        let emit: (WireFrame) throws -> Void = { frame in if case .json(let event) = frame { events.append(event) } }
        try MediaBridge.progress(3, total: 2, label: "Upscaling", emit: emit)
        try MediaBridge.stage(2, of: 3, "Restoring", emit: emit)
        #expect(events[0]["step"]?.number == 3 && events[0]["total"]?.number == 3 && events[0]["label"]?.string == "Upscaling")
        #expect(events[0]["frames"]?.number == 3)
        #expect(events[1]["step"]?.number == 2 && events[1]["total"]?.number == 3 && events[1]["label"]?.string == "Restoring")
    }
}
