//
//  DiffusionSampler.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  Upstream's `mlx_generate_diffusion` (acestep/models/mlx/dit_generate.py
//  at ca1e85f): seeded noise (`mx.random.key(seed)`), optional retake mix,
//  the timestep schedule, then per step a DiT velocity — doubled batch with
//  the null condition when guidance > 1 (APG, momentum, norm 2.5) — the
//  optional stabilisers, an Euler / Heun / SDE step, DCW, and the repaint
//  injection; after the loop the repaint edges are cross-faded.
//

import Foundation
import MLX
import MLXRandom

// The conditions one diffusion run sees.
struct DiffusionConditions {
    // `[B, L, 2048]` packed conditions.
    var encoder: MLXArray
    // `[B, T, 128]` source latents + chunk mask.
    var context: MLXArray
}

// Repaint inputs for the loop.
struct RepaintInputs {
    // `[B, T]` true where the model generates.
    var mask: [[Bool]]
    // `[B, T, 64]` the clean source latents.
    var clean: MLXArray
    var crossfadeFrames: Int
    var injectionRatio: Double
}

struct DiffusionSettings {
    var seeds: [UInt32]
    var retakeSeeds: [UInt32]?
    var retakeVariance: Double = 0
    var shift: Double = 1
    var timesteps: [Double]?
    var steps: Int = 8
    var guidance: Double = 1
    var guidanceStart: Double = 0
    var guidanceEnd: Double = 1
    var coverStrength: Double = 1
    var method: DiffusionMethod = .ode
    var sampler: SamplerKind = .euler
    var velocityNormThreshold: Double = 0
    var velocityEMA: Double = 0
    var dcw = true
    var dcwMode: DCWMode = .double
    var dcwScaler = 0.05
    var dcwHighScaler = 0.02
    // Noise to start from instead of the seeds' (parity tests).
    var noise: MLXArray?
    // Cover noise strength: start part-way along the schedule from the
    // source latents re-noised (upstream's PyTorch path; its MLX path ignores it).
    var startFrom: (latents: MLXArray, strength: Double)?
}

enum DiffusionSampler {

    // The seeded starting noise `[B, T, C]`, one `mx.random.key(seed)` draw per item.
    static func noise(seeds: [UInt32], frames: Int, channels: Int) -> MLXArray {
        if seeds.count == 1 {
            return MLXRandom.normal([1, frames, channels], key: MLXRandom.key(UInt64(seeds[0])))
        }
        return MLX.concatenated(seeds.map { MLXRandom.normal([1, frames, channels], key: MLXRandom.key(UInt64($0))) }, axis: 0)
    }

    // APG as upstream's MLX path writes it, projected along the time axis.
    static func apg(_ cond: MLXArray, _ uncond: MLXArray, scale: Double, running: inout MLXArray?) -> MLXArray {
        var diff = cond - uncond
        // swiftlint:disable:next shorthand_operator
        if let previous = running { diff = diff + previous }
        running = diff
        let norm = MLX.sqrt((diff * diff).sum(axis: 1, keepDims: true))
        // swiftlint:disable:next shorthand_operator
        diff = diff * MLX.minimum(MLX.ones(like: norm), 2.5 / (norm + 1e-8))
        let v1 = cond / (MLX.sqrt((cond * cond).sum(axis: 1, keepDims: true)) + 1e-8)
        let parallel = (diff * v1).sum(axis: 1, keepDims: true) * v1
        return cond + Float(scale - 1) * (diff - parallel)
    }

    static func run(
        decoder: DiTDecoder, conditions: DiffusionConditions, nonCover: DiffusionConditions?,
        nullCondition: MLXArray, frames: Int, settings: DiffusionSettings, repaint: RepaintInputs?,
        progress: ((Int, Int) -> Void)?, cancelled: () -> Bool
    ) throws(ACEStepError) -> MLXArray {
        let batch = settings.seeds.count
        let channels = decoder.configuration.audioAcousticHiddenDim
        let dtype = conditions.encoder.dtype
        var startNoise = settings.noise ?? noise(seeds: settings.seeds, frames: frames, channels: channels)
        if settings.retakeVariance > 0, let retake = settings.retakeSeeds {
            let other = noise(seeds: retake, frames: frames, channels: channels)
            let angle = settings.retakeVariance * Double.pi / 2
            startNoise = Float(cos(angle)) * startNoise + Float(sin(angle)) * other
        }
        startNoise = startNoise.asType(dtype)
        var schedule = Schedule.timesteps(shift: settings.shift, custom: settings.timesteps, steps: settings.steps)
        var initial = startNoise
        if let start = settings.startFrom, start.strength > 0 {
            let level = 1 - start.strength
            let nearest = schedule.min { abs($0 - level) < abs($1 - level) }!
            initial = Float(nearest) * startNoise + Float(1 - nearest) * start.latents.asType(dtype)
            schedule = Array(schedule[schedule.firstIndex(of: nearest)!...])
        }
        let coverSteps = Int(Double(schedule.count) * settings.coverStrength)
        let guided = settings.guidance > 1

        func prepare(_ c: DiffusionConditions) -> (CrossAttentionCache, MLXArray) {
            if guided {
                let nulls = MLX.broadcast(nullCondition.asType(dtype), to: c.encoder.shape)
                return (
                    decoder.crossCache(for: MLX.concatenated([c.encoder, nulls], axis: 0)),
                    MLX.concatenated([c.context, c.context], axis: 0)
                )
            }
            return (decoder.crossCache(for: c.encoder), c.context)
        }
        var (cache, context) = prepare(conditions)
        var switched = false
        var running: MLXArray?
        var previous: MLXArray?

        func velocity(_ x: MLXArray, _ t: Double) -> MLXArray {
            let input = guided ? MLX.concatenated([x, x], axis: 0) : x
            let tt = MLXArray(Array(repeating: Float(t), count: input.dim(0))).asType(dtype)
            var v = decoder(input, t: tt, tr: tt, context: context, cache: cache)
            if guided {
                let cond = v[..<batch], uncond = v[batch...]
                v =
                    (settings.guidanceStart <= t && t <= settings.guidanceEnd)
                    ? apg(cond, uncond, scale: settings.guidance, running: &running) : cond
            }
            return v
        }

        func stabilise(_ v: MLXArray, _ x: MLXArray, _ prior: MLXArray?) -> MLXArray {
            var v = v
            if settings.velocityNormThreshold > 0 {
                let vn = MLX.sqrt((v * v).sum(axes: [1, 2], keepDims: true))
                let xn = MLX.sqrt((x * x).sum(axes: [1, 2], keepDims: true)) + 1e-10
                // swiftlint:disable:next shorthand_operator
                v = v * MLX.minimum(MLX.ones(like: vn), (Float(settings.velocityNormThreshold) * xn) / (vn + 1e-10))
            }
            if settings.velocityEMA > 0, let prior {
                let f = Float(settings.velocityEMA)
                v = (1 - f) * v + f * prior
            }
            return v
        }

        var x = initial
        let maskArray = repaint.map { r in
            MLXArray(r.mask.flatMap { $0 }.map { $0 ? Int32(1) : 0 }, [r.mask.count, r.mask.first?.count ?? 0]) .> 0
        }
        for (index, t) in schedule.enumerated() {
            if cancelled() { throw .cancelled }
            if index >= coverSteps && !switched {
                switched = true
                if let nonCover { (cache, context) = prepare(nonCover) }
            }
            var v = velocity(x, t)
            v = stabilise(v, x, previous)
            let before = x
            let rawV = v
            if index == schedule.count - 1 {
                // swiftlint:disable:next shorthand_operator
                x = x - v * Float(t)
            } else {
                let next = schedule[index + 1]
                if settings.sampler == .heun && settings.method == .ode {
                    let dt = Float(t - next)
                    let predicted = x - v * dt
                    MLX.eval(predicted)
                    var v2 = velocity(predicted, next)
                    v2 = stabilise(v2, predicted, v)
                    let average = 0.5 * (v + v2)
                    // swiftlint:disable:next shorthand_operator
                    x = x - average * dt
                    v = average
                } else if settings.method == .sde {
                    let clean = x - v * Float(t)
                    let fresh = MLXRandom.normal(x.shape).asType(dtype)
                    x = Float(next) * fresh + Float(1 - next) * clean
                } else {
                    // swiftlint:disable:next shorthand_operator
                    x = x - v * Float(t - next)
                }
            }
            if settings.dcw {
                let denoised = before - rawV * Float(t)
                x = WaveletCorrection.apply(
                    x, denoised: denoised, t: t, mode: settings.dcwMode,
                    scaler: settings.dcwScaler, highScaler: settings.dcwHighScaler)
            }
            previous = v
            if let repaint, let maskArray {
                let cutoff = Int((repaint.injectionRatio * Double(schedule.count)).rounded(.toNearestOrEven))
                if index < cutoff {
                    let after = index < schedule.count - 1 ? schedule[index + 1] : 0
                    let z = Float(after) * startNoise + Float(1 - after) * repaint.clean
                    x = MLX.where(maskArray.expandedDimensions(axis: -1), x, z)
                }
            }
            MLX.eval(x)
            progress?(index + 1, schedule.count)
        }
        if let repaint, repaint.crossfadeFrames > 0 {
            x = boundaryBlend(x, clean: repaint.clean, mask: repaint.mask, frames: repaint.crossfadeFrames)
            MLX.eval(x)
        }
        return x
    }

    // Upstream's `_mlx_repaint_boundary_blend`: linear ramps either side of each repainted span.
    static func boundaryBlend(_ x: MLXArray, clean: MLXArray, mask: [[Bool]], frames: Int) -> MLXArray {
        var soft = mask.map { $0.map { $0 ? Float(1) : 0 } }
        for (b, row) in mask.enumerated() {
            guard row.contains(true), row.contains(false), let left = row.firstIndex(of: true),
                let last = row.lastIndex(of: true)
            else { continue }
            let right = last + 1
            let count = row.count
            let fs = max(left - frames, 0)
            if left - fs > 0 {
                let n = left - fs
                for i in 0..<n { soft[b][fs + i] = Float(i + 1) / Float(n + 1) }
            }
            let fe = min(right + frames, count)
            if fe - right > 0 {
                let n = fe - right
                for i in 0..<n { soft[b][right + i] = 1 - Float(i + 1) / Float(n + 1) }
            }
        }
        let m = MLXArray(soft.flatMap { $0 }, [soft.count, soft.first?.count ?? 0]).expandedDimensions(axis: -1).asType(x.dtype)
        return m * x + (1 - m) * clean
    }
}
