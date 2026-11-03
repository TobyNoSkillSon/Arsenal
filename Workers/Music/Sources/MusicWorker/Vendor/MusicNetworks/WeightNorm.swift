//
//  WeightNorm.swift
//  MusicNetworks
//
//  Created by David Sherlock on 2026.
//
//  PyTorch weight norm folded into a plain weight, as both engines' VAEs
//  ship it (`weight_g`, `weight_v`).
//

import Foundation
import MLX

package enum WeightNorm {

    // `g · v / (‖v‖ + 1e-9)`, the norm taken over every axis but the first.
    package static func fold(g: MLXArray, v: MLXArray) -> MLXArray {
        let flat = v.reshaped(v.dim(0), -1)
        let norm = MLX.sqrt((flat * flat).sum(axis: 1)).reshaped(g.shape)
        return g * v / (norm + 1e-9)
    }
}
