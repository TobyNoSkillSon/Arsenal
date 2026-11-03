//
//  Schedule.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  Upstream's `get_timestep_schedule` (acestep/models/mlx/dit_generate.py):
//  custom timesteps snapped to the 20 values the turbo model was distilled
//  on; otherwise a linear schedule of `steps` values from 1, shifted by
//  `s·t / (1 + (s−1)·t)`.
//

import Foundation

enum Schedule {

    static let validTimesteps: [Double] = [
        1.0, 0.9545454545454546, 0.9333333333333333, 0.9, 0.875,
        0.8571428571428571, 0.8333333333333334, 0.7692307692307693, 0.75,
        0.6666666666666666, 0.6428571428571429, 0.625, 0.5454545454545454,
        0.5, 0.4, 0.375, 0.3, 0.25, 0.2222222222222222, 0.125
    ]

    static let shiftTables: [Double: [Double]] = [
        1.0: [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125],
        2.0: [
            1.0, 0.9333333333333333, 0.8571428571428571, 0.7692307692307693,
            0.6666666666666666, 0.5454545454545454, 0.4, 0.2222222222222222
        ],
        3.0: [
            1.0, 0.9545454545454546, 0.9, 0.8333333333333334, 0.75,
            0.6428571428571429, 0.5, 0.3
        ]
    ]

    // The descending timesteps (no trailing 0).
    static func timesteps(shift: Double, custom: [Double]?, steps: Int?) -> [Double] {
        if var list = custom {
            while list.last == 0 { list.removeLast() }
            if !list.isEmpty {
                return list.prefix(20).map { t in validTimesteps.min { abs($0 - t) < abs($1 - t) }! }
            }
        }
        if let steps, steps > 0 {
            let raw = (0..<steps).map { 1.0 - Double($0) / Double(steps) }
            return shift == 1 ? raw : raw.map { shift * $0 / (1 + (shift - 1) * $0) }
        }
        let nearest = shiftTables.keys.min { abs($0 - shift) < abs($1 - shift) }!
        return shiftTables[nearest]!
    }
}
