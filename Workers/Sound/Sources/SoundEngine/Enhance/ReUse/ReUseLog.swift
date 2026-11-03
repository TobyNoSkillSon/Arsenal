// SPDX-License-Identifier: Apache-2.0
// Vendored from faraday/SpeechLens Sources/Inference (Apache-2.0)
// @ d20e11bcafbc32435ef18a17ed56caf22fca6833; Vireo changes listed in CHANGES.md.

import OSLog

enum ReUseLog {
    private static let subsystem = "vireo.enhance.reuse"

    static let inference = Logger(subsystem: subsystem, category: "Inference")
    static let mamba = Logger(subsystem: subsystem, category: "ReUseMamba")
}
