// SPDX-License-Identifier: MIT
import Foundation

/// pyannote/speaker-diarization-community-1 pipeline constants: config.yaml at the pinned mirror
/// revision plus the checkpoints' own specifications (pyannote.audio 4.0.0).
struct Community1Config: Sendable {
    static let sampleRate = 16_000
    /// Segmentation specifications: 10 s chunks, step 0.1 × duration, no warm-up, powerset of 3 speakers (max 2 at once).
    static let chunkSamples = 160_000
    static let localSpeakers = 3
    static let embeddingDimension = 256
    static let pldaDimension = 128
    /// PyanNet receptive field: 991 samples wide, one frame every 270 samples.
    static let frameDuration = 991.0 / 16_000
    static let frameStep = 270.0 / 16_000
    /// WeSpeaker ResNet34 fails below 400 samples (one 25 ms fbank frame): pyannote's min_num_samples.
    static let minimumEmbeddingSamples = 400
    /// config.yaml segmentation_batch_size / embedding_batch_size.
    static let segmentationBatch = 32
    /// Powerset classes in pyannote order: {}, {0}, {1}, {2}, {0,1}, {0,2}, {1,2}.
    static let powerset: [[Float]] = [[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], [1, 0, 1], [0, 1, 1]]

    /// config.yaml params.clustering: threshold 0.6, Fa 0.07, Fb 0.8; VBxClustering fixed values otherwise.
    var threshold = 0.6
    var fa = 0.07
    var fb = 0.8
    var maxIterations = 20
    var initialSmoothing = 7.0
    var convergenceEpsilon = 1e-4
    var minimumActiveRatio = 0.2
    var speakerPriorCutoff = 1e-7
    /// config.yaml params.segmentation.min_duration_off = 0.0 (no gap filling).
    var minimumDurationOff = 0.0

    static let `default` = Community1Config()
}

/// pyannote set_num_speakers: min = num ?? min ?? 1, max = num ?? max ?? ∞, equal bounds fix the count.
struct Community1SpeakerBounds: Sendable {
    let exact: Int?
    let minimum: Int
    let maximum: Int?

    init(num: Int?, min: Int?, max: Int?) throws {
        let lower = num ?? min ?? 1
        let upper = num ?? max
        if let upper, lower > upper { throw audioFailure("min_speakers must not exceed max_speakers.") }
        minimum = lower
        maximum = upper
        exact = upper == lower ? lower : nil
    }
}
