import Foundation
public struct DiarizationConfig: Sendable {
    /// Onset threshold for speaker activity
    public var onset: Float
    /// Offset threshold for speaker activity
    public var offset: Float
    /// Minimum speech segment duration in seconds
    public var minSpeechDuration: Float
    /// Minimum silence duration between segments in seconds
    public var minSilenceDuration: Float
    /// Cosine distance threshold for merging speaker clusters (0.0-2.0).
    /// Lower = more merges (fewer speakers). Default 0.715.
    public var clusteringThreshold: Float

    public init(
        onset: Float = 0.5,
        offset: Float = 0.3,
        minSpeechDuration: Float = 0.3,
        minSilenceDuration: Float = 0.15,
        clusteringThreshold: Float = 0.715
    ) {
        self.onset = onset
        self.offset = offset
        self.minSpeechDuration = minSpeechDuration
        self.minSilenceDuration = minSilenceDuration
        self.clusteringThreshold = clusteringThreshold
    }

    public static let `default` = DiarizationConfig()

    /// Sortformer default: NeMo binarizes with a symmetric 0.5/0.5
    /// onset/offset, and the 0.3 hysteresis offset tuned for the pyannote
    /// pipeline measurably inflates false alarms on Sortformer activity.
    public static let sortformer = DiarizationConfig(offset: 0.5)
}

// MARK: - Result

/// Result of speaker diarization.
public struct DiarizationResult: Sendable {
    /// Diarized speech segments with speaker IDs
    public let segments: [DiarizedSegment]
    /// Number of distinct speakers found
    public let numSpeakers: Int
    /// Centroid embedding for each speaker (speaker ID → 256-dim embedding)
    public let speakerEmbeddings: [[Float]]

    public init(segments: [DiarizedSegment], numSpeakers: Int, speakerEmbeddings: [[Float]]) {
        self.segments = segments
        self.numSpeakers = numSpeakers
        self.speakerEmbeddings = speakerEmbeddings
    }
}

public struct DiarizedSegment: Sendable {
    /// Start time in seconds
    public let startTime: Float
    /// End time in seconds
    public let endTime: Float
    /// Speaker identifier (0-based)
    public let speakerId: Int

    public init(startTime: Float, endTime: Float, speakerId: Int) {
        self.startTime = startTime
        self.endTime = endTime
        self.speakerId = speakerId
    }

    /// Duration in seconds
    public var duration: Float { endTime - startTime }
}
