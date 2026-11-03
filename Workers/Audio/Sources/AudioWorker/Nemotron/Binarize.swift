import Foundation
enum SortformerDiarizer {
    static func binarize(
        probs: [Float],
        frameCount: Int,
        audioDuration: Float,
        config: SortformerConfig,
        thresholds: DiarizationConfig
    ) -> [DiarizedSegment] {
        guard frameCount > 0 else { return [] }
        let numSpeakers = config.maxSpeakers
        var allProbs = probs

        // Apply sigmoid if predictions are logits
        for i in 0..<allProbs.count {
            if allProbs[i] > 1.0 || allProbs[i] < 0.0 {
                allProbs[i] = 1.0 / (1.0 + exp(-allProbs[i]))
            }
        }

        let frameDuration =
            Float(config.predictionSubsamplingFactor * config.hopLength)
            / Float(config.sampleRate)
        var allSegments = [DiarizedSegment]()
        for spk in 0..<numSpeakers {
            var track = [Float](repeating: 0, count: frameCount)
            for f in 0..<frameCount {
                track[f] = allProbs[f * numSpeakers + spk]
            }

            let rawSegments = PowersetDecoder.binarize(
                probs: track,
                onset: thresholds.onset,
                offset: thresholds.offset,
                frameDuration: frameDuration
            )

            for seg in rawSegments {
                let duration = seg.endTime - seg.startTime
                guard duration >= thresholds.minSpeechDuration else { continue }
                allSegments.append(
                    DiarizedSegment(
                        startTime: seg.startTime,
                        endTime: min(seg.endTime, audioDuration),
                        speakerId: spk
                    ))
            }
        }

        allSegments.sort { $0.startTime < $1.startTime }
        let merged = DiarizationHelpers.mergeSegments(
            allSegments, minSilence: thresholds.minSilenceDuration)
        return DiarizationHelpers.compactSpeakerIds(merged)
    }

}
