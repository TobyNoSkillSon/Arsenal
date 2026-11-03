import Foundation

/// Shared helpers for diarization post-processing, used by both
/// PyannoteDiarizationPipeline and SortformerDiarizer.
enum DiarizationHelpers {

    /// Merge adjacent segments from the same speaker when the gap is below `minSilence`.
    ///
    /// Segments are grouped per-speaker, merged within each group, then sorted globally.
    static func mergeSegments(
        _ segments: [DiarizedSegment],
        minSilence: Float
    ) -> [DiarizedSegment] {
        guard !segments.isEmpty else { return [] }

        var bySpeaker = [Int: [DiarizedSegment]]()
        for seg in segments {
            bySpeaker[seg.speakerId, default: []].append(seg)
        }

        var merged = [DiarizedSegment]()
        for (spk, spkSegs) in bySpeaker {
            let sorted = spkSegs.sorted { $0.startTime < $1.startTime }
            var current = sorted[0]

            for i in 1..<sorted.count {
                let next = sorted[i]
                if next.startTime - current.endTime < minSilence {
                    current = DiarizedSegment(
                        startTime: current.startTime,
                        endTime: next.endTime,
                        speakerId: spk
                    )
                } else {
                    merged.append(current)
                    current = next
                }
            }
            merged.append(current)
        }

        merged.sort { $0.startTime < $1.startTime }
        return merged
    }

    /// Remap speaker IDs to a contiguous 0-based range in ascending original-ID order.
    static func compactSpeakerIds(_ segments: [DiarizedSegment]) -> [DiarizedSegment] {
        compactSpeakerIdsWithMapping(segments).segments
    }

    /// Remap speaker IDs and their centroid embeddings together.
    ///
    /// Clustering can produce a centroid that has no surviving segment after
    /// center-zone clipping and minimum-duration filtering. Compacting only the
    /// segments would then shift their IDs while leaving the centroid array in
    /// the old coordinate space. This helper keeps both outputs aligned.
    static func compactSpeakerIdsAndEmbeddings(
        _ segments: [DiarizedSegment],
        speakerEmbeddings: [[Float]],
        missingEmbeddingDimension: Int = 256
    ) -> (segments: [DiarizedSegment], speakerEmbeddings: [[Float]]) {
        let compacted = compactSpeakerIdsWithMapping(segments)
        let dimension =
            speakerEmbeddings.first(where: { !$0.isEmpty })?.count
            ?? missingEmbeddingDimension
        let zeroEmbedding = [Float](repeating: 0, count: max(0, dimension))
        let compactedEmbeddings = compacted.originalSpeakerIds.map { speakerId in
            speakerEmbeddings.indices.contains(speakerId)
                ? speakerEmbeddings[speakerId]
                : zeroEmbedding
        }
        return (compacted.segments, compactedEmbeddings)
    }

    private static func compactSpeakerIdsWithMapping(
        _ segments: [DiarizedSegment]
    ) -> (segments: [DiarizedSegment], originalSpeakerIds: [Int]) {
        let usedIds = Set(segments.map(\.speakerId)).sorted()
        let idMap = Dictionary(uniqueKeysWithValues: usedIds.enumerated().map { ($1, $0) })
        let compacted = segments.map {
            DiarizedSegment(
                startTime: $0.startTime,
                endTime: $0.endTime,
                speakerId: idMap[$0.speakerId] ?? $0.speakerId
            )
        }
        return (compacted, usedIds)
    }

    /// Resample audio via AVAudioConverter (delegates to AudioFileLoader).
    static func resample(_ audio: [Float], from sourceSR: Int, to targetSR: Int) -> [Float] {
        precondition(sourceSR == targetSR, "NativeMedia resamples before diarization")
        return audio
    }

}
