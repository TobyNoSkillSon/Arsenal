import Foundation
enum PowersetDecoder {
    static func binarize(
        probs: [Float],
        onset: Float,
        offset: Float,
        frameDuration: Float
    ) -> [(startTime: Float, endTime: Float)] {
        var segments = [(startTime: Float, endTime: Float)]()
        var inSpeech = false
        var speechStart: Float = 0

        for (i, prob) in probs.enumerated() {
            let time = Float(i) * frameDuration

            if !inSpeech && prob >= onset {
                inSpeech = true
                speechStart = time
            } else if inSpeech && prob < offset {
                inSpeech = false
                segments.append((speechStart, time))
            }
        }

        if inSpeech {
            let endTime = Float(probs.count) * frameDuration
            segments.append((speechStart, endTime))
        }

        return segments
    }
}
