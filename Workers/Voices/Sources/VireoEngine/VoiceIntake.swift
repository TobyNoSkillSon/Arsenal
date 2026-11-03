// SPDX-License-Identifier: MIT
import Foundation
import MLXAudioTTS
import VireoCore

extension MLXSpeech {
    public func measurementComponents(fixture: URL, output: URL, create: Bool, only: Set<String>? = nil, reference: URL? = nil) throws -> Int {
        let capture = try VoiceIntakeCapture(fixture: fixture, create: create, only: only, reference: reference)
        VoiceIntakeCapture.active = capture
        defer { VoiceIntakeCapture.active = nil }
        for index in capture.profiles.indices {
            capture.caseIndex = index
        if let breeze = model as? BreezeTTSModel { try breeze.intakeCapture(capture) }
        else if let qwen = model as? Qwen3TTSModel { try qwen.intakeCapture(capture) }
        else if let pocket = model as? PocketTTSModel { try pocket.intakeCapture(capture) }
        else if let kokoro = model as? KokoroModel { try kokoro.intakeCapture(capture) }
        else { throw EngineError("No saved-input capture adapter for this voice architecture") }
        }
        return try capture.finish(fixture: fixture, output: output)
    }
}
