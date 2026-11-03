// SPDX-License-Identifier: MIT
import VireoCore
public enum Engines {
    public static func loader(_ engine: String) throws -> ModelLoader {
        guard engine == "mlx-audio" else { throw EngineError("Unsupported speech engine") }
        return MLXSpeech.load
    }
}
