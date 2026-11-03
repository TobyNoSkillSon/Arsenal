// SPDX-License-Identifier: MIT
import IntegrationKit

public enum EmbedWorker {
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(["qwen3_5"], conversion: NativeConversion(recipe: "cast:embed-ovis-v2", eligible: TextRetrievalAdapter.eligible)) { OvisAdapter() },
            NativeRegistration(
                ["qwen2_5_omni_thinker"], conversion: NativeConversion(recipe: "cast:embed-lco-omni-v1", eligible: TextRetrievalAdapter.eligible)
            ) { LCOAdapter() },
            NativeRegistration(["qwen3"], conversion: NativeConversion(recipe: "cast:embed-qwen3-v2", eligible: TextRetrievalAdapter.eligible)) { TextRetrievalAdapter() },
            NativeRegistration(
                ["embedding_gemma2", "embedding_gemma2_text"], conversion: NativeConversion(recipe: "cast:embed-gemma2-v2", eligible: TextRetrievalAdapter.eligible)
            ) { EmbeddingGemma2Adapter() }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(family: "embed", kinds: ["embeddings"], operations: ["embed", "rerank"], registrations: registrations)
    }
}
