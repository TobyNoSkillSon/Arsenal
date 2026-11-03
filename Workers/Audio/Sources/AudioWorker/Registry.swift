// SPDX-License-Identifier: MIT
import IntegrationKit

/// Owned by the audio integrator. Register adapters here; no changes to the HTTP API, pins or native dispatcher.
public enum AudioWorker {
    public static var registrations: [NativeRegistration] {
        [
            NativeRegistration(["pyannote_community_1"], conversion: Community1Conversion.tiers) { Community1Adapter() },
            NativeRegistration([], conversion: Community1Conversion.base) { Community1Adapter() },
            NativeRegistration(
                ["nemotron_diarization", "nemotron3_diarization", "nemotron_3_diarization"],
                conversion: NativeConversion(recipe: "audio-affine-v1:nemotron", eligible: audioEligible)
            ) {
                NemotronAdapter()
            },
            // Streaming Sortformer 4spk v2: the pinned .nemo (FP32) is converted once at Get into an FP16 snapshot; 16-bit only.
            NativeRegistration(["sortformer_streaming"], conversion: SortformerV2Conversion.base) { SortformerV2Adapter() },
            // DiariZen Large v2: FP32 PyTorch state dicts (segmentation + the WeSpeaker asset) converted once at Get into FP16; 16-bit only.
            NativeRegistration(["diarizen"], conversion: DiariZenConversion.base) { DiariZenAdapter() },
            NativeRegistration(["cohere_asr"], conversion: NativeConversion(recipe: "audio-affine-v1:cohere", eligible: audioEligible)) { CohereAdapter() },
            NativeRegistration(["granite_speech5_ctc"], conversion: NativeConversion(recipe: "audio-affine-v1:granite", eligible: audioEligible)) { GraniteAdapter() },
            NativeRegistration(["htdemucs_ft"], conversion: NativeConversion(recipe: "audio-affine-v1:demucs", eligible: audioEligible)) { DemucsAdapter() },
            // BS-RoFormer's PyTorch release (.ckpt + YAML) is converted once at Get into an FP16 snapshot; 16-bit only.
            NativeRegistration(["bs_roformer"], conversion: BSRoformerConversion.base) { BSRoformerAdapter() },
            // Kim's Mel-Band RoFormer: one FP32 state dict, converted once at Get into an FP16 snapshot; 16-bit only.
            NativeRegistration(["mel_band_roformer"], conversion: MelBandRoformerConversion.base) { MelBandRoformerAdapter() },
            // MOSS-Transcribe-Diarize: the BF16 release is cast to FP16 at Get; 16-bit only.
            NativeRegistration(["moss_transcribe_diarize"], conversion: NativeConversion(recipe: "cast:audio-moss-transcribe-diarize-v1", eligible: audioEligible)) {
                MossTranscribeAdapter()
            },
            // AudioSR basic: one FP32 PyTorch checkpoint, its three pipeline networks converted once at Get into FP16.
            NativeRegistration(["audiosr"], conversion: AudioSRConversion.base) { AudioSRAdapter() },
            NativeRegistration(["sam_audio"], conversion: SAMConversion.tiers) { SAMAdapter() },
            NativeRegistration([], conversion: SAMConversion.base) { SAMAdapter() },
            // SAM Audio Large from Meta's gated original (FP32 checkpoint.pt), converted once at Get into the same FP16 layout.
            NativeRegistration([], conversion: SAMConversion.official) { SAMAdapter() },
            // Smart Turn's FP32 release is cast to FP16 at Get (the fp32 source rule); int8/int4 derive from that snapshot.
            NativeRegistration([], conversion: NativeConversion(recipe: "cast:audio-smart-turn-v1", eligible: audioEligible)) { SmartTurnAdapter() },
            NativeRegistration(["smart_turn"], conversion: NativeConversion(recipe: "audio-affine-v1:smart-turn", eligible: audioEligible)) { SmartTurnAdapter() },
            NativeRegistration(["qwen3_asr"], conversion: NativeConversion(recipe: "audio-affine-v1:qwen3-asr", eligible: audioEligible)) { QwenASRAdapter() },
            // Vella's dictation runtime (VellaSTT): Parakeet TDT (16-bit only) and Qwen3-ASR (tower half at every tier).
            NativeRegistration(["parakeet_tdt"]) { ParakeetAdapter() },
            NativeRegistration(["qwen3_asr_vella"], conversion: NativeConversion(recipe: "audio-affine-v1:vella-qwen3-asr", eligible: vellaQwenEligible)) {
                VellaQwenASRAdapter()
            }
        ]
    }
    public static func run() {
        NativeFamilyWorker.run(
            family: "audio", kinds: ["audio", "enhancement"],
            operations: ["transcribe", "transcribe_session", "diarize", "diarize_session", "end_of_turn", "separate_audio", "enhance"],
            registrations: registrations)
    }
}
