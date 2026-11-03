// SPDX-License-Identifier: MIT
// swiftlint:disable file_length
// Generated content (docs/MODELS.md, one section per catalog model): its length follows the catalog.
import Foundation

// docs/MODELS.md, embedded read-only for `arsenal://integration/MODELS.md`. Keep byte-for-byte parity; regenerate the
// source with scripts/generate-model-docs.py.
extension IntegrationReferences {
    static let models = ##"""
        # Arsenal models

        102 models, 370.2 GB if you get every one. Each downloads on its own, at the precision Arsenal starts it on.

        Generated from model folders by `scripts/generate-model-docs.py`. Figures are declarations/evidence, never estimates from parameter counts. An operation may be integrated before its weights or quality are qualified.

        | Model | What it does | Download | Weights licence |
        |---|---|---:|---|
        | [ACE-Step 1.5 Turbo](../models/ace-step-1.5/README.md) | Songs from lyrics and style | 10.1 GB | [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5/tree/19671f406d603126926c1b7e2adc169acbcade22) |
        | [astroBERT](../models/astrobert/README.md) | Embed astronomy text | 440 MB | [MIT](https://huggingface.co/adsabs/astroBERT/blob/6b9c25bb799152c5849a5b263ea71025fce548f0/README.md) |
        | [Audiobox Aesthetics](../models/audiobox-aesthetics/README.md) | Rate audio quality | 415 MB | [CC-BY-4.0](https://huggingface.co/facebook/audiobox-aesthetics/blob/9b1dd8e5df9af7216e836a98974fe3b82c56ded6/README.md) |
        | [AudioSR (basic)](../models/audiosr-basic/README.md) | Restore missing treble at 48 kHz | 6.2 GB | [Apache License 2.0](https://huggingface.co/haoheliu/audiosr_basic/blob/74a47f49061a1e788e968cc43ad45c0b6243f37d/README.md) |
        | [Big LaMa](../models/big-lama/README.md) | Photo inpainting | 204 MB | [Apache-2.0](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/LICENSE) |
        | [BiRefNet](../models/birefnet/README.md) | Background removal | 444 MB | [MIT](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4) |
        | [Chandra OCR 2](../models/chandra-ocr-2/README.md) | Read handwriting, forms and documents | 10.6 GB | [OpenRAIL-M (modified)](https://huggingface.co/datalab-to/chandra-ocr-2/tree/af93b47dba1b47b6640c86ccf487ed2260ab9a09) |
        | [Chronos-2](../models/chronos-2/README.md) | Forecast numeric time series | 478 MB | [Apache-2.0](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/README.md) |
        | [Chronos-Bolt Small](../models/chronos-bolt-small/README.md) | Forecast numeric time series | 191 MB | [Apache-2.0](https://huggingface.co/amazon/chronos-bolt-small/blob/772f3d25d38aec6d914c8949dab4462e2d46f5d8/README.md) |
        | [CLAP HTSAT Fused](../models/clap-htsat-fused/README.md) | Label sounds from your own words | 618 MB | [Apache-2.0](https://huggingface.co/laion/clap-htsat-fused/blob/365dea6ef167def6676140ed93bbc43f84dabb28/README.md) |
        | [d1 · 3B](../models/d1-3b/README.md) | Text and image decisions | 6.3 GB | [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE) |
        | [Depth Anything 3 Small](../models/da3-small/README.md) | Relative depth from multiple views | 137 MB | [Apache 2.0](https://huggingface.co/depth-anything/DA3-SMALL/tree/e08cab65ca0ec38e7826075418411ab90cab4da3) |
        | [Depth Anything 3 Metric Large](../models/da3metric-large/README.md) | Metric depth from one photo | 1.3 GB | [Apache-2.0](https://huggingface.co/depth-anything/DA3METRIC-LARGE/blob/4010e39f3634a45bc60553321fb49fb760bd594e/README.md) |
        | [DDColor ModelScope](../models/ddcolor-modelscope/README.md) | Photo colorization | 456 MB | [Apache-2.0](https://github.com/piddnad/DDColor/blob/2adb63f2656ac41cbdf7b894cddd94121a3faf13/LICENSE) |
        | [DeepFilterNet3](../models/deepfilternet3/README.md) | Fast speech denoising at 48 kHz | 9 MB | [MIT License](https://huggingface.co/mlx-community/DeepFilterNet-mlx/blob/220d5dfb7266352272d74c2a7d2025c59e07b391/README.md) |
        | [Depth Anything V2 Small](../models/depth-anything-v2-small/README.md) | Relative depth from one photo | 99 MB | [Apache-2.0](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/5426e4f0f36572d16453bbda7a8389317b1bef99/README.md) |
        | [DiariZen Large v2](../models/diarizen-large-v2/README.md) | Who spoke when, meetings, overlaps | 305 MB | [Creative Commons Attribution-NonCommercial 4.0](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2/blob/027b3e221b9d81dce2be794084f1dbd3ba2da403/README.md) |
        | [EdgeTAM On-Device Track Anything Model](../models/edgetam/README.md) | Segment an object by click or box | 28 MB | [Apache-2.0](https://huggingface.co/mlx-community/EdgeTAM-fp16/blob/e7770c15b913841cb657e6d2fbb33a6541c3bb32/README.md) |
        | [elephant-1](../models/elephant-1/README.md) | Text, image and video decisions | 18.8 GB | [MIT](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/README.md) |
        | [EmbeddingGemma 2](../models/embeddinggemma-2/README.md) | Search text, images, audio and video | 1.5 GB | [Apache-2.0](https://huggingface.co/google/embeddinggemma-2/blob/f9972ac22e696a105c482202646979e33e4b5a1e/README.md) |
        | [EoMT COCO Panoptic Large 640](../models/eomt-coco-panoptic-large-640/README.md) | Label every object and region | 1.3 GB | [MIT](https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/blob/dcd130bed9b1ebda7041fd660fddb16f905b9c3b/README.md) |
        | [ESM-2 650M](../models/esm2-t33-650m/README.md) | Embed protein sequences | 2.6 GB | [MIT](https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/08e4846e537177426273712802403f7ba8261b6c/README.md) |
        | [ESM-2 8M](../models/esm2-t6-8m/README.md) | Embed protein sequences | 31 MB | [MIT](https://huggingface.co/facebook/esm2_t6_8M_UR50D/blob/c731040fcd8d73dceaa04b0a8e6329b345b0f5df/README.md) |
        | [ESM3 Open Small (esm3-sm-open-v1)](../models/esm3-sm-open-v1/README.md) | Protein design and prediction | 5.4 GB | [MIT](https://huggingface.co/biohub/esm3-sm-open-v1/blob/47f0545b2b6daf26a93439a3cd610f4f7f3d5478/README.md) |
        | [Fish Audio Drama 3](../models/fish-drama-3/README.md) | Expressive speech, voice cloning | 23.8 GB | [Fish Audio Community License 1.0](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit/blob/cc639e27aba7edb73429d3e3806f471dcd79d316/LICENSE) |
        | [Florence-2 Base](../models/florence-2-base/README.md) | Caption a photo or read its text | 466 MB | [MIT](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/README.md) |
        | [Florence-2 Large](../models/florence-2-large/README.md) | Caption a photo or read its text | 1.6 GB | [MIT](https://huggingface.co/microsoft/Florence-2-large/blob/21a599d414c4d928c9032694c424fb94458e3594/README.md) |
        | [FLUX.2 Klein 4B](../models/flux2-klein-4b/README.md) | Image generation | 16.0 GB | [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md) |
        | [FLUX 3 Image Klein](../models/flux3-image-klein/README.md) | Image generation and editing | 20.3 GB | [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/LICENSE.md) · gated: accept Black Forest Labs's terms on Hugging Face |
        | [Geneformer V2 104M](../models/geneformer-v2-104m/README.md) | Embed single cells from genes | 418 MB | [Apache-2.0](https://huggingface.co/ctheodoris/Geneformer/blob/1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5/README.md) |
        | [GIMM-VFI-R](../models/gimm-vfi-r/README.md) | In-between frames at any moment | 79 MB | [S-Lab License 1.0](https://github.com/GSeanCDAT/GIMM-VFI/blob/dbc56449994a3c2e045d46fd46bed570239913f8/LICENSE) |
        | [GLM-OCR](../models/glm-ocr/README.md) | Read text from a document | 2.7 GB | [MIT](https://huggingface.co/zai-org/GLM-OCR/blob/2e85a62840ccac27daa451df36c736c4636b8628/README.md) |
        | [Granite Embedding Small English R2](../models/granite-embedding-small-english-r2/README.md) | Embed English text for search | 99 MB | [Apache-2.0](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/blob/2ab6fa8ea2d674564defd37171ae19079b864b33/README.md) |
        | [Granite TimeSeries TTM R3](../models/granite-ttm-r3/README.md) | Forecast numeric time series | 6 MB | [Apache-2.0](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3/blob/ea17cfd2e3edcaea21eb8dcecd18bf88971482fa/README.md) |
        | [Grounding DINO Base](../models/grounding-dino-base/README.md) | Find objects described in words | 934 MB | [Apache-2.0](https://huggingface.co/IDEA-Research/grounding-dino-base/blob/12bdfa3120f3e7ec7b434d90674b3396eccf88eb/README.md) |
        | [Harrier OSS v1 0.6B](../models/harrier-oss-v1-0.6b/README.md) | Create text search embeddings | 1.2 GB | [MIT](https://huggingface.co/microsoft/harrier-oss-v1-0.6b/tree/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee) |
        | [HTDemucs fine-tuned](../models/htdemucs-ft/README.md) | Separate music into four stems | 336 MB | [MIT](https://huggingface.co/mlx-community/demucs-mlx-fp16/tree/908d2d05cf3035bf4017fc0f52be64163689a61f) |
        | [IDOL](../models/idol/README.md) | One picture to a 3D splat | 6.2 GB | [Apache-2.0](https://huggingface.co/speridlabs/idol/blob/2731c2291fc65a797c5cb627a5e5526fc1490902/README.md) · gated: accept Meta's terms on Hugging Face |
        | [Index Translate 2B](../models/index-translate-2b/README.md) | Translate with glossary and style | 4.6 GB | [Apache License 2.0](https://huggingface.co/IndexTeam/Index-Translate-2B/tree/9bc08445315723eb9b77d1b93505915eb03c3f01) |
        | [Index Translate 9B](../models/index-translate-9b/README.md) | Translate with glossary and style | 19.3 GB | [Apache License 2.0](https://huggingface.co/IndexTeam/Index-Translate-9B/tree/4e06279e62352dab96bbeb6fa0579bca5e822d75) |
        | [Kokoro 82M](../models/kokoro-82m/README.md) | Fast English speech | 357 MB | [Apache-2.0](https://huggingface.co/mlx-community/Kokoro-82M-bf16/blob/a71e4d38b236d968966a2002c4c895dbd12b1c3c/README.md) |
        | [Laya · English](../models/laya-english/README.md) | English decisions | 846 MB | [Apache-2.0](https://huggingface.co/aac6fef/laya-mlx/blob/20aed815fc6acde75733882e7ec0e3f28aeb9717/LICENSE) |
        | [Laya · Multilingual](../models/laya-multilingual/README.md) | Decisions in 100+ languages | 678 MB | [Apache-2.0](https://huggingface.co/aac6fef/laya-multilingual-mlx/blob/f2b4faf51023039425946074e2cf1361d2db11d5/LICENSE) |
        | [Laya · Typed decisions](../models/laya-typed-decisions/README.md) | Typed workflow decisions | 846 MB | [Apache-2.0](https://huggingface.co/aac6fef/laya-typed-decisions-mlx/blob/f9e501c2080cc57c13d6887820329758f5351125/LICENSE) |
        | [LCO-Embedding Omni 7B](../models/lco-embedding-omni-7b/README.md) | Search text, images, audio and video | 17.9 GB | [Apache-2.0](https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B/blob/108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c/README.md) |
        | [Manga OCR Base](../models/manga-ocr-base/README.md) | Read Japanese manga text | 444 MB | [Apache-2.0](https://huggingface.co/kha-white/manga-ocr-base/blob/82c77fb84369e0fce0d6e8f0ad96fa82646d0346/README.md) |
        | [Mel-Band RoFormer Vocals (Kim)](../models/melband-roformer-kim/README.md) | Pull voice out of music | 913 MB | [MIT](https://huggingface.co/KimberleyJSN/melbandroformer/blob/ac9b0614ab3cd7f77219e18ba494dfd93956c348/README.md) |
        | [MiniMax Music 3](../models/minimax-music3/README.md) | Songs from lyrics and a description | 28.5 GB | [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/1b525ccb685ae809f33027c51c351eeb5833cf72/LICENSE) |
        | [Mitra-v2 Classifier](../models/mitra-classifier-2/README.md) | Classify table rows in context | 303 MB | [Apache-2.0](https://huggingface.co/autogluon/mitra-classifier-2/blob/edada0d20759c58ada8c8605c25f22f6e98ea5f0/README.md) |
        | [MoGe-3 ViT-L](../models/moge-3-vitl/README.md) | Metric depth and 3D from one image | 1.5 GB | [MIT](https://huggingface.co/mlx-community/moge-3-vitl-mlx-fp32/tree/f5ceca7512506731fb1c50307c313da76aac500f) |
        | [MoLFormer-XL Both 10%](../models/molformer-xl-both-10pct/README.md) | Embed molecules from SMILES | 187 MB | [Apache-2.0](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/blob/361063d0ad524ef77cf39b08469f6be770dc550f/README.md) |
        | [MOSS-Transcribe-Diarize 0.9B](../models/moss-transcribe-diarize/README.md) | Transcribe who said what, and when | 1.8 GB | [Apache License 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/10304d926740cf5a5cee05a268086e3265197910/README.md) |
        | [MossFormer2 SE 48K](../models/mossformer2-se-48k/README.md) | Speech denoising at 48 kHz | 222 MB | [Apache License 2.0](https://huggingface.co/alibabasglab/MossFormer2_SE_48K/blob/eff8c97925c8bec812af707814b3e5d777fd4503/README.md) |
        | [Nemotron 3 Diarization](../models/nemotron-3-diarization/README.md) | Identify who spoke when | 199 MB | [OpenMDW License Agreement 1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md) |
        | [NuExtract3](../models/nuextract3/README.md) | Fill JSON templates from documents | 9.1 GB | [Apache License 2.0](https://huggingface.co/numind/NuExtract3/tree/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1) |
        | [Ovis VL Embedding 2B](../models/ovis-vl-embedding-2b/README.md) | Find text, images and video by meaning | 4.4 GB | [Apache-2.0](https://huggingface.co/ATH-MaaS/Ovis-VL-Embedding-2B/tree/855b64a8246cf83b34730e4470ef66949e030659) |
        | [OWLv2 Base Patch16 Ensemble](../models/owlv2-base-patch16-ensemble/README.md) | Find objects named in words | 622 MB | [Apache-2.0](https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/cfd3195ba4ea9592eec887ded089f4c08eff231d/README.md) |
        | [Parakeet v3 Ultra](../models/parakeet-v3-ultra/README.md) | Speech transcription, 25 languages | 1.3 GB | [CC-BY-4.0](https://huggingface.co/selcukkubur/parakeet-ultra-mlx/tree/b554592c50b2a48471add2daa3d46fa9f00fef5e) |
        | [Pocket TTS](../models/pocket-tts/README.md) | Streaming speech, voice cloning | 239 MB | [CC-BY-4.0](https://huggingface.co/mlx-community/pocket-tts/blob/cbf71d5f6657bbc3f4bc02f85ee408261225bec7/README.md) |
        | [Pyannote Community-1](../models/pyannote-community-1/README.md) | Who spoke when, with overlaps | 33 MB | [Creative Commons Attribution 4.0](https://huggingface.co/pyannote-community/speaker-diarization-community-1/tree/8a527374977391da736e0daaef26855d949d9685) |
        | [Qwen3 ASR 0.6B](../models/qwen3-asr-0.6b/README.md) | Speech transcription, 30 languages | 1.6 GB | [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16/tree/eae2b51f96265328f1e7beced788adb0e4536f92) |
        | [Qwen3 ASR 1.7B](../models/qwen3-asr-1.7b/README.md) | Speech transcription, 30 languages | 4.1 GB | [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16/tree/e1f6c266914abc5a46e8756e02580f834a6cf8a7) |
        | [Qwen3 Embedding 0.6B](../models/qwen3-embedding-0.6b/README.md) | Multilingual text embeddings | 1.2 GB | [apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3) |
        | [Qwen3-Embedding-8B](../models/qwen3-embedding-8b/README.md) | Create text search embeddings | 15.1 GB | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af) |
        | [Qwen3 Reranker 8B](../models/qwen3-reranker-8b/README.md) | Rank search results by relevance | 16.4 GB | [Apache-2.0](https://huggingface.co/Qwen/Qwen3-Reranker-8B/tree/77d193c791ed757ca307ee72715aa132723da912) |
        | [Qwen3 TTS 12Hz 1.7B Base](../models/qwen3-tts-base/README.md) | Speech in 10 languages, voice cloning | 4.5 GB | [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16/blob/a6eb4f68e4b056f1215157bb696209bc82a6db48/README.md) |
        | [Qwen3 TTS 12Hz 1.7B VoiceDesign](../models/qwen3-tts-voicedesign/README.md) | Speech in 10 languages, voice design | 4.5 GB | [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/blob/7d3824abff87e49756bb0f83fb5411de75d160c4/README.md) |
        | [RE-USE](../models/re-use/README.md) | Speech cleanup | 39 MB | [OpenMDW-1.1](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/31f82f2efad1095034492f236e8dac9239ba713b/LICENSE) |
        | [Real-ESRGAN x4plus](../models/real-esrgan-x4plus/README.md) | Upscale a photo 4x | 67 MB | [BSD-3-Clause](https://huggingface.co/Comfy-Org/Real-ESRGAN/blob/5fd49b7b278836f48af63ecd314d0f98ab336105/README.md) |
        | [RF-DETR Large](../models/rf-detr-large/README.md) | Find 80 types of objects | 136 MB | [Apache-2.0](https://huggingface.co/Roboflow/rf-detr-large/blob/f62f7dd5252b61097cbace33886045816dadbde9/README.md) |
        | [RF-DETR Nano](../models/rf-detr-nano/README.md) | Find 80 types of objects | 122 MB | [Apache-2.0](https://huggingface.co/Roboflow/rf-detr-nano/blob/9f201577d9415f0378084748d27e68586e6b3600/README.md) |
        | [RIFE 4.25](../models/rife-4.25/README.md) | Video frame interpolation | 23 MB | [MIT](https://huggingface.co/mlx-community/RIFE-4.25/tree/6b650eaa5664aab280119511fe944861491aa15e) |
        | [SAM Audio Large](../models/sam-audio-large/README.md) | Isolate a sound by description | 15.8 GB | [SAM License](https://huggingface.co/facebook/sam-audio-large/blob/5f2cd3a9471a08c7282c06036be6893e18de8b70/LICENSE) · gated: accept Meta's terms on Hugging Face |
        | [SAM 2.1 Hiera Large](../models/sam2.1-hiera-large/README.md) | Cut out an object by click or box | 898 MB | [Apache-2.0](https://huggingface.co/facebook/sam2.1-hiera-large/blob/665f8e2ad61cf5f53d65644ff27c8ee525124610/README.md) |
        | [SAM 2.1 Hiera Tiny](../models/sam2.1-hiera-tiny/README.md) | Cut out an object by click or box | 156 MB | [Apache-2.0](https://huggingface.co/facebook/sam2.1-hiera-tiny/blob/de431c4043854a71d8101e17995dfe596bf101a5/README.md) |
        | [SAM 3.1](../models/sam3.1/README.md) | Select objects and track video | 3.5 GB | [SAM License](https://huggingface.co/facebook/sam3.1/blob/daa63191845a41281374e725f4c9e51c7a824460/LICENSE) · gated: accept Meta's terms on Hugging Face |
        | [SeedVR2 3B](../models/seedvr2-3b/README.md) | Image and video restoration | 8.4 GB | [Apache-2.0](https://huggingface.co/mlx-community/SeedVR2-3B-mlx/tree/46f851fe23b2a9fbefee6419b94e4fdf07b92965) |
        | [Smart Turn v3](../models/smart-turn-v3/README.md) | Detect the end of a spoken turn | 32 MB | [BSD-2-Clause](https://huggingface.co/mlx-community/smart-turn-v3/tree/74af1538ebace37b0fcc667a6a8a3d902d73eee0) |
        | [Streaming Sortformer Diarizer 4spk v2](../models/sortformer-4spk-v2/README.md) | Who spoke when, live, up to 4 | 471 MB | [Creative Commons Attribution 4.0](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/blob/84edd514b8ef68004c10086918cd62f2148cbd59/README.md) |
        | [Stable Audio 3 Medium](../models/stable-audio-3-medium/README.md) | Sound and music generation | 5.2 GB | [Stability AI Community](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/da6edc54ddba10bfd79a077102ded687f80e882b/LICENSE.md) |
        | [Stable Audio 3 Small SFX](../models/stable-audio-3-small-sfx/README.md) | Sound effects | 1.7 GB | [Stability AI Community](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/da6edc54ddba10bfd79a077102ded687f80e882b/LICENSE.md) |
        | [T0 Alpha](../models/t0-alpha/README.md) | Forecast numeric time series | 407 MB | [Apache-2.0](https://huggingface.co/theforecastingcompany/t0-alpha/blob/741927847fe02367624749bb9290b2f09f128e7d/README.md) |
        | [T0 Gamma](../models/t0-gamma/README.md) | Forecast series with covariates | 1.0 GB | [Apache-2.0](https://huggingface.co/theforecastingcompany/t0-gamma/blob/e08aee083778f1f32da4974c0a34276776a35c52/LICENSE) |
        | [TeleOCR](../models/teleocr/README.md) | Read text, tables, formulas and code | 2.8 GB | [Apache License 2.0](https://huggingface.co/XingChen-AGI/TeleOCR/tree/70a19ec6a3b036311af19ae5e762e650f41a5980) |
        | [TIGER DnR](../models/tiger-dnr/README.md) | Split dialogue, music and effects | 17 MB | [Apache-2.0](https://huggingface.co/JusperLee/TIGER-DnR/blob/b7a59560bbca10febbcd46fb01600f868e587f57/README.md) |
        | [TIGER Speech](../models/tiger-speech/README.md) | Separate two overlapping speakers | 3 MB | [Apache-2.0](https://huggingface.co/JusperLee/TIGER-speech/blob/f0340340b2d9bbf72074edf8c076dcab59a10ba2/README.md) |
        | [TimesFM 2.5 200M](../models/timesfm-2.5-200m/README.md) | Forecast numeric time series | 925 MB | [Apache-2.0](https://huggingface.co/google/timesfm-2.5-200m-pytorch/blob/1d952420fba87f3c6dee4f240de0f1a0fbc790e3/README.md) |
        | [Timestral 209M](../models/timestral-209m/README.md) | Forecast series with covariates | 837 MB | [Modified MIT License (Mistral AI)](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE) |
        | [TIPSv2 L/14](../models/tipsv2-l14/README.md) | Search photos with text or images | 2.0 GB | [Apache-2.0](https://huggingface.co/google/tipsv2-l14/blob/52847a71eb02a3082c370fbcd98cd799687bd4d0/README.md) |
        | [TRELLIS.2 4B](../models/trellis2-4b/README.md) | Image to textured 3D | 16.2 GB | [MIT](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d) · gated: accept Meta's terms on Hugging Face |
        | [TripoSR](../models/triposr/README.md) | Make a 3D model from one photo | 1.7 GB | [MIT](https://huggingface.co/stabilityai/TripoSR/blob/4e22858523f295f3c2b3852c39c90f399ea0e7d3/README.md) |
        | [UI Venus 1.5 2B](../models/ui-venus-1.5-2b/README.md) | Locate screen elements from instructions | 4.9 GB | [Apache License 2.0](https://huggingface.co/mlx-community/UI-Venus-1.5-2B-bf16/tree/d5286cd3d2580c331e2f57a976987744cd65e51d) |
        | [ULTRA 50g](../models/ultra-50g/README.md) | Predict missing knowledge-graph links | 1 MB | [MIT](https://huggingface.co/mgalkin/ultra_50g/blob/d46107d8415549e7c7386d26903ad11c9a8de4fd/README.md) |
        | [UniPASE](../models/unipase/README.md) | Universal speech enhancement | 2.2 GB | [Apache License 2.0](https://huggingface.co/Xiaobin-Rong/unipase/blob/f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0/README.md) |
        | [UVDoc](../models/uvdoc/README.md) | Flatten a photographed page | 32 MB | [Apache-2.0](https://huggingface.co/PaddlePaddle/UVDoc_safetensors/blob/bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4/README.md) |
        | [ViTPose Base Simple](../models/vitpose-base-simple/README.md) | Find body keypoints | 344 MB | [Apache-2.0](https://huggingface.co/usyd-community/vitpose-base-simple/blob/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45/README.md) |
        | [V-JEPA 2 ViT-L](../models/vjepa2-vitl-fpc64-256/README.md) | Embed video clips | 652 MB | [MIT](https://huggingface.co/mlx-community/V-JEPA2-vitl-fpc64-256/blob/8eec39e665ffce8ac09174ea3ab2f2d65940ef5f/README.md) |
        | [Voice Gender Classifier (ECAPA-TDNN)](../models/voice-gender-ecapa/README.md) | Tell male or female voice | 62 MB | [MIT](https://huggingface.co/JaesungHuh/voice-gender-classifier/blob/db1222153bd60337e900be22add7af180452adc0/README.md) |
        | [Von · 1.1](../models/von-1.1/README.md) | Option scoring, order-sensitive | 1.6 GB | [Apache-2.0](https://huggingface.co/wfzyx/von/blob/d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd/README.md) |
        | [Von · 1.2](../models/von-1.2/README.md) | Option scoring, order-independent | 1.6 GB | [Apache-2.0](https://huggingface.co/wfzyx/von/blob/5df8185a4f2327ad0a7cd117cc4f701ac557b9ae/README.md) |
        | [VoxCPM2](../models/voxcpm2/README.md) | Speech in 30 languages, voice cloning | 5.0 GB | [Apache License 2.0](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md) |
        | [YuE2 3B](../models/yue2-3b/README.md) | Songs from lyrics and style | 7.8 GB | [CC-BY-NC-4.0](https://huggingface.co/m-a-p/YuE2-3B/tree/8f24312f187763b9854adeec87e437e2b03bbff1) |

        ## ACE-Step 1.5 Turbo (`ace-step-1.5`)

        Songs from lyrics and style

        Generate songs with a planner, text and lyric encoders, diffusion transformer and audio VAE.

        Good for: Songs with vocals in 50 languages; Instrumental music; Covers of a supplied song; Repainting a section of a song; Timbre from a reference recording.
        Limits: Turbo model: no stem extraction, Lego or completion tasks; Trained FP16 and quantized output quality awaits qualification; Song files count against the 4 GiB per-job artifact limit, checked before generation; Generation produces complete files with progress, not live PCM.
        Parameters: 5.01B.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/ACE-Step/Ace-Step1.5/tree/19671f406d603126926c1b7e2adc169acbcade22).
        Tools: sound.

        - `generate_music`: `POST /arsenal/v1/models/ace-step-1.5/ops/generate_music`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 10091986191 | 10091986191 | 15.032 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~5361367663 | 9.700 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~2838371116 | 7.252 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/ace-step-1.5/README.md)

        ## astroBERT (`astrobert`)

        Embed astronomy text

        Turns astronomy and astrophysics text into 768-number vectors, so similar passages land close together.

        Good for: Grouping, clustering or classifying astronomy abstracts and paper passages by topic; Features for a small downstream classifier on astrophysics text (the maker's own use).
        Limits: Not trained for sentence similarity or search: its vectors cluster by topic and vocabulary, so unrelated texts still score around 0.5-0.7 cosine. For question-to-passage search a retrieval embedder does better; English astronomy text only; it does not handle Polish or other languages; 512 tokens per text (about 350 words); longer texts are refused unless truncate is on; No instruction, no shorter dimensions; text input only. The masked-word, NER and SciX classifier heads on the card are not served; Parity: matches the maker's transformers code on MPS fp16 at cosine 0.99999 (log/003-parity.md).
        Parameters: 109.7M.
        Languages: en.
        Licence: [MIT](https://huggingface.co/adsabs/astroBERT/blob/6b9c25bb799152c5849a5b263ea71025fce548f0/README.md).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/astrobert/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 439737407 | ~219868703 | 0.387 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/astrobert/README.md)

        ## Audiobox Aesthetics (`audiobox-aesthetics`)

        Rate audio quality

        Rates a speech, music or sound clip on four axes: enjoyment, usefulness, production complexity and production quality, each about 1 to 10.

        Good for: Ranking takes, renders or recordings of the same kind by production quality (PQ); Filtering a sound or music collection for clean, enjoyable clips before training or publishing; Spotting noisy, clipped or thin recordings without a clean reference.
        Limits: Scores are predictions of human ratings, not measurements; compare clips of the same kind rather than reading one number alone; Clips are scored in 10-second windows and averaged by length; a long file with one bad stretch gets an average, not the low point; Very short clips (under about a second) mostly see silence padding and score less reliably; Files not at 16 kHz are resampled with macOS's converter rather than the maker's torchaudio one; on a 1.6 s chime this moved scores by up to 0.1; It does not read language, so Polish speech is rated like any other speech; its training speech was mostly English.
        Parameters: 103.9M.
        Languages: mul.
        Licence: [CC-BY-4.0](https://huggingface.co/facebook/audiobox-aesthetics/blob/9b1dd8e5df9af7216e836a98974fe3b82c56ded6/README.md).
        Tools: the generic window built from its operations.

        - `score_quality`: `POST /arsenal/v1/models/audiobox-aesthetics/ops/score_quality`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 415473484 | ~207736742 | unknown |

        [Model documentation](../models/audiobox-aesthetics/README.md)

        ## AudioSR (basic) (`audiosr-basic`)

        Restore missing treble at 48 kHz

        Rebuild missing high frequencies of band-limited audio at 48 kHz using a native Swift MLX worker.

        Good for: Old recordings, phone or low-rate audio whose top end is missing; Speech, music and effects alike.
        Limits: Mono 48 kHz output; stereo input is mixed down; Up to 60 seconds per clip, processed whole as the reference does (about 13 GB of GPU memory at 60 s); Measured on a 1916 record: it adds a noise-like band above about 10 kHz, so listen before keeping the result; A diffusion model: each seed gives a different high band.
        Parameters: 672M.
        Languages: und.
        Licence: [Apache License 2.0](https://huggingface.co/haoheliu/audiosr_basic/blob/74a47f49061a1e788e968cc43ad45c0b6243f37d/README.md).
        Tools: the generic window built from its operations.

        - `enhance`: `POST /arsenal/v1/models/audiosr-basic/ops/enhance`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 6178189563 | ~3089094781 | 3.243 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/audiosr-basic/README.md)

        ## Big LaMa (`big-lama`)

        Photo inpainting

        Fill the marked part of a photo with what plausibly lies behind it.

        Good for: Removing objects, people, text and scratches; Large holes over repeating backgrounds.
        Limits: The mask must be the photo's size; every pixel that is not black is filled; Fills invent plausible texture: lettering, faces and fine structure inside the hole are not recovered; Checked against the makers' FP32 graph at 1024x576 (43 dB inside the hole); at 3840x2160 the FP16 fill drifts in tint (31.6 dB, mean 5.5 levels); Up to 8.3 megapixels per photo (3840×2160); larger photos are refused, so downscale them first.
        Parameters: 51M.
        Languages: und.
        Licence: [Apache-2.0](https://github.com/advimman/lama/blob/786f5936b27fb3dacd2b1ad799e4de968ea697e7/LICENSE).
        Tools: the generic window built from its operations.

        - `inpaint`: `POST /arsenal/v1/models/big-lama/ops/inpaint`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 204316668 | ~102158334 | 2.072 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/big-lama/README.md)

        ## BiRefNet (`birefnet`)

        Background removal

        Remove an image's background; returns a transparent PNG and its alpha mask.

        Good for: image-segmentation; background-removal.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 220M.
        Languages: und.
        Licence: [MIT](https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4).
        Tools: background.

        - `remove_background`: `POST /arsenal/v1/models/birefnet/ops/remove_background`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 444474001 | 444474001 | 5.761 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/birefnet/README.md)

        ## Chandra OCR 2 (`chandra-ocr-2`)

        Read handwriting, forms and documents

        Read handwriting, forms and documents.

        Good for: Read handwriting, forms and documents.
        Limits: Checked live on 20 Oct at FP16, int8 and int4: a text page and a table read exactly; Page Markdown follows the maker's parse_markdown and blocks follow parse_layout; Image and Figure blocks keep their region and description but no extracted image file; Modified OpenRAIL-M: free for research, personal use and startups under US$2M in revenue or funding; no use that competes with Datalab's API.
        Parameters: 4.5B.
        Languages: mul.
        Licence: [OpenRAIL-M (modified)](https://huggingface.co/datalab-to/chandra-ocr-2/tree/af93b47dba1b47b6640c86ccf487ed2260ab9a09). Free for research, personal use and startups under US$2M in funding or revenue; not for uses that compete with Datalab's API. Broader commercial use needs a licence from Datalab.
        Tools: the generic window built from its operations.

        - `read_document`: `POST /arsenal/v1/models/chandra-ocr-2/ops/read_document`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 10611238433 | 10611238433 | 20.401 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/chandra-ocr-2/README.md)

        ## Chronos-2 (`chronos-2`)

        Forecast numeric time series

        Forecasts the next values of one numeric series as a range of likely outcomes.

        Good for: Probabilistic zero-shot forecasts of one numeric series, such as sales, load or counts, up to 1024 steps in one pass; Long histories: it reads up to the last 8192 values; Wide uncertainty bands: 21 quantile levels from 0.01 to 0.99; Seasonal series: on monthly airline passengers it continues the summer peak and winter dip.
        Limits: Univariate here: one series in; the maker's multivariate, covariate and cross-learning modes are not ported; Horizon at most 1024 steps (the model's one-pass limit); the maker's long-horizon unrolling beyond that is not ported; Uses only the last 8192 values; missing values are not accepted; Returns only the trained quantile levels 0.01, 0.05, 0.1 to 0.9 in steps of 0.05, 0.95 and 0.99; The outer quantiles (0.01, 0.99) grow very wide far out on long horizons; Reads and writes no language, so the Polish question does not apply; Parity with the maker code (chronos-forecasting, MPS fp16): cosine 0.99999997, max abs diff 0.89 on forecasts up to 804 (airline series, 24 steps); cosine 0.9999965 over 1024 steps.
        Parameters: 119.5M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/amazon/chronos-2/blob/29ec3766d36d6f73f0696f85560a422f50e8498c/README.md).
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/chronos-2/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 477931539 | ~238965769 | unknown |

        [Model documentation](../models/chronos-2/README.md)

        ## Chronos-Bolt Small (`chronos-bolt-small`)

        Forecast numeric time series

        Forecasts the next values of one numeric series as a range of likely outcomes.

        Good for: Probabilistic zero-shot forecasts of one numeric series, such as sales, load or counts, 64 steps in one pass; Long histories: it reads up to the last 2048 values; Seasonal series: on monthly airline passengers it continues the summer peak and winter dip.
        Limits: Univariate here: one series in; covariates are an AutoGluon feature and are not ported; Past 64 steps it uses the maker's rollout heuristic (each quantile fed back, 81 forecasts reduced to 9), which the maker says may degrade quality; horizon at most 1024; Uses only the last 2048 values; Returns only the trained quantile levels 0.1 to 0.9 in steps of 0.1; Amazon now points to Chronos-2 as the newer model; Reads and writes no language, so the Polish question does not apply; Parity with the maker code (chronos-forecasting, MPS fp16): cosine 1.0000000, max abs diff 0.47 on forecasts up to 742 (airline series, 24 steps) and 0.81 on a 100-step rollout.
        Parameters: 47.7M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/amazon/chronos-bolt-small/blob/772f3d25d38aec6d914c8949dab4462e2d46f5d8/README.md).
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/chronos-bolt-small/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 190889945 | ~95444972 | unknown |

        [Model documentation](../models/chronos-bolt-small/README.md)

        ## CLAP HTSAT Fused (`clap-htsat-fused`)

        Label sounds from your own words

        Labels a sound or music clip with whichever of your own text descriptions fits it best, with no training.

        Good for: Zero-shot labelling of sound effects, environmental sounds and music against any list of descriptions; Sorting or searching a sample library by describing the sound (with return_embedding); Telling speech from music from noise before choosing a heavier model.
        Limits: Labels are scored against each other: the probabilities always sum to 1 over the labels you give, so add a catch-all label when none might fit; Very short clips are repeated to fill 10 seconds and get weaker, less certain labels (0.3 s of a chime came out as music, not a bell); Clips over 10 seconds are summarised from a shrunken whole plus three fixed 10-second crops (start, middle and end thirds); the maker picks those crops at random; English labels only: the text encoder is English RoBERTa, so Polish or other-language labels score poorly. It does not transcribe speech; Files not at 48 kHz are resampled with macOS's converter; the card leaves resampling to the user.
        Parameters: 153.6M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/laion/clap-htsat-fused/blob/365dea6ef167def6676140ed93bbc43f84dabb28/README.md).
        Tools: the generic window built from its operations.

        - `label`: `POST /arsenal/v1/models/clap-htsat-fused/ops/label`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 617866170 | ~308933085 | unknown |

        [Model documentation](../models/clap-htsat-fused/README.md)

        ## d1 · 3B (`d1-3b`)

        Text and image decisions

        Single-pass text, JSON and image decisions; 32,768-token context.

        Good for: yes-no; choice; score; visual-inspection.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 3.12B.
        Languages: ar, zh, en, fr, de, hi, id, it, ja, ko, pl, pt, ru, es, th, vi.
        Licence: [LFM Open License v1.0](https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/LICENSE). Commercial use has a US$10 million annual-revenue threshold (legal entity, including affiliates); above it, a separate Liquid AI licence is required. Redistribution must retain the licence and attribution and identify modifications. See the pinned LICENSE for boundary wording.
        Tools: the generic window built from its operations.

        - `decide`: `POST /arsenal/v1/models/d1-3b/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/d1-3b/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 6264991368 | 6264991368 | 5.896 (m5-max, 2026-10-15, quickCheck) |
        | int8 | made from the 16-bit download | ~3328276664 | 3.493 (m5-max, 2026-10-15, quickCheck) |
        | int4 | made from the 16-bit download | ~1762028822 | 2.153 (m5-max, 2026-10-15, quickCheck) |

        [Model documentation](../models/d1-3b/README.md)

        ## Depth Anything 3 Small (`da3-small`)

        Relative depth from multiple views

        Relative depth from multiple views.

        Good for: Scene structure without physical scale.
        Limits: Checked live at FP16 on 19 Oct: in a living-room photo, near and far were ordered correctly; At most eight views with matching processed dimensions; learned camera decoder not ported.
        Parameters: 34M.
        Languages: und.
        Licence: [Apache 2.0](https://huggingface.co/depth-anything/DA3-SMALL/tree/e08cab65ca0ec38e7826075418411ab90cab4da3).
        Tools: the generic window built from its operations.

        - `estimate_depth`: `POST /arsenal/v1/models/da3-small/ops/estimate_depth`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 137250142 | ~68625071 | 0.537 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/da3-small/README.md)

        ## Depth Anything 3 Metric Large (`da3metric-large`)

        Metric depth from one photo

        Estimates how far everything in one photo is, in metres when the focal length is known.

        Good for: Distances in metres from one photo with a known focal length or EXIF; Sky-aware depth for outdoor photos; Relative depth when no focal length is known.
        Limits: Metres need a focal length: focal_length in the photo's pixels, or EXIF 35 mm equivalent; otherwise the map is relative; One photo per request; no camera poses, intrinsics or points; The depth map comes at the processed size (504 pixels on the long side), not the photo's size; Sky pixels take the 99th percentile of the other depths over every pixel; the maker samples randomly, so sky values can differ by a few percent; Parity with the maker's code on MPS FP16: depth cosine 0.99968 or better on two photos.
        Parameters: 334.2M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/depth-anything/DA3METRIC-LARGE/blob/4010e39f3634a45bc60553321fb49fb760bd594e/README.md).
        Tools: the generic window built from its operations.

        - `estimate_depth`: `POST /arsenal/v1/models/da3metric-large/ops/estimate_depth`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1336735295 | ~668367647 | 2.223 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/da3metric-large/README.md)

        ## DDColor ModelScope (`ddcolor-modelscope`)

        Photo colorization

        Colour a black-and-white photo, keeping its own detail and brightness.

        Good for: Black-and-white photos and film frames; Old portraits and street scenes.
        Limits: Colour comes from a learned prior: skin, clothing and paint colours are plausible, not recovered; 16 megapixels per photo.
        Parameters: 0.23B.
        Languages: und.
        Licence: [Apache-2.0](https://github.com/piddnad/DDColor/blob/2adb63f2656ac41cbdf7b894cddd94121a3faf13/LICENSE).
        Tools: the generic window built from its operations.

        - `colorize`: `POST /arsenal/v1/models/ddcolor-modelscope/ops/colorize`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 455830765 | 455830765 | 1.417 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/ddcolor-modelscope/README.md)

        ## DeepFilterNet3 (`deepfilternet3`)

        Fast speech denoising at 48 kHz

        Remove background noise from speech at 48 kHz in real time using a native Swift MLX worker.

        Good for: Removing steady and changing background noise from calls, voice notes, meetings and podcasts; Full-band 48 kHz speech; Long recordings, up to 10 minutes per clip, with a 2.2M-parameter model; Keeping some ambience: the attenuation limit caps how much noise is removed.
        Limits: Denoising only: no reverb removal, clipping repair or bandwidth extension; Speech only; music under the noise is suppressed with it; Up to 10 minutes per clip. Each channel is cleaned in one pass over the whole file, so memory grows with length; the peak for a 10-minute file is not measured (estimated at a few GB); Arsenal cleans a whole file and then returns it; the maker's live microphone mode (LADSPA plugin) is not offered; The attenuation limit keeps some noise on purpose; at 0 (the default) the model removes as much as it can.
        Parameters: 2.2M.
        Languages: und.
        Licence: [MIT License](https://huggingface.co/mlx-community/DeepFilterNet-mlx/blob/220d5dfb7266352272d74c2a7d2025c59e07b391/README.md).
        Tools: the generic window built from its operations.

        - `enhance`: `POST /arsenal/v1/models/deepfilternet3/ops/enhance`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 8683568 | ~4341784 | unknown |

        [Model documentation](../models/deepfilternet3/README.md)

        ## Depth Anything V2 Small (`depth-anything-v2-small`)

        Relative depth from one photo

        Estimates how near or far everything in a photo is, as a relative depth map.

        Good for: Fast relative depth for a single photo, indoors or outdoors; Sharp object edges in the depth map; Depth cues for masking, blurring or 3D effects.
        Limits: Relative inverse depth only: larger values are nearer, with no unit and no metric scale; One image per request; no camera intrinsics, poses or point maps; The image is resized to about 518 pixels on its fitted side, so very small detail is lost; the map is returned at the photo's size; Parity with the maker's transformers code on MPS FP16: cosine 0.99996 on a fixed photo; edge pixels can differ by up to 0.2 of the range.
        Parameters: 24.8M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/blob/5426e4f0f36572d16453bbda7a8389317b1bef99/README.md).
        Tools: the generic window built from its operations.

        - `estimate_depth`: `POST /arsenal/v1/models/depth-anything-v2-small/ops/estimate_depth`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 99175385 | ~49587692 | 0.526 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/depth-anything-v2-small/README.md)

        ## DiariZen Large v2 (`diarizen-large-v2`)

        Who spoke when, meetings, overlaps

        Find who spoke when in meetings and far-field recordings, with up to four overlapping speakers, using a native Swift MLX worker.

        Good for: Meetings and far-field room recordings; Overlapping speech, up to four speakers at once; Recordings with many speakers (the number is found automatically); Exclusive one-speaker-at-a-time turns for aligning transcripts; RTTM export.
        Limits: At most ten minutes per recording; At most four speakers active at the same moment within a 16-second window; The number of speakers is always found automatically; the maker's pipeline takes no speaker count; Reads the first channel of a multichannel file; Speakers are numbered, not identified; no speaker embeddings are returned; Not streaming: the whole recording is processed at once; Polish is untested here; training data is mostly English and Mandarin.
        Parameters: 76M.
        Languages: und.
        Licence: [Creative Commons Attribution-NonCommercial 4.0](https://huggingface.co/BUT-FIT/diarizen-wavlm-large-s80-md-v2/blob/027b3e221b9d81dce2be794084f1dbd3ba2da403/README.md). Non-commercial use only (CC BY-NC 4.0), as the model card asks; credit BUT Speech@FIT. The VBx PLDA files are CC BY 4.0 and the WeSpeaker ResNet34 embedding (pyannote/wespeaker-voxceleb-resnet34-LM) is CC BY 4.0.
        Tools: the generic window built from its operations.

        - `diarize`: `POST /arsenal/v1/models/diarizen-large-v2/ops/diarize`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 278340411 | ~139170205 | unknown |

        [Model documentation](../models/diarizen-large-v2/README.md)

        ## EdgeTAM On-Device Track Anything Model (`edgetam`)

        Segment an object by click or box

        Cut one object out of a photo by clicking or boxing it, on the Mac.

        Good for: Boxing one object in a photo for an exact mask; Fast mask prediction on the Mac in FP16.
        Limits: A single click can be ambiguous: on the test photo a click on a cab window gave only the window, and a click on the truck bed gave speckled edges, while a box gave a clean whole-truck mask; Image path only: the video memory and tracking modules are not loaded; Masks are predicted at 256 pixels and upsampled, so fine edges are soft; No language input or output; Parity with the maker's transformers EdgeTAM on MPS FP16: mask logits cosine 0.99996 or better on five photo and prompt pairs.
        Parameters: 13.9M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/EdgeTAM-fp16/blob/e7770c15b913841cb657e6d2fbb33a6541c3bb32/README.md).
        Tools: the generic window built from its operations.

        - `segment`: `POST /arsenal/v1/models/edgetam/ops/segment`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 28002263 | 28002263 | unknown |

        [Model documentation](../models/edgetam/README.md)

        ## elephant-1 (`elephant-1`)

        Text, image and video decisions

        Calibrated yes/no, choice and score decisions on text, JSON, pictures and short videos, every question from one shared prefill.

        Good for: routing and triage with calibrated probabilities; questions about screenshots, photos and scans next to their text; what happens across a short clip given as frames; yes/no checks, one-of-many choices and graded scores on the same state; many questions about one document from a single prefill; guardrail and policy decisions on structured JSON.
        Limits: Pictures (PNG or JPEG, up to 64 megapixels) and videos as frame lists (2 to 768 stills of one size, with an fps) go in media and are placed by <image:N> / <video:N> in the state; video files themselves are not decoded, send their frames. A picture costs 64 to 16,384 tokens, a video up to 768 per pair of frames (960 for a 5-frame list). JPEGs are decoded by macOS, not PIL, and can differ slightly; Trained and evaluated in English; one Polish request answered correctly in a quick check, not measured; Trained on sequences up to 16,384 tokens; up to 32,768 for the state plus one question (65,536 per request, the makers' serving limits) is accepted but unverified beyond one long-input check. Longer requests are refused, not truncated; No calibration file ships with the checkpoint, so probabilities are its raw softmax.
        Parameters: 9.41B.
        Languages: en.
        Licence: [MIT](https://huggingface.co/baselabs/elephant-1/blob/b3f8a93462fb1051a9de031e94a7ff380e254072/README.md).
        Tools: .

        - `decide`: `POST /arsenal/v1/models/elephant-1/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/elephant-1/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 18839717602 | 18839717602 | unknown |

        [Model documentation](../models/elephant-1/README.md)

        ## EmbeddingGemma 2 (`embeddinggemma-2`)

        Search text, images, audio and video

        Find text, photos, audio and video by meaning in one small 768-number space.

        Good for: semantic search across text, code, photos, PDF pages, sound and video in one small index; task-tuned vectors for search, question answering, fact checking, code search, classification, clustering and similarity; short vectors (512, 256 or 128 numbers) to cut storage; interleaved inputs, such as a caption with its photos and a clip, as one vector.
        Limits: 8,192 tokens per input: an image takes up to 280, a video frame up to 140 (one frame a second, at most 32 frames), a second of audio 25; Audio past 30 seconds is refused unless truncate is set; then its first 30 seconds are used, as the maker's processor does; Video soundtracks are not used; embed the audio as a separate part; Google states 100+ languages without listing them; a Polish probe here scored a Polish aurora text above every unrelated item for an English aurora query (0.68 against 0.55 for a pasta text), but Polish retrieval quality was not measured.
        Parameters: 744M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/google/embeddinggemma-2/blob/f9972ac22e696a105c482202646979e33e4b5a1e/README.md). Use must also follow Google's Gemma Prohibited Use Policy.
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/embeddinggemma-2/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1521091852 | 1521091852 | unknown |

        [Model documentation](../models/embeddinggemma-2/README.md)

        ## EoMT COCO Panoptic Large 640 (`eomt-coco-panoptic-large-640`)

        Label every object and region

        Splits a photo into its objects and regions (people, animals, cars, sky, road, wall) and returns a labelled mask for each.

        Good for: One-shot breakdown of a scene: each person, animal, vehicle and object instance plus stuff such as sky, road, grass and wall; Picking a ready-made mask for a cut-out without clicking; Counting the instances of a COCO class in a photo.
        Limits: Only the 133 COCO classes: anything else is folded into the nearest class or left out (a rainbow sculpture was not labelled); The photo is resized to 640 pixels on its longest side, so small or thin things are missed and edges are soft; Takes no clicks, boxes or text: it segments the whole photo by itself; segments below the maker's 0.8 confidence are dropped; Photos above about 4 megapixels get their map computed at a scaled size and enlarged, so edges there are blockier; Class names are English; no language input; Parity with the maker's transformers EoMT on MPS FP16: identical labels and segments on four images, mask logits cosine 0.99999 or better on the same pixels.
        Parameters: 316.7M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/blob/dcd130bed9b1ebda7041fd660fddb16f905b9c3b/README.md).
        Tools: the generic window built from its operations.

        - `segment`: `POST /arsenal/v1/models/eomt-coco-panoptic-large-640/ops/segment`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1266836428 | ~633418214 | 1.600 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/eomt-coco-panoptic-large-640/README.md)

        ## ESM-2 650M (`esm2-t33-650m`)

        Embed protein sequences

        Turns a protein sequence (one-letter amino-acid codes) into a vector of numbers for similarity search, clustering or downstream prediction.

        Good for: Comparing proteins: similar sequences, homologues and point mutants get high cosine similarity; Features for a small classifier or regressor (function, localisation, stability) trained on top; Per-residue features for residue-level tasks such as binding-site or secondary-structure prediction.
        Limits: At most 1,022 residues per request (ESM-2's training crop); longer proteins are refused, so embed them in windows yourself; Uppercase one-letter amino-acid codes only (the 20 standard plus X, B, U, Z, O); other characters are refused. Not for DNA or RNA; Embeddings alone predict nothing: a property needs a model trained on top of them. Structure prediction (ESMFold) and the masked-residue head are not served; Slower and heavier than ESM-2 8M; the 3B and 15B sizes are larger still but are not part of Arsenal; Reads and writes no human language, so the Polish question does not apply; Parity with transformers EsmModel (MPS, float16) checked: see log/003-parity.md.
        Parameters: 652.4M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/facebook/esm2_t33_650M_UR50D/blob/08e4846e537177426273712802403f7ba8261b6c/README.md).
        Tools: the generic window built from its operations.

        - `embed_sequence`: `POST /arsenal/v1/models/esm2-t33-650m/ops/embed_sequence`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 2609507429 | ~1304753714 | unknown |

        [Model documentation](../models/esm2-t33-650m/README.md)

        ## ESM-2 8M (`esm2-t6-8m`)

        Embed protein sequences

        Turns a protein sequence (one-letter amino-acid codes) into a vector of numbers for similarity search, clustering or downstream prediction.

        Good for: Comparing proteins: similar sequences, homologues and point mutants get high cosine similarity; Features for a small classifier or regressor (function, localisation, stability) trained on top; Per-residue features for residue-level tasks such as binding-site or secondary-structure prediction.
        Limits: At most 1,022 residues per request (ESM-2's training crop); longer proteins are refused, so embed them in windows yourself; Uppercase one-letter amino-acid codes only (the 20 standard plus X, B, U, Z, O); other characters are refused. Not for DNA or RNA; Embeddings alone predict nothing: a property needs a model trained on top of them. Structure prediction (ESMFold) and the masked-residue head are not served; It is the weakest ESM-2 size: the card says larger checkpoints are more accurate. Use ESM-2 650M when quality matters more than speed; Reads and writes no human language, so the Polish question does not apply; Parity with transformers EsmModel (MPS, float16) checked: see log/003-parity.md.
        Parameters: 7.8M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/facebook/esm2_t6_8M_UR50D/blob/c731040fcd8d73dceaa04b0a8e6329b345b0f5df/README.md).
        Tools: the generic window built from its operations.

        - `embed_sequence`: `POST /arsenal/v1/models/esm2-t6-8m/ops/embed_sequence`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 31385380 | ~15692690 | unknown |

        [Model documentation](../models/esm2-t6-8m/README.md)

        ## ESM3 Open Small (esm3-sm-open-v1) (`esm3-sm-open-v1`)

        Protein design and prediction

        Reads and writes proteins across sequence, structure and function: fills in masked residues, scores mutations, folds a sequence into a backbone, designs a sequence for a structure, and predicts what a protein does.

        Good for: Filling in masked residues of a sequence (_ marks a mask), with the alternatives and their probabilities, plus the predicted secondary structure and solvent accessibility of every residue; Scoring mutations without any training data: masked or wild-type marginals over single and multiple substitutions (ProteinGym zero-shot average Spearman 0.466, the maker's benchmark table, not reproduced here); Folding a sequence into a backbone: structure tokens by iterative unmasking, then the structure decoder's N, CA, C coordinates with per-residue pLDDT, pTM and the predicted aligned error, written as a PDB file; Inverse folding: a PDB backbone in, a designed sequence out, with the recovery against the file's own sequence; Predicting what a protein does: InterPro entries and function keywords over residue ranges, from sequence, structure or both; Prompting across tracks at once (a motif's coordinates inside a longer protein, a secondary-structure string, SASA values, function keywords) and generating one track at a time, including the maker's chain of thought (function, secondary structure, structure, then sequence); Per-residue and pooled embeddings of the final layer for search, clustering and features.
        Limits: At most 1,024 residues per request, one chain: multi-chain complexes (the maker's | chain break and its chain_id track) are not ported, so a complex must be sent one chain at a time; Structure prompts read N, CA and C of the first model of a PDB file, amino acids only; mmCIF, gzip files, and residue-level insertion handling beyond the first alternate location are not covered; a PDB with no residue holding N, CA and C is refused; A reply carries at most 640,000 numbers (16 MiB as JSON): per-residue embeddings up to 415 residues, the predicted aligned error up to 800 residues, returned logits up to 640,000 numbers in all; larger asks are refused before the model runs; Residue annotations can be read back from the model's own head but cannot be prompted: the maker's residue-annotation tokenizer needs its per-residue reference letters, so such labels are accepted and leave that track empty; No batching: one protein per request. Generation runs one forward pass per step, at most 1,024 passes per generate request summed over its tracks, so it ends inside the 15-minute limit; Not ported from the maker's code: the Forge/Biohub service API, batch generation, chain-of-thought as one call (send the tracks in order yourself), guided and constrained generation, cRMSD against a reference, SASA from a structure (the maker computes it with biotite, not the model), and the ESMFold2, ESMC and SAE models in the same repository; The model generates plausible proteins, not verified ones: a design needs wet-lab or at least independent structure prediction. The maker filtered viruses and select agents from its training data and its keyword vocabulary, and asks users to follow its acceptable-use policy; Reads and writes no human language, so the Polish question does not apply; Matches the maker's own esm code on Apple GPU (bf16 reference, FP16 port) on fixed inputs per component; see log/002-parity.md and gate/golden/golden.json. The maker's benchmark figures are not reproduced here.
        Parameters: 2.70B.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/biohub/esm3-sm-open-v1/blob/47f0545b2b6daf26a93439a3cd610f4f7f3d5478/README.md).
        Tools: the generic window built from its operations.

        - `embed_sequence`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/embed_sequence`.
        - `predict_masked`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/predict_masked`.
        - `score_variants`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/score_variants`.
        - `generate`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/generate`.
        - `fold`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/fold`.
        - `inverse_fold`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/inverse_fold`.
        - `predict_function`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/predict_function`.
        - `encode_structure`: `POST /arsenal/v1/models/esm3-sm-open-v1/ops/encode_structure`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 5446204669 | 5446204669 | unknown |

        [Model documentation](../models/esm3-sm-open-v1/README.md)

        ## Fish Audio Drama 3 (`fish-drama-3`)

        Expressive speech, voice cloning

        Expressive speech in the voice of a short reference clip, with inline delivery cues and up to 16 speakers. Built with Fish Audio.

        Good for: text-to-speech.
        Limits: No preset voices: a request clones a reference clip with its exact transcript, a saved voice, or draws an unprompted voice; No speed control or instructions; delivery cues go inline, e.g. [whispers], and speakers as <|speaker:0|> to <|speaker:15|>; Runs only the vendor 4/8-bit checkpoint (about 27 GB resident); the 16-bit release is 72 GB and has no runner here; Fish Audio Community License: never imitate an identifiable person's voice without their consent.
        Parameters: 35B.
        Languages: zh, en, ja, ko, es, pt, ar, ru, fr, de, sv, it, tr, no, nl, cy, eu, ca, da, gl, ta, hu, fi, pl, et, hi, la, ur, th, vi, jw, bn, yo, sl, cs, sw, nn, he, ms, uk, id, kk, bg, lv, my, tl, sk, ne, fa, af, el, bo, hr, ro, sn, mi, yi, am, be, km, is, az, sd, br, sq, ps, mn, ht, ml, sr, sa, te, ka, bs, pa, lt, kn, si, hy, mr, as, gu, fo.
        Licence: [Fish Audio Community License 1.0](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit/blob/cc639e27aba7edb73429d3e3806f471dcd79d316/LICENSE). Free for research and non-commercial use; commercial use only while you and your affiliates earn under US$500,000 a year; no hosted speech service without a Fish Audio licence; never imitate a real person's voice without consent. Show "Built with Fish Audio" where you distribute it.
        Tools: voice, sound.

        - `speak`: `POST /arsenal/v1/models/fish-drama-3/ops/speak`.
        - `voices`: `POST /arsenal/v1/models/fish-drama-3/ops/voices`.
        - `clone_voice`: `POST /arsenal/v1/models/fish-drama-3/ops/clone_voice`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 72376539989 | 72376539989 | unknown |
        | int4 | 23781106243 | 23781106243 | 25.770 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/fish-drama-3/README.md)

        ## Florence-2 Base (`florence-2-base`)

        Caption a photo or read its text

        Writes an English caption for a photo, short or detailed, or reads the text in it.

        Good for: One-sentence captions of everyday photos; Paragraph descriptions with the prompt <DETAILED_CAPTION> or <MORE_DETAILED_CAPTION>; Reading short printed English text such as signs and labels with the prompt <OCR>.
        Limits: English only; it does not write Polish, and OCR of Polish diacritics is untested; The prompt is either a task token (<CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <OCR>) or the start of a caption to continue; detection, grounding and segmentation tasks are not offered because their boxes do not fit the caption output; OCR returns one string with line breaks dropped: two sign lines came back as 'Fresh bread dailyOpen 7am to 5pm'; Every image is squashed to 768x768, so very wide or tall images lose detail; a tiny image still gets a confident caption (a 33x17 zebra came back as a spotted hyena); The detailed captions can invent details (a green car was also called light blue with a white stripe); Parity with the maker code: output tokens identical on 9 of 9 fixed requests across all four tasks and a caption prefix (maker's repo code under transformers 5.22, MPS fp16); see log/003-parity.md.
        Parameters: 231.6M.
        Languages: en.
        Licence: [MIT](https://huggingface.co/microsoft/Florence-2-base/blob/5ca5edf5bd017b9919c05d08aebef5e4c7ac3bac/README.md).
        Tools: the generic window built from its operations.

        - `caption`: `POST /arsenal/v1/models/florence-2-base/ops/caption`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 465680283 | 465680283 | unknown |

        [Model documentation](../models/florence-2-base/README.md)

        ## Florence-2 Large (`florence-2-large`)

        Caption a photo or read its text

        Writes an English caption for a photo, short or detailed, or reads the text in it line by line.

        Good for: One-sentence captions of everyday photos; Paragraph descriptions with the prompt <DETAILED_CAPTION> or <MORE_DETAILED_CAPTION>; Reading printed English text such as signs and labels with the prompt <OCR>, one line per line.
        Limits: English only; it does not write Polish, and OCR of Polish diacritics is untested; The prompt is either a task token (<CAPTION>, <DETAILED_CAPTION>, <MORE_DETAILED_CAPTION>, <OCR>) or the start of a caption to continue; detection, grounding and segmentation tasks are not offered because their boxes do not fit the caption output; The card warns this continued-pretraining release used only 0.1B samples and might not be trained well; Every image is squashed to 768x768, so very wide or tall images lose detail; a tiny image still gets a confident caption (a 33x17 zebra came back as a cheetah); The detailed captions can invent details (a green car was also called bright turquoise); Parity with the maker code: output tokens identical on 9 of 9 fixed requests across all four tasks and a caption prefix (maker's repo code under transformers 5.22, MPS fp16); see log/003-parity.md.
        Parameters: 776.7M.
        Languages: en.
        Licence: [MIT](https://huggingface.co/microsoft/Florence-2-large/blob/21a599d414c4d928c9032694c424fb94458e3594/README.md).
        Tools: the generic window built from its operations.

        - `caption`: `POST /arsenal/v1/models/florence-2-large/ops/caption`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1556022541 | 1556022541 | unknown |

        [Model documentation](../models/florence-2-large/README.md)

        ## FLUX.2 Klein 4B (`flux2-klein-4b`)

        Image generation

        Four-step image generation with a local Qwen3 text encoder.

        Good for: text-to-image.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 4B.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md).
        Tools: image.

        - `generate`: `POST /arsenal/v1/models/flux2-klein-4b/ops/generate`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 15980140843 | 15980140843 | 20.398 (m5-max, 2026-10-16, quickCheck) |
        | int8 | made from the 16-bit download | ~8489449822 | 12.771 (m5-max, 2026-10-16, quickCheck) |
        | int4 | made from the 16-bit download | ~4494414612 | 9.005 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/flux2-klein-4b/README.md)

        ## FLUX 3 Image Klein (`flux3-image-klein`)

        Image generation and editing

        Black Forest Labs' 16-step image model: text to image, edits and compositions from 1 to 10 reference images, legible text.

        Good for: Photographs, illustrations and product shots from full scene captions, up to 4 megapixels; Text rendered inside the image: signs, labels, posters; Editing a photo from a caption of the result, and composing up to 10 reference images; Layout with bounding boxes written into the prompt (a JSON list of elements on a 0 to 1000 grid).
        Limits: FLUX Kommunity License: non-commercial use; commercial output only for Qualifying Users under US$5M annual revenue; FLUX trademarks need permission; Takes full scene captions: the maker says not to use it without a prompt upsampler (its 9B Pup model, not included); a short prompt gives generic images; No negative prompt or guidance (distilled, 16 steps); About 31 GB of memory at 1024 x 1024 in BF16 (the maker's figure); a 20 GB download from a gated repository.
        Parameters: 5.6B.
        Languages: en.
        Licence: [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/LICENSE.md). FLUX Kommunity License: you accept it on Hugging Face and Arsenal downloads from black-forest-labs/flux-3-image-klein with your token. Non-commercial use; generating outputs for commercial use is allowed only for Qualifying Users under US$5M gross annualized revenue (including affiliates). Black Forest Labs and FLUX trademarks need prior written permission. The Qwen3-VL-4B-Instruct text encoder is Apache-2.0. Gated: accept Black Forest Labs's terms on Hugging Face and add a read token in Settings.
        Tools: image.

        - `generate`: `POST /arsenal/v1/models/flux3-image-klein/ops/generate`.
        - `edit`: `POST /arsenal/v1/models/flux3-image-klein/ops/edit`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 20316385092 | 20316385092 | unknown |

        [Model documentation](../models/flux3-image-klein/README.md)

        ## Geneformer V2 104M (`geneformer-v2-104m`)

        Embed single cells from genes

        Turns one human cell's single-cell RNA profile, given as its genes in rank order, into a vector of numbers for comparing, clustering or classifying cells.

        Good for: Comparing cells: cells of the same type get closer embeddings than cells of different types; Features for a small cell type or cell state classifier trained on top; Clustering, visualising or batch-integrating single-cell datasets.
        Limits: The cell must arrive as human Ensembl gene IDs already in Geneformer's rank order: expression normalised by the cell's total counts and divided by each gene's median across Genecorpus-104M. Arsenal does not rank raw counts; the maker's TranscriptomeTokenizer does that step; Human protein-coding genes only (about 20,000 in the 104M dictionary); other IDs, gene symbols and other species are skipped and counted in genes_skipped. At most 4,094 genes per cell; lower-ranked genes are dropped; Non-cancer training data: malignant cells and cell lines are out of domain (the maker's separate cancer-tuned variant is not served); Raw <cls> cosines run high even between different cell types (about 0.96 for a T cell vs a hepatocyte marker profile against 0.996 for two T cell profiles), so compare cells relative to each other; The embedding alone predicts nothing: classification and in silico perturbation need models or code on top; the masked-gene head is not served; Reads and writes no human language, so the Polish question does not apply; Parity with transformers BertForMaskedLM (MPS, float16) checked: see log/003-parity.md.
        Parameters: 104.4M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/ctheodoris/Geneformer/blob/1f7fbae4e469a5f4f1af8c111a529cfe1b3829f5/README.md).
        Tools: the generic window built from its operations.

        - `embed_cells`: `POST /arsenal/v1/models/geneformer-v2-104m/ops/embed_cells`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 417997426 | ~208998713 | unknown |

        [Model documentation](../models/geneformer-v2-104m/README.md)

        ## GIMM-VFI-R (`gimm-vfi-r`)

        In-between frames at any moment

        Create in-between frames at any point between two images, or raise a video's frame rate, by modelling the motion between them.

        Good for: Arbitrary in-between moments (any t from 0 to 1, up to sixteen per pair); Slow motion and frame-rate conversion of video; Large and non-linear motion between frames.
        Limits: Frames up to about a megapixel run at full flow resolution; larger frames estimate motion at half (1080p, 1440p) or a quarter (4K) of their size, as the makers run 2K footage; Each frame pair runs RAFT for 20 iterations in both directions, so it is slower than RIFE; Occlusions, motion blur and scene cuts between the frames produce ghosting; Checked against the makers' code on one 281x240 pair; not benchmarked here.
        Parameters: 19.8M.
        Languages: und.
        Licence: [S-Lab License 1.0](https://github.com/GSeanCDAT/GIMM-VFI/blob/dbc56449994a3c2e045d46fd46bed570239913f8/LICENSE). Non-commercial use only; redistribution in source or binary form must keep S-Lab's copyright notice, conditions and disclaimer. Commercial use needs the authors' permission.
        Tools: the generic window built from its operations.

        - `interpolate_video`: `POST /arsenal/v1/models/gimm-vfi-r/ops/interpolate_video`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 79305223 | ~39652611 | unknown |

        [Model documentation](../models/gimm-vfi-r/README.md)

        ## GLM-OCR (`glm-ocr`)

        Read text from a document

        Reads the text in a document image or screenshot and returns it as plain text, keeping lines and paragraphs.

        Good for: Printed documents, receipts, forms and screenshots: text comes back line by line with blank lines between blocks; Pages in Chinese, English, French, Spanish, Russian, German, Japanese or Korean.
        Limits: Runs only the card's Text Recognition prompt; the Formula Recognition, Table Recognition and JSON extraction prompts are not offered, and there is no layout analysis (the maker's SDK adds PP-DocLayoutV3 for that); Polish is not among the card's languages; plain Latin text reads fine, but Polish diacritics are untested; Column alignment is lost: a three-column table came back as single-spaced rows (Oak shelf 2 149.00); A tiny image still gets an answer: a 33x17 photo of a zebra was read as 'Zebra'; A large page becomes thousands of image tokens (up to about 4.8 million pixels after resizing), which needs several GB of memory; Parity with the maker code: output tokens identical on 3 of 3 fixed inputs (transformers 5.22 GlmOcrForConditionalGeneration, MPS fp16); see log/003-parity.md.
        Parameters: 1.3B.
        Languages: zh, en, fr, es, ru, de, ja, ko.
        Licence: [MIT](https://huggingface.co/zai-org/GLM-OCR/blob/2e85a62840ccac27daa451df36c736c4636b8628/README.md).
        Tools: the generic window built from its operations.

        - `read`: `POST /arsenal/v1/models/glm-ocr/ops/read`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 2657422814 | 2657422814 | unknown |

        [Model documentation](../models/glm-ocr/README.md)

        ## Granite Embedding Small English R2 (`granite-embedding-small-english-r2`)

        Embed English text for search

        Turns English text into 384-number vectors so you can search and compare passages by meaning.

        Good for: Semantic search and retrieval over English passages; Comparing sentences, queries and documents by cosine similarity; Documents up to 8192 tokens.
        Limits: English only: the card says it was trained for English texts, so it does not handle Polish; Vectors are unnormalized unless normalize is true; no query or document prefixes are used, as the card specifies; Texts over 8192 tokens are cut when truncate is true (the default); otherwise the request is refused; Text only: images, audio and mixed content are refused; Parity with the maker code not run; benchmark quality not measured here.
        Parameters: 47.7M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/ibm-granite/granite-embedding-small-english-r2/blob/2ab6fa8ea2d674564defd37171ae19079b864b33/README.md).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/granite-embedding-small-english-r2/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 98938597 | 98938597 | 0.301 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/granite-embedding-small-english-r2/README.md)

        ## Granite TimeSeries TTM R3 (`granite-ttm-r3`)

        Forecast numeric time series

        Forecasts the next 30 values of one numeric series as a range of likely outcomes.

        Good for: Very fast zero-shot forecasts of one numeric series, such as sales, load or counts, up to 30 steps ahead; Seasonal series: on monthly airline passengers it continues the summer peak and winter dip of the last years; Short histories: it pads anything under 512 steps and still answers from 3 values.
        Limits: Univariate here: one series in; the model's multivariate and exogenous modes are not ported; Horizon is at most 30 steps (the checkpoint's length; no rollout); Uses at most the last 512 values for the trend and the last 90 for the residual; older values only set the scale; Returns only the trained quantile levels 0.1 to 0.9 in steps of 0.1; Reads and writes no language, so the Polish question does not apply; Parity with the maker code (granite-tsfm, MPS, networks fp16): cosine 1.0000000, max abs diff 0.29 passengers on values up to 663 (airline series).
        Parameters: 1.4M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3/blob/ea17cfd2e3edcaea21eb8dcecd18bf88971482fa/README.md).
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/granite-ttm-r3/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 5732092 | ~2866046 | unknown |

        [Model documentation](../models/granite-ttm-r3/README.md)

        ## Grounding DINO Base (`grounding-dino-base`)

        Find objects described in words

        Finds and boxes the things you describe in a short English phrase

        Good for: Finding objects named in plain words, including descriptive phrases ("black cable", "a remote control"); Several things at once; each box is labelled with the phrase it matched.
        Limits: English only; Polish words are not understood; Labels and phrases are lowercased and joined as "a. b." for the model; full stops, commas and semicolons split a phrase into separate things to find, at most 256 tokens in all; No duplicate suppression, as in the maker's post-processing: one object can get two overlapping boxes, and a box around a whole group can appear beside the single objects; Each box gets the one phrase whose words score highest; the maker's own post-processing can return a partial phrase instead; Needs more memory than OWLv2: a Swin-B backbone on an 800-pixel view plus six encoder and six decoder layers; At most 128 things per request.
        Parameters: 232.8M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/IDEA-Research/grounding-dino-base/blob/12bdfa3120f3e7ec7b434d90674b3396eccf88eb/README.md).
        Tools: the generic window built from its operations.

        - `detect`: `POST /arsenal/v1/models/grounding-dino-base/ops/detect`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 934347332 | ~467173666 | 3.758 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/grounding-dino-base/README.md)

        ## Harrier OSS v1 0.6B (`harrier-oss-v1-0.6b`)

        Create text search embeddings

        Turn text into vectors for semantic search and similarity.

        Good for: semantic search; similarity; clustering; classification.
        Limits: Checked live at FP16 on 19 Oct: a cat/kitten pair scored 0.82, unrelated texts 0.40 to 0.42.
        Parameters: 596M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/microsoft/harrier-oss-v1-0.6b/tree/f9b9dc8d367d443f2479d27aa5d8d2850c0774ee).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/harrier-oss-v1-0.6b/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1203563696 | 1203563696 | 1.954 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/harrier-oss-v1-0.6b/README.md)

        ## HTDemucs fine-tuned (`htdemucs-ft`)

        Separate music into four stems

        Separate music into four stems using a native Swift MLX worker.

        Good for: Separate music into four stems.
        Limits: All four fine-tuned submodels run; Length depends on selected outputs under the 4 GiB per-job file limit; Trained-weight and precision-quality verification is pending.
        Parameters: 168M.
        Languages: und.
        Licence: [MIT](https://huggingface.co/mlx-community/demucs-mlx-fp16/tree/908d2d05cf3035bf4017fc0f52be64163689a61f).
        Tools: the generic window built from its operations.

        - `separate_audio`: `POST /arsenal/v1/models/htdemucs-ft/ops/separate_audio`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 336150390 | 336150390 | 1.460 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/htdemucs-ft/README.md)

        ## IDOL (`idol`)

        One picture to a 3D splat

        Turns one picture of an object into a 3D Gaussian splat (PLY).

        Good for: One object on a removed background: products, toys, furniture, food, figurines, renders and generated images; A splat that opens in any 3D Gaussian Splatting viewer, with the input camera's angle and field of view; 32,768 to 1,048,576 Gaussians per object.
        Limits: The background is removed first with BiRefNet; give a cutout with a transparent background, or keep BiRefNet on the Mac; A splat is not a mesh: it has no faces to edit, print or rig, and the Mac's own viewers do not open PLY splats; The hidden back is invented; one object per picture; A seed gives a different (equally valid) result than IDOL's PyTorch code with the same seed: the random numbers come from MLX; Precision: every weight 16-bit (IDOL ships FP32; Get casts its modules to FP16, DINOv3 to BF16 and the VAE to FP16); norms, rotary angles and embeddings compute in float32 as IDOL's do.
        Parameters: 0.66B.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/speridlabs/idol/blob/2731c2291fc65a797c5cb627a5e5526fc1490902/README.md). IDOL itself is Apache-2.0. Its DINOv3 image encoder is under Meta's DINOv3 License, accepted on Hugging Face (Meta approves each request): Arsenal downloads from facebook/dinov3-vith16plus-pretrain-lvd1689m with your token. No ITAR or other trade-controlled uses; anything redistributed carries that licence. Gated: accept Meta's terms on Hugging Face and add a read token in Settings.
        Tools: the generic window built from its operations.

        - `image_to_3d`: `POST /arsenal/v1/models/idol/ops/image_to_3d`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 2634514375 | ~1317257187 | unknown |

        [Model documentation](../models/idol/README.md)

        ## Index Translate 2B (`index-translate-2b`)

        Translate with glossary and style

        Translate with glossary and style.

        Good for: Translate with glossary and style.
        Limits: Checked live on 20 Oct at FP16 and int8: Polish correct; German dropped one word in one sentence. int4 failed that check and is not offered.
        Parameters: 2.2B.
        Languages: mul.
        Licence: [Apache License 2.0](https://huggingface.co/IndexTeam/Index-Translate-2B/tree/9bc08445315723eb9b77d1b93505915eb03c3f01).
        Tools: the generic window built from its operations.

        - `translate`: `POST /arsenal/v1/models/index-translate-2b/ops/translate`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 4561044082 | 4561044082 | 6.067 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~2423054668 | 3.405 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/index-translate-2b/README.md)

        ## Index Translate 9B (`index-translate-9b`)

        Translate with glossary and style

        Translate with glossary and style.

        Good for: Translate with glossary and style.
        Limits: Checked live on 20 Oct at FP16, int8 and int4: Polish and German fluent and correct.
        Parameters: 9.4B.
        Languages: mul.
        Licence: [Apache License 2.0](https://huggingface.co/IndexTeam/Index-Translate-9B/tree/4e06279e62352dab96bbeb6fa0579bca5e822d75).
        Tools: the generic window built from its operations.

        - `translate`: `POST /arsenal/v1/models/index-translate-9b/ops/translate`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 19319201802 | 19319201802 | 22.881 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~10263325957 | 11.811 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~5433525506 | 7.038 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/index-translate-9b/README.md)

        ## Kokoro 82M (`kokoro-82m`)

        Fast English speech

        Fast preset-voice reader; the default for spoken replies. No cloning or voice design.

        Good for: text-to-speech.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 82M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/Kokoro-82M-bf16/blob/a71e4d38b236d968966a2002c4c895dbd12b1c3c/README.md).
        Tools: voice, sound.

        - `speak`: `POST /arsenal/v1/models/kokoro-82m/ops/speak`.
        - `voices`: `POST /arsenal/v1/models/kokoro-82m/ops/voices`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 341742463 | ~170871231 | 1.386 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/kokoro-82m/README.md)

        ## Laya · English (`laya-english`)

        English decisions

        Answers yes-no, choice and score questions about a text in one pass; English only.

        Good for: yes-no; choice; score.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 421M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/aac6fef/laya-mlx/blob/20aed815fc6acde75733882e7ec0e3f28aeb9717/LICENSE).
        Tools: the generic window built from its operations.

        - `decide`: `POST /arsenal/v1/models/laya-english/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/laya-english/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 846226754 | 846226754 | 1.186 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/laya-english/README.md)

        ## Laya · Multilingual (`laya-multilingual`)

        Decisions in 100+ languages

        The Laya decision model for non-English text (100+ languages); the fastest Laya.

        Good for: yes-no; choice; score.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 322M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/aac6fef/laya-multilingual-mlx/blob/f2b4faf51023039425946074e2cf1361d2db11d5/LICENSE).
        Tools: the generic window built from its operations.

        - `decide`: `POST /arsenal/v1/models/laya-multilingual/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/laya-multilingual/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 678232831 | 678232831 | 1.332 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/laya-multilingual/README.md)

        ## Laya · Typed decisions (`laya-typed-decisions`)

        Typed workflow decisions

        English Laya tuned for typed workflow decisions (invoices, incidents, tickets); the best-calibrated Laya.

        Good for: yes-no; choice; score.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 421M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/aac6fef/laya-typed-decisions-mlx/blob/f9e501c2080cc57c13d6887820329758f5351125/LICENSE).
        Tools: the generic window built from its operations.

        - `decide`: `POST /arsenal/v1/models/laya-typed-decisions/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/laya-typed-decisions/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 846226968 | 846226968 | 1.186 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/laya-typed-decisions/README.md)

        ## LCO-Embedding Omni 7B (`lco-embedding-omni-7b`)

        Search text, images, audio and video

        Find text, images, PDF pages, audio and video by meaning, all in one vector space.

        Good for: semantic search across text, photos, PDF pages, sound and video in one index; text-to-image, text-to-audio and text-to-video retrieval; visual document retrieval; similarity, clustering and classification.
        Limits: The soundtrack of a video is not used; embed it separately as audio; 32,768 tokens per input: about 13 letter-size PDF pages or 40 seconds of full-size video; at most 5 minutes of audio per part; One prompt for queries and documents; no instructions or shorter dimensions.
        Parameters: 8.93B.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/LCO-Embedding/LCO-Embedding-Omni-7B/blob/108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c/README.md).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/lco-embedding-omni-7b/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 17875210374 | 17875210374 | unknown |

        [Model documentation](../models/lco-embedding-omni-7b/README.md)

        ## Manga OCR Base (`manga-ocr-base`)

        Read Japanese manga text

        Reads the Japanese text in a manga speech bubble or text crop, vertical or horizontal.

        Good for: One speech bubble or caption cropped from a manga page, vertical or horizontal; Short printed Japanese text crops: signs, screenshots, titles, furigana-heavy text.
        Limits: Reads one text block per image and has no text detector: crop each bubble first, a whole page is not split into bubbles; Trained for Japanese only; it is not meant for Polish. Latin letters and digits inside Japanese text come out full-width (ＬＩＮＫ, ７); On a tiny or blurred crop it invents plausible Japanese instead of refusing (a 33x17 crop gave a made-up sentence); Up to 299 characters per image; Parity with the maker code: 4-beam output tokens identical on 6 of 6 fixed inputs (transformers 5.22, MPS fp16); see log/003-parity.md.
        Parameters: 111.0M.
        Languages: ja.
        Licence: [Apache-2.0](https://huggingface.co/kha-white/manga-ocr-base/blob/82c77fb84369e0fce0d6e8f0ad96fa82646d0346/README.md).
        Tools: the generic window built from its operations.

        - `read`: `POST /arsenal/v1/models/manga-ocr-base/ops/read`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 444162320 | ~222081160 | unknown |

        [Model documentation](../models/manga-ocr-base/README.md)

        ## Mel-Band RoFormer Vocals (Kim) (`melband-roformer-kim`)

        Pull voice out of music

        Pull the vocals out of a song, or keep the instrumental, using a native Swift MLX worker.

        Good for: Vocals out of a full song, for karaoke, remixes, covers or transcription; An instrumental bed without the voice; Speech or narration out of music and sound effects; Long tracks: it works through 8 s chunks, so memory stays flat.
        Limits: Two stems only: vocals and everything else (instrumental); Stereo 44.1 kHz; mono is duplicated to both channels and other rates are resampled; Up to ten minutes per clip; Backing vocals and choirs count as vocals; A whole 8 s chunk runs even for a shorter clip.
        Parameters: 228M.
        Languages: und.
        Licence: [MIT](https://huggingface.co/KimberleyJSN/melbandroformer/blob/ac9b0614ab3cd7f77219e18ba494dfd93956c348/README.md).
        Tools: the generic window built from its operations.

        - `separate_audio`: `POST /arsenal/v1/models/melband-roformer-kim/ops/separate_audio`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 913106900 | ~456553450 | unknown |

        [Model documentation](../models/melband-roformer-kim/README.md)

        ## MiniMax Music 3 (`minimax-music3`)

        Songs from lyrics and a description

        Complete 44.1 kHz stereo songs up to six minutes from lyrics and a music description, with vocals.

        Good for: Complete songs with sung vocals, up to six minutes; Structured captions: genre, BPM, key, vocal timbre and section-by-section arrangement; Instrumental pieces.
        Limits: Section tags such as [verse] and [chorus] must stand on their own lines; text after a tag on the same line is dropped; Duration is an upper bound: the model may end the song earlier, and past the given lyrics it may sing lines of its own; Tempo, key, instrumentation and structure are guidance, not guarantees; The maker documents no vocal languages; its examples are English, and Polish is untested; Generation produces a complete file with progress, not live audio; 16-bit only: about 22 GB in memory when loaded.
        Parameters: 11.7B.
        Languages: en, zh.
        Licence: [MiniMax-Music3 Community License](https://huggingface.co/MiniMaxAI/MiniMax-Music3/blob/1b525ccb685ae809f33027c51c351eeb5833cf72/LICENSE). MiniMax-Music3 Community License: keep the copyright and licence notice with copies; follow its acceptable-use policy (no infringing outputs, AI-generated content disclosed when published); a commercial product shows "MiniMax-Music3" in its interface, and above US$20 million yearly revenue needs MiniMax's written authorization. Anyone offering others a product or service that generates with it must keep reasonable safeguards against infringing or prohibited uses and outputs (section 4).
        Tools: sound.

        - `generate_music`: `POST /arsenal/v1/models/minimax-music3/ops/generate_music`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 28517608999 | 28517608999 | unknown |

        [Model documentation](../models/minimax-music3/README.md)

        ## Mitra-v2 Classifier (`mitra-classifier-2`)

        Classify table rows in context

        Labels the rows of a numeric table from a few rows you have already labelled, with no training.

        Good for: Small and medium numeric tables: give labelled rows and get labels and class probabilities for new rows in one call, no training step; Up to 10 classes, 2048 labelled rows and 100 columns; The card's wine example: all 36 held-out wines get the right cultivar from 142 labelled ones.
        Limits: Zero-shot only: the maker's benchmark results use 50-step fine-tuning and 8-fold bagging, which this runner does not do; Numbers only: categorical or text columns must be encoded as numbers first, and missing values are not accepted; At most 10 classes, 2048 labelled rows and 100 columns; columns with one value in the labelled rows are ignored; One estimator with the maker's random column sign flips and shuffles switched off, so answers are deterministic; Classification only; the regressor is a separate model (mitra-regressor-2); Reads and writes no language, so the Polish question does not apply; Parity with the maker code (autogluon.tabular 1.6.3, MPS fp16): same labels on all 36 wine rows, probability cosine 0.9999999 and largest gap 0.0016; on a 2048-row stress table 699 of 700 labels match, the other an exact tie in the reference.
        Parameters: 75.7M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/autogluon/mitra-classifier-2/blob/edada0d20759c58ada8c8605c25f22f6e98ea5f0/README.md).
        Tools: the generic window built from its operations.

        - `classify_rows`: `POST /arsenal/v1/models/mitra-classifier-2/ops/classify_rows`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 302717990 | ~151358995 | unknown |

        [Model documentation](../models/mitra-classifier-2/README.md)

        ## MoGe-3 ViT-L (`moge-3-vitl`)

        Metric depth and 3D from one image

        Metric depth and 3D from one image.

        Good for: Metric point maps, surface normals and camera intrinsics.
        Limits: Checked live at FP16 on 19 Oct: metric depth with sharp edges, near and far in the right order; Single image only.
        Parameters: 370M.
        Languages: und.
        Licence: [MIT](https://huggingface.co/mlx-community/moge-3-vitl-mlx-fp32/tree/f5ceca7512506731fb1c50307c313da76aac500f).
        Tools: the generic window built from its operations.

        - `estimate_depth`: `POST /arsenal/v1/models/moge-3-vitl/ops/estimate_depth`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1481206911 | ~740603455 | 2.856 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/moge-3-vitl/README.md)

        ## MoLFormer-XL Both 10% (`molformer-xl-both-10pct`)

        Embed molecules from SMILES

        Turns a molecule written as a SMILES string into a vector of numbers for similarity search, clustering or property prediction.

        Good for: Comparing molecules: structurally related compounds (caffeine and theobromine) get a higher cosine than unrelated ones (caffeine and aspirin); Features for a small classifier or regressor of a molecular property (solubility, toxicity, permeability) trained on top; Clustering or visualising a compound library.
        Limits: At most 200 SMILES tokens per molecule (training dropped longer ones); larger molecules and macromolecules are refused, and the card says they are untested; SMILES only: input with characters outside SMILES is refused. Invalid or non-canonical SMILES and stereo marks give weaker embeddings (training used canonical SMILES with isomeric information removed); tokens missing from the vocabulary are read as unknown and listed in the reply; Embeddings alone predict nothing: a property needs a model trained on top of them. Not for molecule generation; the masked-token head is not served; This is the 10% ZINC + 10% PubChem checkpoint; the card's full MoLFormer-XL scores higher on most benchmarks; Reads and writes no human language, so the Polish question does not apply; Parity with the maker's modeling_molformer.py (MPS, float32, since its float16 path overflows) checked: see log/003-parity.md.
        Parameters: 46.8M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/ibm-research/MoLFormer-XL-both-10pct/blob/361063d0ad524ef77cf39b08469f6be770dc550f/README.md).
        Tools: the generic window built from its operations.

        - `embed_molecule`: `POST /arsenal/v1/models/molformer-xl-both-10pct/ops/embed_molecule`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 187304941 | ~93652470 | unknown |

        [Model documentation](../models/molformer-xl-both-10pct/README.md)

        ## MOSS-Transcribe-Diarize 0.9B (`moss-transcribe-diarize`)

        Transcribe who said what, and when

        Transcribe long recordings with speaker labels and timestamps in one pass, using a native Swift MLX worker.

        Good for: Meetings, calls, interviews and podcasts: who said what, with start and end times per segment; Long recordings in one pass, without a separate diarization model; Domain words: pass hotwords to steer the spelling of names and terms; Custom instructions, such as speaker-only transcription without timestamps.
        Limits: Speakers are anonymous labels in order of appearance ([S01], [S02], …), not identities, and are numbered afresh for each recording; Timestamps are per segment, not per word; Accepts up to 90 minutes per recording, as the maker states, but checked here only on recordings under a minute; a full hour of speech is estimated to need longer than the 590 s request limit; Trained mostly on Chinese and English; the maker claims 50+ languages but names none, so other languages, Polish included, are not verified by the maker; The default max_new_tokens of 5,120 covers about 13 minutes of dense speech; a reply cut short by it says truncated: true, and a longer recording needs a larger max_new_tokens.
        Parameters: 908.5M.
        Languages: zh, en.
        Licence: [Apache License 2.0](https://huggingface.co/OpenMOSS-Team/MOSS-Transcribe-Diarize/blob/10304d926740cf5a5cee05a268086e3265197910/README.md).
        Tools: the generic window built from its operations.

        - `transcribe`: `POST /arsenal/v1/models/moss-transcribe-diarize/ops/transcribe`.
        - `diarize`: `POST /arsenal/v1/models/moss-transcribe-diarize/ops/diarize`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1828539928 | 1828539928 | unknown |

        [Model documentation](../models/moss-transcribe-diarize/README.md)

        ## MossFormer2 SE 48K (`mossformer2-se-48k`)

        Speech denoising at 48 kHz

        Remove background noise from speech at full 48 kHz bandwidth using a native Swift MLX worker.

        Good for: Removing background noise from voice recordings, calls, podcasts and interviews; Full-band 48 kHz speech, keeping the high frequencies other 16 kHz denoisers drop; Long recordings: over 20 s it cleans 4 s windows at a 3 s stride, so memory stays flat; Mono or stereo at any rate; each channel is cleaned and returned at the input's rate.
        Limits: Denoising only: it does not remove reverb, repair clipping or raise the sample rate; Speech only; music and other sounds under the noise are treated as noise; Up to 10 minutes per clip; Runs in BF16: its FP16 cast overflows inside the network, so Macs without BF16 arithmetic (M1, M2) emulate it more slowly; Output is not streamed while it is made; the whole clip is cleaned first.
        Parameters: 55.3M.
        Languages: und.
        Licence: [Apache License 2.0](https://huggingface.co/alibabasglab/MossFormer2_SE_48K/blob/eff8c97925c8bec812af707814b3e5d777fd4503/README.md).
        Tools: the generic window built from its operations.

        - `enhance`: `POST /arsenal/v1/models/mossformer2-se-48k/ops/enhance`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 221552019 | ~110776009 | unknown |

        [Model documentation](../models/mossformer2-se-48k/README.md)

        ## Nemotron 3 Diarization (`nemotron-3-diarization`)

        Identify who spoke when

        Identify who spoke when using a native Swift MLX worker.

        Good for: Identify who spoke when.
        Limits: Up to eight speakers; Two streaming sessions; ten minutes per recording; a session idle for ten minutes is dropped when a new one needs its slot; Trained-weight and precision-quality verification is pending.
        Parameters: 99M.
        Languages: und.
        Licence: [OpenMDW License Agreement 1.1](https://huggingface.co/nvidia/Nemotron-3-Diarization/blob/f667ed73aee57d40cc39428eb768b4fd87a0a29e/README.md).
        Tools: the generic window built from its operations.

        - `diarize`: `POST /arsenal/v1/models/nemotron-3-diarization/ops/diarize`.
        - `diarize_session`: `POST /arsenal/v1/models/nemotron-3-diarization/ops/diarize_session`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 198633741 | 198633741 | 0.715 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/nemotron-3-diarization/README.md)

        ## NuExtract3 (`nuextract3`)

        Fill JSON templates from documents

        Fill JSON templates from documents.

        Good for: Fill JSON templates from documents.
        Limits: FP16 only: int8 and int4 changed read_document output too much. In the 22 Oct live check extract returned invalid JSON; check its output; The extraction template is a NuExtract template, not a JSON Schema; A document is read as at most 99 page images per request.
        Parameters: 4.5B.
        Languages: mul.
        Licence: [Apache License 2.0](https://huggingface.co/numind/NuExtract3/tree/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1).
        Tools: the generic window built from its operations.

        - `extract`: `POST /arsenal/v1/models/nuextract3/ops/extract`.
        - `read_document`: `POST /arsenal/v1/models/nuextract3/ops/read_document`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 9098678301 | 9098678301 | 17.180 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/nuextract3/README.md)

        ## Ovis VL Embedding 2B (`ovis-vl-embedding-2b`)

        Find text, images and video by meaning

        Find text, images, PDF pages and video by meaning.

        Good for: semantic search; similarity; clustering; classification.
        Limits: Checked live on 20 Oct at FP16, int8 and int4: text-to-photo search found the right photo 3 of 3 times; No audio. PDF documents are rendered as page images; Serving bounds: 262144 tokens, 128 PDF pages, 768 sampled video frames.
        Parameters: 2.21B.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/ATH-MaaS/Ovis-VL-Embedding-2B/tree/855b64a8246cf83b34730e4470ef66949e030659).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/ovis-vl-embedding-2b/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 4446561309 | 4446561309 | 5.949 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~2362235695 | 3.290 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~1250595368 | 1.920 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/ovis-vl-embedding-2b/README.md)

        ## OWLv2 Base Patch16 Ensemble (`owlv2-base-patch16-ensemble`)

        Find objects named in words

        Finds and boxes whatever you name in a photo, from a list of short English phrases

        Good for: Finding everyday objects by name with no training ("a cat", "a red car", "a coffee mug"); Several queries at once; each box gets the best-matching query.
        Limits: English queries only; Polish words are not understood; Each query is cut to 14 word pieces (the text encoder takes 16 tokens); No duplicate suppression, as in the maker's post-processing: one object can get two overlapping boxes; Scores are calibrated low; good matches often score 0.2 to 0.7, so lower the threshold below the default 0.3 for hard queries; Small objects in a large photo get few patches after the 960 by 960 resize.
        Parameters: 155.0M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/google/owlv2-base-patch16-ensemble/blob/cfd3195ba4ea9592eec887ded089f4c08eff231d/README.md).
        Tools: the generic window built from its operations.

        - `detect`: `POST /arsenal/v1/models/owlv2-base-patch16-ensemble/ops/detect`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 621505532 | ~310752766 | 0.666 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/owlv2-base-patch16-ensemble/README.md)

        ## Parakeet v3 Ultra (`parakeet-v3-ultra`)

        Speech transcription, 25 languages

        Fast transcription in 25 European languages, with word timings

        Good for: Fast dictation and transcription in 25 European languages; Sentence and word timestamps.
        Limits: Only the 25 European languages it was trained on; Detects the spoken language and cannot be told which one to expect; Ten minutes per recording, decoded in 30-second windows with 2 seconds of overlap; Checked live at FP16 on 20 Oct: an English test sentence transcribed exactly.
        Parameters: 627M.
        Languages: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk.
        Licence: [CC-BY-4.0](https://huggingface.co/selcukkubur/parakeet-ultra-mlx/tree/b554592c50b2a48471add2daa3d46fa9f00fef5e).
        Tools: the generic window built from its operations.

        - `transcribe`: `POST /arsenal/v1/models/parakeet-v3-ultra/ops/transcribe`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1254428555 | 1254428555 | 1.825 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/parakeet-v3-ultra/README.md)

        ## Pocket TTS (`pocket-tts`)

        Streaming speech, voice cloning

        Small streaming voice for short replies with preset or cloned voices. No voice design.

        Good for: text-to-speech.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 118M.
        Languages: en.
        Licence: [CC-BY-4.0](https://huggingface.co/mlx-community/pocket-tts/blob/cbf71d5f6657bbc3f4bc02f85ee408261225bec7/README.md).
        Tools: voice, sound.

        - `speak`: `POST /arsenal/v1/models/pocket-tts/ops/speak`.
        - `voices`: `POST /arsenal/v1/models/pocket-tts/ops/voices`.
        - `clone_voice`: `POST /arsenal/v1/models/pocket-tts/ops/clone_voice`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 239296565 | 239296565 | 0.503 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/pocket-tts/README.md)

        ## Pyannote Community-1 (`pyannote-community-1`)

        Who spoke when, with overlaps

        Find who spoke when, including overlapping speech, using a native Swift MLX worker.

        Good for: Meeting and interview speaker turns; Overlapping speech; Exclusive one-speaker-at-a-time turns for aligning transcripts; Known, bounded or automatic speaker counts; RTTM export and per-speaker embeddings.
        Limits: At most ten minutes per recording and 32 requested speakers; Not streaming: the whole recording is processed at once; Trained-weight and precision-quality verification is pending.
        Parameters: 8.1M.
        Languages: und.
        Licence: [Creative Commons Attribution 4.0](https://huggingface.co/pyannote-community/speaker-diarization-community-1/tree/8a527374977391da736e0daaef26855d949d9685).
        Tools: the generic window built from its operations.

        - `diarize`: `POST /arsenal/v1/models/pyannote-community-1/ops/diarize`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 32821421 | ~16410710 | 0.483 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/pyannote-community-1/README.md)

        ## Qwen3 ASR 0.6B (`qwen3-asr-0.6b`)

        Speech transcription, 30 languages

        The smaller Qwen3 ASR: the same 30 languages in less memory

        Good for: Chinese, Japanese, Korean and 27 more languages on Macs with less memory; Language detection and context prompts.
        Limits: Ten minutes per recording, transcribed in segments of up to 30 seconds; No word timestamps: the separate Qwen forced aligner is not included; A little less accurate than the 1.7B in Vella's measurements; Checked live at FP16 on 20 Oct: an English test sentence transcribed exactly.
        Parameters: 782M.
        Languages: zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16/tree/eae2b51f96265328f1e7beced788adb0e4536f92).
        Tools: the generic window built from its operations.

        - `transcribe`: `POST /arsenal/v1/models/qwen3-asr-0.6b/ops/transcribe`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1569391676 | 1569391676 | 2.609 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/qwen3-asr-0.6b/README.md)

        ## Qwen3 ASR 1.7B (`qwen3-asr-1.7b`)

        Speech transcription, 30 languages

        Transcribe speech in 30 languages, including Chinese, Japanese and Korean

        Good for: Chinese, Japanese, Korean and 27 more languages; Language detection and context prompts.
        Limits: Ten minutes per recording, transcribed in segments of up to 30 seconds; No word timestamps: the separate Qwen forced aligner is not included; Checked live at FP16 on 20 Oct: an English test sentence transcribed exactly.
        Parameters: 2038M.
        Languages: zh, en, yue, ar, de, fr, es, pt, id, it, ko, ru, th, vi, ja, tr, hi, ms, nl, sv, da, fi, pl, cs, fil, fa, el, hu, mk, ro.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16/tree/e1f6c266914abc5a46e8756e02580f834a6cf8a7).
        Tools: the generic window built from its operations.

        - `transcribe`: `POST /arsenal/v1/models/qwen3-asr-1.7b/ops/transcribe`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 4080656442 | 4080656442 | 5.036 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/qwen3-asr-1.7b/README.md)

        ## Qwen3 Embedding 0.6B (`qwen3-embedding-0.6b`)

        Multilingual text embeddings

        Turns text into vectors for search by meaning, in over 100 languages.

        Good for: Search by meaning across many languages, including Polish; Similarity, clustering and classification of short and long texts; Queries with a task instruction (task query) against plain documents.
        Limits: Vectors match Qwen's own code (cosine 0.99999 on six texts); vectors from Arsenal builds before 1.0 left out the end token and are not interchangeable with these; Vectors are not interchangeable with other embedding models; Text only.
        Parameters: 595M.
        Languages: mul.
        Licence: [apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-0.6B/tree/97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/qwen3-embedding-0.6b/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1203020554 | 1203020554 | 1.954 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/qwen3-embedding-0.6b/README.md)

        ## Qwen3-Embedding-8B (`qwen3-embedding-8b`)

        Create text search embeddings

        Turn text into vectors for semantic search and similarity.

        Good for: semantic search; similarity; clustering; classification.
        Limits: FP16 matches Qwen's own code at cosine 0.99999 on short texts (results/real-check-2026-11-02.md); inputs near the 32k context, reduced dimensions, int8 and int4 are not checked against it.
        Parameters: 7.57B.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/Qwen/Qwen3-Embedding-8B/tree/1d8ad4ca9b3dd8059ad90a75d4983776a23d44af).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/qwen3-embedding-8b/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 15146065500 | 15146065500 | unknown |
        | int8 | made from the 16-bit download | ~8046347296 | unknown |
        | int4 | made from the 16-bit download | ~4259830921 | unknown |

        [Model documentation](../models/qwen3-embedding-8b/README.md)

        ## Qwen3 Reranker 8B (`qwen3-reranker-8b`)

        Rank search results by relevance

        Rank candidate passages against a search query.

        Good for: reranking search results.
        Limits: Checked live on 20 Oct at FP16 and int8: the relevant document ranked first. int4 ran without errors; its ranking was not judged.
        Parameters: 8.19B.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/Qwen/Qwen3-Reranker-8B/tree/77d193c791ed757ca307ee72715aa132723da912).
        Tools: the generic window built from its operations.

        - `rerank`: `POST /arsenal/v1/models/qwen3-reranker-8b/ops/rerank`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 16388575361 | 16388575361 | 18.254 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~8706430660 | 9.692 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~4609286820 | 5.267 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/qwen3-reranker-8b/README.md)

        ## Qwen3 TTS 12Hz 1.7B Base (`qwen3-tts-base`)

        Speech in 10 languages, voice cloning

        Clones a voice from a short reference clip with its transcript. No voice design.

        Good for: text-to-speech.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 1.7B.
        Languages: zh, en, ja, ko, de, fr, ru, pt, es, it.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16/blob/a6eb4f68e4b056f1215157bb696209bc82a6db48/README.md).
        Tools: voice, sound.

        - `speak`: `POST /arsenal/v1/models/qwen3-tts-base/ops/speak`.
        - `voices`: `POST /arsenal/v1/models/qwen3-tts-base/ops/voices`.
        - `clone_voice`: `POST /arsenal/v1/models/qwen3-tts-base/ops/clone_voice`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 4544210194 | 4544210194 | 4.820 (m5-max, 2026-10-16, quickCheck) |
        | int8 | made from the 16-bit download | ~2414111665 | 4.395 (m5-max, 2026-10-16, quickCheck) |
        | int4 | made from the 16-bit download | ~1278059117 | 1.964 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/qwen3-tts-base/README.md)

        ## Qwen3 TTS 12Hz 1.7B VoiceDesign (`qwen3-tts-voicedesign`)

        Speech in 10 languages, voice design

        Makes a new voice from a written description and follows delivery instructions. No cloning.

        Good for: text-to-speech.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 1.7B.
        Languages: zh, en, ja, ko, de, fr, ru, pt, es, it.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/blob/7d3824abff87e49756bb0f83fb5411de75d160c4/README.md).
        Tools: voice, sound.

        - `speak`: `POST /arsenal/v1/models/qwen3-tts-voicedesign/ops/speak`.
        - `voices`: `POST /arsenal/v1/models/qwen3-tts-voicedesign/ops/voices`.
        - `design_voice`: `POST /arsenal/v1/models/qwen3-tts-voicedesign/ops/design_voice`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 4520192405 | 4520192405 | 4.755 (m5-max, 2026-10-16, quickCheck) |
        | int8 | made from the 16-bit download | ~2401352215 | 4.382 (m5-max, 2026-10-16, quickCheck) |
        | int4 | made from the 16-bit download | ~1271304113 | 1.877 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/qwen3-tts-voicedesign/README.md)

        ## RE-USE (`re-use`)

        Speech cleanup

        Universal speech cleanup: noise, reverb, clipping and codec damage.

        Good for: audio-to-audio; denoise.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 9.6M.
        Languages: und.
        Licence: [OpenMDW-1.1](https://huggingface.co/nvidia/NVIDIA-NemotronLabs-RE-USE/blob/31f82f2efad1095034492f236e8dac9239ba713b/LICENSE).
        Tools: the generic window built from its operations.

        - `enhance`: `POST /arsenal/v1/models/re-use/ops/enhance`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 38596151 | ~19298075 | 1.247 (m5-max, 2026-10-15, quickCheck) |

        [Model documentation](../models/re-use/README.md)

        ## Real-ESRGAN x4plus (`real-esrgan-x4plus`)

        Upscale a photo 4x

        Enlarges a photo 4x, sharpening edges and cleaning blur, noise and JPEG artefacts as it goes.

        Good for: Small or soft photos and web images that need to print or display larger; Removing JPEG blocking, mild blur and noise while enlarging; Landscapes, buildings, objects and illustrations.
        Limits: Fine natural texture (grass, foliage, hair) can come out smooth or painterly; it invents plausible detail, not the real one; Faces are not specially restored (the maker pairs it with GFPGAN for that), and small text can be garbled; Anime and line art do better with the maker's separate anime model; The network is fixed at 4x; scale 2 and target_height resize its 4x result with a high-quality filter. Input at most 4 megapixels and 4096 pixels per side; output at most 64 megapixels and 16384 pixels per side; Transparency is flattened; the output is an opaque PNG; It has no language input or output, so Polish does not apply.
        Parameters: 16.7M.
        Languages: mul.
        Licence: [BSD-3-Clause](https://huggingface.co/Comfy-Org/Real-ESRGAN/blob/5fd49b7b278836f48af63ecd314d0f98ab336105/README.md).
        Tools: the generic window built from its operations.

        - `upscale_image`: `POST /arsenal/v1/models/real-esrgan-x4plus/ops/upscale_image`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 66857836 | ~33428918 | 2.792 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/real-esrgan-x4plus/README.md)

        ## RF-DETR Large (`rf-detr-large`)

        Find 80 types of objects

        Find 80 types of objects, more accurately than Nano. Native MLX inference in FP16.

        Good for: Fixed COCO object detection where small or distant objects matter more than speed; Boxes to crop, count or feed into a segmentation model.
        Limits: Only 80 COCO classes; no open-vocabulary detection; Matches the maker's Transformers code on Apple GPU in FP16: the same five detections on the COCO two-cats photo, scores within 0.005 and boxes within 0.0003 (2 Nov 2026); No masks or keypoints from this checkpoint.
        Parameters: 33.9M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/Roboflow/rf-detr-large/blob/f62f7dd5252b61097cbace33886045816dadbde9/README.md).
        Tools: the generic window built from its operations.

        - `detect`: `POST /arsenal/v1/models/rf-detr-large/ops/detect`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 135801147 | ~67900573 | unknown |

        [Model documentation](../models/rf-detr-large/README.md)

        ## RF-DETR Nano (`rf-detr-nano`)

        Find 80 types of objects

        Find 80 types of objects. Native MLX inference in FP16.

        Good for: Fixed COCO object detection.
        Limits: Only 80 COCO classes; no open-vocabulary detection; Checked live at FP16 on 19 Oct: found both cats and both remotes in the COCO two-cats photo.
        Parameters: 30.5M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/Roboflow/rf-detr-nano/blob/9f201577d9415f0378084748d27e68586e6b3600/README.md).
        Tools: the generic window built from its operations.

        - `detect`: `POST /arsenal/v1/models/rf-detr-nano/ops/detect`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 121932107 | ~60966053 | 0.387 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/rf-detr-nano/README.md)

        ## RIFE 4.25 (`rife-4.25`)

        Video frame interpolation

        Create in-between frames from a pair of images or increase a video frame rate.

        Good for: Frame-rate conversion; Single in-between images.
        Limits: Checked live at FP16 on 19 Oct: a 24-pixel pan was interpolated; a 300-pixel jump of a blank square failed.
        Parameters: 5.7M.
        Languages: und.
        Licence: [MIT](https://huggingface.co/mlx-community/RIFE-4.25/tree/6b650eaa5664aab280119511fe944861491aa15e).
        Tools: the generic window built from its operations.

        - `interpolate_video`: `POST /arsenal/v1/models/rife-4.25/ops/interpolate_video`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 22674665 | ~11337332 | 0.601 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/rife-4.25/README.md)

        ## SAM Audio Large (`sam-audio-large`)

        Isolate a sound by description

        Isolate a sound by description using a native Swift MLX worker.

        Good for: Isolate a sound by description.
        Limits: Text and time prompts; no visual encoder in this port; Anchors support ten-second clips; unanchored audio uses overlapping windows; Checked live on 20 Oct at FP16, int8 and int4 (on the FP16 re-upload): speech isolated cleanly from a mix with a beep; Get downloads Meta's 14.9 GB original checkpoint and converts it to about 6.3 GB at FP16 (with the T5 text encoder), then deletes the original. Get needs about 45 GB free when it starts.
        Parameters: 3151M.
        Languages: en.
        Licence: [SAM License](https://huggingface.co/facebook/sam-audio-large/blob/5f2cd3a9471a08c7282c06036be6893e18de8b70/LICENSE). Meta's SAM License: you accept it on Hugging Face and Meta approves each request; Arsenal then downloads from facebook/sam-audio-large with your Hugging Face token. No ITAR or other trade-controlled uses; anything redistributed carries the same licence; published research credits SAM. Gated: accept Meta's terms on Hugging Face and add a read token in Settings.
        Tools: the generic window built from its operations.

        - `separate_audio`: `POST /arsenal/v1/models/sam-audio-large/ops/separate_audio`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 14861358462 | ~7430679231 | 10.737 (m5-max, 2026-10-22, quickCheck) |
        | int8 | made from the 16-bit download | ~3947548341 | 8.432 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~2089878533 | 6.749 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/sam-audio-large/README.md)

        ## SAM 2.1 Hiera Large (`sam2.1-hiera-large`)

        Cut out an object by click or box

        Cuts out the object you click or box in a photo and returns its mask.

        Good for: Cutting one object out of a photo from a click or a box; Several boxes at once: one mask per box from one image encoding; Three alternative masks for an ambiguous click, each with a predicted quality score.
        Limits: Needs a click or box: it does not find objects by name or segment everything at once; A single click can be ambiguous (a window or the whole cab); a box or a few clicks is more reliable; Image path only: the video memory and tracking modules are not loaded, so it segments one photo at a time; Masks are predicted at 256x256 and upsampled, so fine edges such as hair are soft; No language input or output; Parity with the maker's transformers Sam2Model on MPS FP16: mask logits cosine 0.99990 or better on six photo and prompt pairs.
        Parameters: 224.4M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/facebook/sam2.1-hiera-large/blob/665f8e2ad61cf5f53d65644ff27c8ee525124610/README.md).
        Tools: the generic window built from its operations.

        - `segment`: `POST /arsenal/v1/models/sam2.1-hiera-large/ops/segment`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 897904604 | ~448952302 | 3.157 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/sam2.1-hiera-large/README.md)

        ## SAM 2.1 Hiera Tiny (`sam2.1-hiera-tiny`)

        Cut out an object by click or box

        Cuts out the object you click or box in a photo and returns its mask.

        Good for: Cutting one object out of a photo from a click or a box; Several boxes at once: one mask per box from one image encoding; Three alternative masks for an ambiguous click, each with a predicted quality score.
        Limits: Needs a click or box: it does not find objects by name or segment everything at once; A single click can be ambiguous (a window or the whole cab); a box or a few clicks is more reliable; Image path only: the video memory and tracking modules are not loaded, so it segments one photo at a time; Masks are predicted at 256x256 and upsampled, so fine edges such as hair are soft; No language input or output; Parity with the maker's transformers Sam2Model on MPS FP16: mask logits cosine 0.99991 or better on six photo and prompt pairs.
        Parameters: 39.0M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/facebook/sam2.1-hiera-tiny/blob/de431c4043854a71d8101e17995dfe596bf101a5/README.md).
        Tools: the generic window built from its operations.

        - `segment`: `POST /arsenal/v1/models/sam2.1-hiera-tiny/ops/segment`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 155915242 | ~77957621 | 1.450 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/sam2.1-hiera-tiny/README.md)

        ## SAM 3.1 (`sam3.1`)

        Select objects and track video

        Select objects and track video. Native MLX inference in FP16.

        Good for: Prompted image masks and object tracking.
        Limits: Four sessions; 16 tracked objects; at most 128 frames and 256 PNG masks per call; upright videos only.
        Parameters: 870M.
        Languages: en.
        Licence: [SAM License](https://huggingface.co/facebook/sam3.1/blob/daa63191845a41281374e725f4c9e51c7a824460/LICENSE). Meta's SAM License: you accept it on Hugging Face (Meta approves each request) and Arsenal downloads from facebook/sam3.1 with your token. No ITAR or other trade-controlled uses; anything redistributed carries the same licence; published research credits SAM. Gated: accept Meta's terms on Hugging Face and add a read token in Settings.
        Tools: the generic window built from its operations.

        - `segment`: `POST /arsenal/v1/models/sam3.1/ops/segment`.
        - `segment_session`: `POST /arsenal/v1/models/sam3.1/ops/segment_session`.
        - `video_session`: `POST /arsenal/v1/models/sam3.1/ops/video_session`.
        - `track_video`: `POST /arsenal/v1/models/sam3.1/ops/track_video`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 3502781560 | ~1751390780 | unknown |

        [Model documentation](../models/sam3.1/README.md)

        ## SeedVR2 3B (`seedvr2-3b`)

        Image and video restoration

        Restore and enlarge images or video with one diffusion step.

        Good for: Degraded photos; Video detail restoration.
        Limits: 16 megapixels per output frame; at most 8192 pixels per side; Video restores 5- to 21-frame chunks with one noise draw per clip; chunk boundaries need trained-weight review; The VAE works in overlapping spatial tiles, which upstream does not use; Job files are limited by the host to 4 GiB; FP16 and derived int8/int4 output quality awaits trained weights.
        Parameters: 3B.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/mlx-community/SeedVR2-3B-mlx/tree/46f851fe23b2a9fbefee6419b94e4fdf07b92965).
        Tools: the generic window built from its operations.

        - `upscale_image`: `POST /arsenal/v1/models/seedvr2-3b/ops/upscale_image`.
        - `upscale_video`: `POST /arsenal/v1/models/seedvr2-3b/ops/upscale_video`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 8438508899 | 8438508899 | 38.655 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~4482957852 | unknown |
        | int4 | made from the 16-bit download | ~2373330627 | unknown |

        [Model documentation](../models/seedvr2-3b/README.md)

        ## Smart Turn v3 (`smart-turn-v3`)

        Detect the end of a spoken turn

        Detect the end of a spoken turn using a native Swift MLX worker.

        Good for: Detect the end of a spoken turn.
        Limits: Uses the final eight seconds of audio; Checked live at FP16 on 19 Oct: end-of-turn probability 0.98 for a complete sentence, 0.02 for a sentence cut mid-word, and 0.97 for a phrase cut after a whole word.
        Parameters: 8M.
        Languages: und.
        Licence: [BSD-2-Clause](https://huggingface.co/mlx-community/smart-turn-v3/tree/74af1538ebace37b0fcc667a6a8a3d902d73eee0).
        Tools: the generic window built from its operations.

        - `end_of_turn`: `POST /arsenal/v1/models/smart-turn-v3/ops/end_of_turn`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 32010367 | ~16005183 | 0.215 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/smart-turn-v3/README.md)

        ## Streaming Sortformer Diarizer 4spk v2 (`sortformer-4spk-v2`)

        Who spoke when, live, up to 4

        Find who spoke when, live or on a recording, for up to four speakers, using a native Swift MLX worker.

        Good for: Meetings, calls and interviews with up to four speakers; Live diarization sessions with latency from 0.32 s to 30 s; Long recordings without clustering; Overlapping speech (each speaker has its own activity track); Per-speaker activity probabilities every 80 ms.
        Limits: At most four speakers; accuracy drops with five or more; Trained mostly on English; other languages, including Polish, are untested here; Speakers are numbered by arrival, not identified; no speaker embeddings; Up to one hour per recording, ten minutes per live session; Noisy or out-of-domain audio degrades accuracy.
        Parameters: 117M.
        Languages: en.
        Licence: [Creative Commons Attribution 4.0](https://huggingface.co/nvidia/diar_streaming_sortformer_4spk-v2/blob/84edd514b8ef68004c10086918cd62f2148cbd59/README.md).
        Tools: the generic window built from its operations.

        - `diarize`: `POST /arsenal/v1/models/sortformer-4spk-v2/ops/diarize`.
        - `diarize_session`: `POST /arsenal/v1/models/sortformer-4spk-v2/ops/diarize_session`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 471367680 | ~235683840 | unknown |

        [Model documentation](../models/sortformer-4spk-v2/README.md)

        ## Stable Audio 3 Medium (`stable-audio-3-medium`)

        Sound and music generation

        Prompted stereo sound effects and instrumental music.

        Good for: text-to-audio.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 1.45B.
        Languages: en.
        Licence: [Stability AI Community](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/da6edc54ddba10bfd79a077102ded687f80e882b/LICENSE.md). Commercial use only after registering with Stability AI and while you and your affiliates earn under US$1M a year; above that an enterprise licence is needed.
        Tools: sound.

        - `generate_sound`: `POST /arsenal/v1/models/stable-audio-3-medium/ops/generate_sound`.
        - `generate_music`: `POST /arsenal/v1/models/stable-audio-3-medium/ops/generate_music`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 5179078480 | 5179078480 | 5.510 (m5-max, 2026-10-16, quickCheck) |
        | int8 | made from the 16-bit download | ~2751385442 | 3.401 (m5-max, 2026-10-16, quickCheck) |
        | int4 | made from the 16-bit download | ~1456615822 | 2.201 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/stable-audio-3-medium/README.md)

        ## Stable Audio 3 Small SFX (`stable-audio-3-small-sfx`)

        Sound effects

        Prompted stereo sound effects.

        Good for: text-to-audio.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 459M.
        Languages: en.
        Licence: [Stability AI Community](https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/da6edc54ddba10bfd79a077102ded687f80e882b/LICENSE.md). Commercial use only after registering with Stability AI and while you and your affiliates earn under US$1M a year; above that an enterprise licence is needed.
        Tools: sound.

        - `generate_sound`: `POST /arsenal/v1/models/stable-audio-3-small-sfx/ops/generate_sound`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1704750192 | 1704750192 | 2.529 (m5-max, 2026-10-16, quickCheck) |
        | int8 | made from the 16-bit download | ~905648539 | 1.575 (m5-max, 2026-10-16, quickCheck) |
        | int4 | made from the 16-bit download | ~479460991 | 1.082 (m5-max, 2026-10-16, quickCheck) |

        [Model documentation](../models/stable-audio-3-small-sfx/README.md)

        ## T0 Alpha (`t0-alpha`)

        Forecast numeric time series

        Forecasts the next values of one numeric series as a range of likely outcomes.

        Good for: Probabilistic forecasts of one numeric series, such as load, sales or counts, up to 1024 steps ahead in one pass; Clean seasonal series: a sine with period 24 is forecast almost exactly in a local check (RMSE 0.01).
        Limits: Univariate here: one series in, no covariates and no multivariate input; Horizons above 1024 steps need the maker's autoregressive rollout, which this runner does not port; Returns only the trained quantile levels 0.1, 0.25, 0.5, 0.75 and 0.9; other levels are not interpolated; Under-extrapolates trends: on a linear ramp the 24-step median ended about 0.5 low, and on a sine plus trend about 0.7 low at its peak; Matches the maker's tfc-t0 code on Apple GPU in fp16 (cosine 0.99999996, largest gap 0.68 on forecasts up to 740); the maker's benchmark figures are not reproduced here; Reads and writes no language, so the Polish question does not apply.
        Parameters: 101.6M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/theforecastingcompany/t0-alpha/blob/741927847fe02367624749bb9290b2f09f128e7d/README.md).
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/t0-alpha/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 406601742 | ~203300871 | unknown |

        [Model documentation](../models/t0-alpha/README.md)

        ## T0 Gamma (`t0-gamma`)

        Forecast series with covariates

        Forecasts the next values of a numeric series as a range of likely outcomes, optionally using related series and known future inputs.

        Good for: Probabilistic forecasts of one numeric series, such as load, sales or counts, up to 1024 steps ahead in one pass, at any quantile levels; Forecasts that should use known future inputs (calendar features, holidays, promotions, weather forecasts) given as covariates over the past and the horizon; Several related series forecast jointly: past covariates attend to the target and get their own forecasts; Series with gaps: missing observations are sent as null.
        Limits: Horizons above 1024 steps need the maker's autoregressive rollout, which this runner does not port; One target series per request, with up to 31 past and 31 known-future covariates; batching several independent series in one call is not supported; A reply holds at most 640,000 forecast numbers (series x horizon x levels with the median, 16 MiB as JSON): 99 levels at horizon 1024 allow 6 series; larger asks are refused before the model runs; A request body is at most 1 MiB of JSON over HTTP and MCP (about 100,000 numbers in all), so the 8192-step context with every covariate only fits with short numbers; Values must lie inside the float32 range (about 3.4e38); quantile levels must not round to 0 or 1 in float32, nor with symmetric_averaging their mirrors 1 - q (so not below about 3e-8); Missing values inside a series are supported; trimming a leading missing stretch (the maker's trim_leading_missing) and explicit padding masks are not; Historical-only covariates are sent as past_covariates and forecast jointly, as the maker's array API does; the maker's separate HISTORICAL role is not exposed; Matches the maker's tfc-t0 code on Apple GPU in fp16 on fixed inputs (see log/003-parity.md); the maker's benchmark figures are not reproduced here; Reads and writes no language, so the Polish question does not apply.
        Parameters: 255.6M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/theforecastingcompany/t0-gamma/blob/e08aee083778f1f32da4974c0a34276776a35c52/LICENSE).
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/t0-gamma/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1022493452 | ~511246726 | unknown |

        [Model documentation](../models/t0-gamma/README.md)

        ## TeleOCR (`teleocr`)

        Read text, tables, formulas and code

        Read text, tables, formulas and code.

        Good for: Read text, tables, formulas and code.
        Limits: Checked live on 20 Oct at FP16, int8 and int4: three text lines and a three-column table read exactly; Decoding follows the maker client: greedy, presence and frequency penalties on recognition calls, and its 100-token no-repeat n-gram block on every call.
        Parameters: 1.4B.
        Languages: zh, en.
        Licence: [Apache License 2.0](https://huggingface.co/XingChen-AGI/TeleOCR/tree/70a19ec6a3b036311af19ae5e762e650f41a5980).
        Tools: the generic window built from its operations.

        - `read_document`: `POST /arsenal/v1/models/teleocr/ops/read_document`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 2841656459 | 2841656459 | 5.680 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~1509629993 | 4.291 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~799215879 | 3.613 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/teleocr/README.md)

        ## TIGER DnR (`tiger-dnr`)

        Split dialogue, music and effects

        Splits a film or video soundtrack into dialogue, music and sound effects.

        Good for: Pulling the dialogue out of a film, TV or game clip for transcription or dubbing; Removing the music or the effects from a soundtrack; Isolating background music from a video.
        Limits: Three fixed stems: dialog, effects and music; Works at 44.1 kHz mono: other rates are resampled and stereo is mixed to mono before separation; the stems come back as 44.1 kHz mono; Runs the makers' long-audio path (12-second windows every 4 seconds, averaged), so every second of audio is processed three times by each requested stem's network; input is limited to 5 minutes; Sung vocals in music can be split between the dialog and music stems; It does not read or write language; the training data is English-language film audio and other languages, Polish included, were not tested.
        Parameters: 4.2M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/JusperLee/TIGER-DnR/blob/b7a59560bbca10febbcd46fb01600f868e587f57/README.md).
        Tools: the generic window built from its operations.

        - `separate_audio`: `POST /arsenal/v1/models/tiger-dnr/ops/separate_audio`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 17130818 | ~8565409 | 2.405 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/tiger-dnr/README.md)

        ## TIGER Speech (`tiger-speech`)

        Separate two overlapping speakers

        Splits a recording of two people talking over each other into one clean track per speaker.

        Good for: Two people talking at the same time in a meeting, interview or call recording; Speech with background noise and room echo (it was trained on noisy, reverberant mixtures); Preparing overlapped speech for transcription, one speaker per track.
        Limits: Always returns exactly two tracks; with one speaker the second track is mostly residue, and three or more speakers are not separated; Which speaker lands in speaker1 or speaker2 is arbitrary and can differ between files; Works at 16 kHz: higher-rate input is resampled and the tracks come back as 16 kHz mono; stereo input is mixed to mono; One pass over the whole file, so input is limited to 60 seconds; peak GPU memory was about 2.5 GB for a 60-second file; Trained on speech only: music or sound effects are not separated into their own stem; It does not read or write language; it was trained on English speech and other languages, Polish included, were not tested.
        Parameters: 0.8M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/JusperLee/TIGER-speech/blob/f0340340b2d9bbf72074edf8c076dcab59a10ba2/README.md).
        Tools: the generic window built from its operations.

        - `separate_audio`: `POST /arsenal/v1/models/tiger-speech/ops/separate_audio`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 3367601 | ~1683800 | 2.480 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/tiger-speech/README.md)

        ## TimesFM 2.5 200M (`timesfm-2.5-200m`)

        Forecast numeric time series

        Forecasts the next values of one numeric series as a range of likely outcomes.

        Good for: Probabilistic zero-shot forecasts of one numeric series, such as sales, traffic or counts, up to 1024 steps; Long histories: it reads up to the last 8192 values here (the model allows 16k with the horizon); Non-negative series stay non-negative, and quantiles never cross; Seasonal series: on monthly airline passengers it continues the summer peak and winter dip.
        Limits: Univariate here: one series in; the maker's covariate regression (XReg) is not ported; Horizon at most 1024 steps (the continuous quantile head's length); past 128 steps it decodes autoregressively on its own median; Uses the model card's settings, fixed: input normalisation, flip invariance (two passes), non-negativity when the history is non-negative, the continuous quantile head and quantile-crossing repair; Returns the quantile levels 0.1 to 0.9 in steps of 0.1; missing values are not accepted; Reads and writes no language, so the Polish question does not apply; Parity with the maker code (timesfm 3.0.2, MPS fp16 autocast): cosine 0.99999996, max abs diff 0.53 on forecasts up to 697 (airline series, 24 steps); cosine 0.999999 over 1024 steps.
        Parameters: 231.3M.
        Languages: und.
        Licence: [Apache-2.0](https://huggingface.co/google/timesfm-2.5-200m-pytorch/blob/1d952420fba87f3c6dee4f240de0f1a0fbc790e3/README.md).
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/timesfm-2.5-200m/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 925181579 | ~462590789 | unknown |

        [Model documentation](../models/timesfm-2.5-200m/README.md)

        ## Timestral 209M (`timestral-209m`)

        Forecast series with covariates

        Forecasts the next values of a numeric series as a range of likely outcomes, optionally using related series and known future inputs.

        Good for: Probabilistic forecasts of a numeric series such as load, sales, traffic or counts, with 21 quantile levels from 0.01 to 0.99; Long histories: up to 32,768 past steps, and horizons up to 4,096 steps (1,024 per pass, longer ones block by block); Forecasts that should use known future inputs (calendar features, holidays, promotions, prices, weather forecasts), numeric or categorical; Several related series forecast jointly (co_targets), plus covariates only known in the past; Series with gaps: missing observations are sent as null.
        Limits: One series (with its co-targets and covariates) per request; the maker's batching of independent series in one call is not exposed; At most 32 series in all per request, and 31 in each covariate list; A reply holds at most 640,000 forecast numbers (forecast series x horizon x levels with the median, 16 MiB as JSON): at horizon 4,096 with all 21 levels, the target and 6 co-targets; larger asks are refused before the model runs; A request body is at most 1 MiB of JSON over HTTP and MCP (about 100,000 numbers in all), so the 32,768-step context fits for one or two series, not for 32; Returns only the 21 trained quantile levels; no interpolation between them; Matches the maker's code on fixed inputs on the Apple GPU in FP16 (see log); the maker's benchmark figures are not reproduced here; Reads and writes no language, so the Polish question does not apply.
        Parameters: 209.3M.
        Languages: und.
        Licence: [Modified MIT License (Mistral AI)](https://huggingface.co/mistralai/Timestral-209M/blob/9874d252ab0aae16f7f7c999f98f98d6ad7d8365/LICENSE). No rights under the licence for a company whose global consolidated monthly revenue (or its employer's) exceeded US$20 million in the preceding month, for the model and any derivative; Mistral AI sells a commercial licence. Copies must keep the attribution notice "2026 - Mistral AI" and the licence. See the pinned LICENSE.
        Tools: the generic window built from its operations.

        - `forecast`: `POST /arsenal/v1/models/timestral-209m/ops/forecast`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 837272778 | ~418636389 | unknown |

        [Model documentation](../models/timestral-209m/README.md)

        ## TIPSv2 L/14 (`tipsv2-l14`)

        Search photos with text or images

        Search photos with text or images. Native MLX inference in FP16.

        Good for: Photo retrieval and zero-shot image labels.
        Limits: English text; 64 token text limit; 256 inputs per batch; no trained DPT heads.
        Parameters: 488M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/google/tipsv2-l14/blob/52847a71eb02a3082c370fbcd98cd799687bd4d0/README.md).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/tipsv2-l14/ops/embed`.
        - `detect`: `POST /arsenal/v1/models/tipsv2-l14/ops/detect`.
        - `segment`: `POST /arsenal/v1/models/tipsv2-l14/ops/segment`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1952554374 | ~976277187 | 1.847 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/tipsv2-l14/README.md)

        ## TRELLIS.2 4B (`trellis2-4b`)

        Image to textured 3D

        Generate a textured 3D mesh from an image.

        Good for: Single-image 3D with base color, metallic, roughness and opacity; 512, 1024 and 1536 voxel resolution.
        Limits: Requires BiRefNet added with installed weights for background removal; Trained-weight parity and FP16/int8/int4 quality await the approved live trial; Mesh retexturing is not exposed by the current image_to_3d contract; 4 GiB per-job artifact limit; Precision exception: the DINOv3 encoder keeps its residual stream in float32, because in FP16 it overflows (about 153,000 against FP16's 65,504) and the condition turns to NaN.
        Parameters: 4B.
        Languages: und.
        Licence: [MIT](https://huggingface.co/microsoft/TRELLIS.2-4B/tree/af44b45f2e35a493886929c6d786e563ec68364d). TRELLIS.2 itself is MIT. Its DINOv3 image encoder is under Meta's DINOv3 License, accepted on Hugging Face (Meta approves each request): Arsenal downloads from facebook/dinov3-vitl16-pretrain-lvd1689m with your token. No ITAR or other trade-controlled uses; anything redistributed carries that licence. Gated: accept Meta's terms on Hugging Face and add a read token in Settings.
        Tools: the generic window built from its operations.

        - `image_to_3d`: `POST /arsenal/v1/models/trellis2-4b/ops/image_to_3d`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 14819878398 | 14819878398 | 26.638 (m5-max, 2026-10-22, quickCheck) |
        | int8 | made from the 16-bit download | ~7873060398 | 20.196 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~4168090799 | unknown |

        [Model documentation](../models/trellis2-4b/README.md)

        ## TripoSR (`triposr`)

        Make a 3D model from one photo

        Turns one photo of an object into a coloured 3D mesh (GLB).

        Good for: A single object cut out from its background: furniture, toys, figurines, products, food; Quick draft 3D assets and previews from product shots or concept art; Coloured meshes to view in Quick Look, a browser or a 3D editor.
        Limits: Needs a cut-out object: give a PNG with a transparent background (Arsenal's Remove Background makes one). An opaque photo is used as it is, which only works when the object already sits on a plain grey background and fills most of the frame; the makers' automatic background removal (rembg) is not included; One object per image; scenes, people in context and very thin parts (wires, antennas) come out poorly or blobby; The hidden back is guessed, so it is smoother and less detailed than the visible side; Colours are per vertex, not a texture map; texture_size, simplify and seed are accepted but have no effect. resolution sets the marching-cubes grid, 256 to 512 per side (default 256); 512 is this port's limit, larger values are refused; It has no language input or output, so Polish does not apply.
        Parameters: 419.3M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/stabilityai/TripoSR/blob/4e22858523f295f3c2b3852c39c90f399ea0e7d3/README.md).
        Tools: the generic window built from its operations.

        - `image_to_3d`: `POST /arsenal/v1/models/triposr/ops/image_to_3d`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 1677170968 | ~838585484 | 2.555 (m5-max, 2026-11-01, quickCheck) |

        [Model documentation](../models/triposr/README.md)

        ## UI Venus 1.5 2B (`ui-venus-1.5-2b`)

        Locate screen elements from instructions

        Locate screen elements from instructions.

        Good for: Locate screen elements from instructions.
        Limits: Checked live on 20 Oct at FP16, int8 and int4: both test clicks landed inside their targets.
        Parameters: 2.4B.
        Languages: en.
        Licence: [Apache License 2.0](https://huggingface.co/mlx-community/UI-Venus-1.5-2B-bf16/tree/d5286cd3d2580c331e2f57a976987744cd65e51d).
        Tools: the generic window built from its operations.

        - `ground_screen`: `POST /arsenal/v1/models/ui-venus-1.5-2b/ops/ground_screen`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 4888957599 | 4888957599 | 12.885 (m5-max, 2026-11-01, quickCheck) |
        | int8 | made from the 16-bit download | ~2597258724 | 5.901 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~1375019324 | 4.375 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/ui-venus-1.5-2b/README.md)

        ## ULTRA 50g (`ultra-50g`)

        Predict missing knowledge-graph links

        Fills in missing facts in a knowledge graph you supply as (head, relation, tail) triples: give an entity and a relation, get the likeliest missing ends ranked.

        Good for: Knowledge-graph completion on your own graph without training: (Italy, located_in, ?) on a small geography graph ranks Europe first; Ranking candidate links in any multi-relational data: people and organisations, products and categories, genes and diseases; Finding missing heads as well as tails: give tail instead of head.
        Limits: Names carry no meaning to the model: it reasons only from the graph's structure, so an entity or relation with few triples gets weak predictions, and tiny graphs give rough answers; The query entity and relation must appear in the triples; an unseen relation cannot be asked about; The whole graph is sent with every request (at most 1,000,000 triples, 1,000,000 distinct entities and 4,096 distinct relations) and rebuilt each time; Scores rank candidates for one query; the probabilities are the model's sigmoid and are not calibrated across graphs. Known links are left out of the list unless exclude_known is false; Reads and writes no human language (names are opaque labels), so the Polish question does not apply; Parity with the maker's ultra package (fused rspmm path, MPS, float32 since its code cannot run float16) checked: see log/003-parity.md.
        Parameters: 0.2M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/mgalkin/ultra_50g/blob/d46107d8415549e7c7386d26903ad11c9a8de4fd/README.md).
        Tools: the generic window built from its operations.

        - `predict_links`: `POST /arsenal/v1/models/ultra-50g/ops/predict_links`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 684637 | ~342318 | unknown |

        [Model documentation](../models/ultra-50g/README.md)

        ## UniPASE (`unipase`)

        Universal speech enhancement

        Universal speech enhancement: noise, reverb, clipping, lost packets and narrow bandwidth, up to 48 kHz, using a native Swift MLX worker.

        Good for: Cleaning recorded speech with several problems at once: background noise, reverb, clipping, codec damage and a narrow band; Restoring old, phone or low-rate recordings to full 48 kHz bandwidth (upscale); Filling short dropouts of digital silence in calls and streams (packet-loss concealment); English, Chinese, Spanish, French and German speech, the languages its makers name; Long recordings: past 20 s it works in 8 s segments, so memory stays flat.
        Limits: Speech only: music and other sounds are treated as noise; Generative: it re-synthesises the voice, so on very damaged input it can change details of the speech rather than only remove noise; the makers designed it to keep this low; Packet-loss concealment only sees dropouts that are digital silence (at least 99% of a 20 ms packet below 1e-7); quiet but non-zero gaps are not filled; Each channel is enhanced on its own as mono; stereo is not modelled jointly; Up to 10 minutes per clip; at least 0.1 s; Output is not streamed while it is made; the whole clip is enhanced first.
        Parameters: 546.5M.
        Languages: en, zh, es, fr, de.
        Licence: [Apache License 2.0](https://huggingface.co/Xiaobin-Rong/unipase/blob/f0b4d4c4411fe08fc2dddbf2d9f33260c27ac4a0/README.md).
        Tools: the generic window built from its operations.

        - `enhance`: `POST /arsenal/v1/models/unipase/ops/enhance`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 2186263833 | ~1093131916 | unknown |

        [Model documentation](../models/unipase/README.md)

        ## UVDoc (`uvdoc`)

        Flatten a photographed page

        Flattens a photo of a curved or tilted page into a straight, readable scan

        Good for: Phone photos of book pages and curled or folded paper before OCR; Straightening tilted or perspective-skewed documents.
        Limits: Reads no text, so language does not matter (Polish pages work like any other); The output keeps the photo's size and aspect ratio rather than the page's, and background around the page can stay in or be stretched; Works from a 488 by 712 view of the photo, so very fine local wrinkles are not corrected; Expects one page or one open book spread filling most of the frame.
        Parameters: 7.9M.
        Languages: mul.
        Licence: [Apache-2.0](https://huggingface.co/PaddlePaddle/UVDoc_safetensors/blob/bfa6aad03b549c7e2cf89cefd2fb02ba3bef2ae4/README.md).
        Tools: the generic window built from its operations.

        - `dewarp`: `POST /arsenal/v1/models/uvdoc/ops/dewarp`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 31669041 | ~15834520 | unknown |

        [Model documentation](../models/uvdoc/README.md)

        ## ViTPose Base Simple (`vitpose-base-simple`)

        Find body keypoints

        Finds 17 body keypoints, from nose to ankles, for the person in a photo.

        Good for: One person filling most of a photo: nose, eyes, ears, shoulders, elbows, wrists, hips, knees and ankles; Posture or exercise-form checks on a cropped person.
        Limits: No person detector in this port: the whole photo is treated as one person. In a wide room shot with a small person, the legs land on the furniture; cropped to the person, the skeleton fit and the scores went from 0.16-0.61 to 0.63-0.94; Several people in one photo are not separated. The card's own example runs a separate detector first; Weak joints score below about 0.4. Occluded people are the card's hardest case (OCHuman); It has no language input or output, so Polish does not apply; Parity with the maker's transformers code on MPS FP16: joints within 2.2 pixels on a 640x425 photo.
        Parameters: 85.9M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/usyd-community/vitpose-base-simple/blob/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45/README.md).
        Tools: the generic window built from its operations.

        - `estimate_pose`: `POST /arsenal/v1/models/vitpose-base-simple/ops/estimate_pose`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 343675237 | ~171837618 | unknown |

        [Model documentation](../models/vitpose-base-simple/README.md)

        ## V-JEPA 2 ViT-L (`vjepa2-vitl-fpc64-256`)

        Embed video clips

        Turns a video clip, a sequence of frames or a still image into a vector of numbers for video search, retrieval, clustering or a classifier of your own.

        Good for: Video similarity and retrieval: clips of the same activity land closer together (at 64 frames: two archery clips 0.73, archery vs bowling 0.65 to 0.66); Features for a small action or event classifier trained on top; Time-resolved features: one embedding per pair of frames with per_step.
        Limits: Embeddings only: this pretrained checkpoint has no classifier, so it names no actions (Meta's separately fine-tuned SSv2 and Diving-48 heads are not served), and the JEPA predictor is not served; At most 64 frames per request, centre-cropped to 256 x 256: whatever lies outside the centre square of a wide or tall frame is not seen, and long videos are sampled sparsely (use start and end to pick a window); Video sound is ignored; Reads and writes no human language, so the Polish question does not apply; Parity with transformers VJEPA2Model (MPS, float16) checked on the same weights: see log/003-parity.md.
        Parameters: 326.0M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/mlx-community/V-JEPA2-vitl-fpc64-256/blob/8eec39e665ffce8ac09174ea3ab2f2d65940ef5f/README.md).
        Tools: the generic window built from its operations.

        - `embed`: `POST /arsenal/v1/models/vjepa2-vitl-fpc64-256/ops/embed`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 652004836 | 652004836 | unknown |

        [Model documentation](../models/vjepa2-vitl-fpc64-256/README.md)

        ## Voice Gender Classifier (ECAPA-TDNN) (`voice-gender-ecapa`)

        Tell male or female voice

        Labels one speaker in a recording as male or female, with probabilities.

        Good for: One speaker per clip, a few seconds or longer; Speech like its training data: VoxCeleb celebrity interview clips.
        Limits: Two labels only, male and female, as the VoxCeleb training data labels them; the card warns the data may not represent the global population; The 98.7% figure is the card's claim, not measured here; Clips under 0.03 s are refused; silence scores male at 0.81, so check the label against probabilities; The card names no languages; none were tested, Polish included; Parity with the maker code not run.
        Parameters: 15.5M.
        Languages: mul.
        Licence: [MIT](https://huggingface.co/JaesungHuh/voice-gender-classifier/blob/db1222153bd60337e900be22add7af180452adc0/README.md).
        Tools: the generic window built from its operations.

        - `classify`: `POST /arsenal/v1/models/voice-gender-ecapa/ops/classify`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 61907527 | ~30953763 | unknown |

        [Model documentation](../models/voice-gender-ecapa/README.md)

        ## Von · 1.1 (`von-1.1`)

        Option scoring, order-sensitive

        Earlier Von: answers can change when options are reordered; 2048-token context; English only.

        Good for: yes-no; choice; score.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 395M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/wfzyx/von/blob/d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd/README.md).
        Tools: the generic window built from its operations.

        - `decide`: `POST /arsenal/v1/models/von-1.1/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/von-1.1/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 1584907147 | ~792453573 | 1.177 (m5-max, 2026-10-15, quickCheck) |

        [Model documentation](../models/von-1.1/README.md)

        ## Von · 1.2 (`von-1.2`)

        Option scoring, order-independent

        Scores every option in one pass, independent of option order; English only.

        Good for: yes-no; choice; score.
        Limits: Not declared in model.toml; inspect the model README and operation bounds.
        Parameters: 395M.
        Languages: en.
        Licence: [Apache-2.0](https://huggingface.co/wfzyx/von/blob/5df8185a4f2327ad0a7cd117cc4f701ac557b9ae/README.md).
        Tools: the generic window built from its operations.

        - `decide`: `POST /arsenal/v1/models/von-1.2/ops/decide`.
        - `judge`: `POST /arsenal/v1/models/von-1.2/ops/judge`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | bf16 | 1584908301 | ~792454150 | 1.172 (m5-max, 2026-10-15, quickCheck) |

        [Model documentation](../models/von-1.2/README.md)

        ## VoxCPM2 (`voxcpm2`)

        Speech in 30 languages, voice cloning

        Speech in 30 languages, Polish included, at 48 kHz: a new voice, one described in words, or a clone of a short clip.

        Good for: Polish and 29 more languages from one model, with no language setting; Cloning a voice from a few seconds of audio, closest with the clip's exact transcript; A new voice from a description such as (a calm older man, deep voice); Steering pace, tone or emotion of a cloned voice with a description.
        Limits: No preset voices: a request speaks in a fresh voice, a described voice, a reference clip or a saved voice; The maker's reference denoiser (ZipEnhancer) and text normalizer (wetext) are not included; give clean clips and write numbers as words; No speed control; ask for a pace in the description instead; Audio arrives per chunk of text (about 400 characters), not per patch; Results vary between runs, as the maker says of voice design and style control; set seed to repeat one; Strictly forbidden by the maker for impersonation, fraud or disinformation; label AI-generated speech.
        Parameters: 2.4B.
        Languages: ar, my, zh, da, nl, en, fi, fr, de, el, he, hi, id, it, ja, km, ko, lo, ms, no, pl, pt, ru, es, sw, sv, tl, th, tr, vi.
        Licence: [Apache License 2.0](https://huggingface.co/openbmb/VoxCPM2/blob/32279effe8c19989596f05d353d1447f51d9e915/README.md).
        Tools: voice, sound.

        - `speak`: `POST /arsenal/v1/models/voxcpm2/ops/speak`.
        - `voices`: `POST /arsenal/v1/models/voxcpm2/ops/voices`.
        - `clone_voice`: `POST /arsenal/v1/models/voxcpm2/ops/clone_voice`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 4959015656 | 4959015656 | unknown |

        [Model documentation](../models/voxcpm2/README.md)

        ## YuE2 3B (`yue2-3b`)

        Songs from lyrics and style

        Generate 48 kHz stereo songs from style tags and lyrics with optional score planning.

        Good for: Full songs with vocals; Instrumental music; ABC score-conditioned songs.
        Limits: Vocal language follows the lyrics; English and Mandarin are documented; Trained FP16 and quantized output quality awaits qualification; Song files count against the 4 GiB per-job artifact limit, checked before generation; Generation produces complete files with progress, not live PCM.
        Parameters: 3.70B.
        Languages: en, zh.
        Licence: [CC-BY-NC-4.0](https://huggingface.co/m-a-p/YuE2-3B/tree/8f24312f187763b9854adeec87e437e2b03bbff1). Non-commercial use only (CC BY-NC 4.0), crediting M-A-P; the YuE2 VAE carries the same licence.
        Tools: sound.

        - `generate_music`: `POST /arsenal/v1/models/yue2-3b/ops/generate_music`.

        | Precision | Download (bytes) | On disk after Get (bytes) | Peak memory (GB, recorded evidence) |
        |---|---:|---:|---:|
        | fp16 | 7264003817 | 7264003817 | 11.811 (m5-max, 2026-10-22, quickCheck) |
        | int8 | made from the 16-bit download | ~3859002027 | 8.485 (m5-max, 2026-10-22, quickCheck) |
        | int4 | made from the 16-bit download | ~2043001073 | 6.540 (m5-max, 2026-10-22, quickCheck) |

        [Model documentation](../models/yue2-3b/README.md)
        """##
}
