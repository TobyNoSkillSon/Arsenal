<!-- SPDX-License-Identifier: MIT -->
# LCO-Embedding Omni 7B

LCO-Embedding-Omni-7B puts text, images, PDF pages, audio and video into one 3,584-number vector space, so one index can be searched with any of them. It is the thinker of Qwen2.5-Omni 7B (a Qwen2.5 7B language model with a window-attention vision encoder and a Whisper-style audio encoder; 8.93B parameters) trained with language-centric contrastive learning ("Scaling Language-Centric Omnimodal Representation Learning", NeurIPS 2025). The makers report state of the art on MIEB-Lite (51 image-embedding tasks) and on the MAEB audio benchmark without audio training data.

## Usage notes (2 Nov 2026)

**What it does.** `embed` returns one L2-normalized 3,584-dimension vector per input. Compare vectors with cosine similarity (a dot product, since they are normalized). Text, a photo, a PDF page, a sound clip and a video of the same thing land close together.

**Inputs.** One operation, `embed`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `input` | array, 1–2,048 items | yes | Each item is a string, `{"type":"text","text":…}`, `{"type":"image","image":{"path":…}}`, `{"type":"audio","audio":{"path":…}}`, `{"type":"video","video":{"path":…}}`, `{"type":"document","document":{"path":…}}` (a PDF, each page embedded as an image), or `{"type":"content","content":[…]}` to embed up to 32 parts together as one vector (for example a caption with its video). |
| `task` | `query` or `document` | no | Accepted for the embeddings API. The model uses one prompt for both, so it changes nothing. |
| `dimensions` | 3584 | no | The only size; there is no shorter projection. |

Every input goes through the makers' prompt (`additional_chat_templates/sentence_transformers.jinja`): the fixed system turn, the parts in order, then "Summarize the above image / video / audio / text in one word:" by the first of those present, and the vector is the final hidden state of the last prompt token. Media follow the model card's Transformers path, `qwen-omni-utils` with `Qwen2_5OmniProcessor` (not the sentence-transformers video processor, whose defaults sample differently): images keep their stored RGB values (no colour management), transparent areas become white, and they are resized with PIL's bicubic to 4 to 16,384 visual tokens with sides rounded to 28 px; video is sampled at 2 frames a second (at least 4, at most 768, an even count) at 128 to 768 tokens per frame pair, decoded with the file's colour matrix (BT.601 when the file has none, as FFmpeg does); audio is resampled to 16 kHz and its channels averaged.

MCP: `arsenal_run(model:"lco-embedding-omni-7b", input:{"input":["a dog on a beach", {"type":"image","image":{"path":"/absolute/path/photo.jpg"}}]})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/lco-embedding-omni-7b/ops/embed" -H 'Content-Type: application/json' -d '{"input":[{"type":"audio","audio":{"path":"/absolute/path/clip.wav"}}]}'`. The OpenAI-style `/v1/embeddings` route also takes `input_audio` parts.

**Limits.**
- The soundtrack of a video is not used (the makers' default, `use_audio_in_video=False`); embed it as a separate audio part.
- Audio up to 5 minutes per part (the processor's 300 s window); longer files are refused, not cut.
- 32,768 tokens per input (the language model's context), counted before any media is encoded. A large photo can take up to 16,384 tokens; full-size video takes 768 tokens a second, so about 40 seconds fit (audio is 25 tokens a second).
- PDFs: each page renders at twice its point size (at most 2,048 px); a US-letter page is 2,508 tokens, so about 13 pages fit in one input; a longer PDF is refused before its pages render. Embed long documents page by page.
- A rotated phone video is turned upright from its track transform; the makers' FFmpeg readers leave it as stored.
- No instructions and no shorter vectors.
- Polish: the language model is multilingual, but Polish retrieval was not checked.

**Parity result.** Checked on 2 Nov 2026 against the makers' own code: the model card's Transformers path (Transformers 5.22.1, `Qwen2_5OmniProcessor`, `qwen-omni-utils`, `Qwen2_5OmniThinkerForConditionalGeneration`, PyTorch 2.15 on MPS), captured once in BF16 and once in FP16 on fixed inputs (four card texts, a 336×252 image, a 3.2 s tone, a 2 s video). Full numbers: `gate/golden/golden.json`.
- Without weights, in the default test suite: the prompt, the rotary positions for mixed image/audio/video sequences, the vision window index, the resize and frame rules, and the Whisper log-mel (largest difference 7e-5) match the makers' functions.
- Token ids match the makers' processor exactly for every fixture (text, image, audio, video).
- Each component on the makers' own input tensors, Arsenal FP16 against the makers' FP16 run: language model pooled state cosine 0.999994 or better (relative L2 0.002 to 0.0035), vision tower 0.99990 (image) and 0.99996 (video), audio tower 0.996, whose differences sit on the few tokens where the makers' own BF16 and FP16 runs also disagree. For scale, the makers' BF16 run against their FP16 run: vision 0.995 to 0.997, audio 0.995, pooled state 0.9989 to 0.9999.
- End to end through `embed` with Arsenal's own tokenizer, decoding and features, against the makers' FP16 vectors: text 0.99999 to 1.0, image 0.99995, audio 0.9998, video 0.9992. Arsenal's video pixels match FFmpeg's decode to within 2 levels (cosine 0.99998); resized images match the makers' processor to within one 8-bit level, and transparent areas come out white as theirs do (weight-free tests on a 1000×750 image, an RGBA image and two clips, one resized). The card's retrieval example ranks the same way (Everest 0.645, Kilimanjaro 0.479, K2 0.432; the card prints 0.6456, 0.4788, 0.4331; the makers' own FP16 run here gives 0.6457, 0.4790, 0.4325).
- FP16 is the default: every hidden state stayed finite, the largest activation was 7,184 of FP16's 65,504.

## How Arsenal runs it

Checkpoint: `LCO-Embedding/LCO-Embedding-Omni-7B@108f6f1a5de3b2eedd1d1c7b7005aaca6ed3802c`, four BF16 safetensors shards (17.9 GB). Get casts them to FP16 file by file (`cast:embed-lco-omni-v1`); Load skips the language-model head and the audio tower's unused begin/end table (the embedding never reads them, 1.1 GB). Runner: the Embed family of the bundled native worker, sequential only, macOS 26 or later. Port notes: `log/001-port.md`.

## Licence and credit

Apache-2.0 per the pinned model card. Made by Chenghao Xiao, Hou Pong Chan, Hao Zhang, Weiwen Xu, Mahani Aljunied and Yu Rong (LCO-Embedding); built on Qwen2.5-Omni by the Qwen team. Paper: https://arxiv.org/abs/2510.11693.
