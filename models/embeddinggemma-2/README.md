<!-- SPDX-License-Identifier: MIT -->
# EmbeddingGemma 2

EmbeddingGemma 2 is Google DeepMind's open multimodal embedding model: a 270M-parameter bidirectional Gemma 4 text encoder (24 layers, width 512, per-layer embeddings, a 512-to-768 projection) with Gemma 4's 170M vision tower and 300M conformer audio tower, 744M parameters in all. Text (100+ languages, code), images, video and audio, alone or interleaved, land in one 768-number vector space. Google reports 61.4 on multilingual MTEB v2, 78.7 on MTEB code, 65.7 on MIEB-lite and 69.5 MRR@10 on MSEB audio retrieval, with vectors that can be cut to 512, 256 or 128 numbers at a small cost.

## What it is for

Small, fast semantic search across everything in one index: documents, code, photos, PDF pages, sound and video.

- Search: a text query against documents, photos, PDF pages, recordings and clips
- Question answering, fact checking and code search with the card's task prefixes
- Classification, clustering and similarity
- Short vectors (512, 256, 128) to cut storage

## Usage notes (2 Nov 2026)

**What it does.** `embed` returns one vector per input, mean-pooled over every token and L2-normalized; compare vectors with cosine similarity (a dot product, since they are normalized). Text, a photo, a recording and a clip of the same thing land close together.

**Inputs.** One operation, `embed`:

| Field | Type | Required | Notes |
|---|---|---|---|
| `input` | array, 1–2,048 items | yes | Each item is a string, `{"type":"text","text":…}`, `{"type":"image","image":{"path":…}}`, `{"type":"audio","audio":{"path":…}}`, `{"type":"video","video":{"path":…}}`, `{"type":"document","document":{"path":…}}` (a PDF, each page embedded as an image), or `{"type":"content","content":[…]}` with 1–32 parts embedded together as one vector. Text in a content item may hold `<\|image\|>`, `<\|audio\|>` and `<\|video\|>` placeholders; the media parts then fill them in order, as in the card's interleaving example. |
| `task` | `query`, `document`, `similarity`, `classification`, `clustering`, `question_answering`, `fact_checking`, `code_retrieval` | no | The card's text prefixes: `query` is "task: search result \| query: ", `document` is "title: none \| text: ". Default `document` for text-only items; images, audio and video alone take no prefix (the card's advice). Use the same task for both sides of symmetric tasks. |
| `title` | string, 1–1,000 | no | Document title: "title: {title} \| text: …". |
| `instruction` | string, 1–1,000 | no | Custom task name: "task: {instruction} \| query: …". |
| `dimensions` | 768, 512, 256 or 128 | no | Matryoshka size: the leading numbers, re-normalized. Queries and documents must use the same size. Google: near-lossless to 256; 128 is meant for text. |
| `normalize` | boolean | no | Default true. False returns the raw mean-pooled vector (the maker's pipeline always normalizes). |
| `truncate` | boolean | no | Cut text past 8,192 tokens and audio past 30 s instead of refusing. |

Returns `embeddings` (one array per input) and `input_tokens`.

Images are resized, keeping their shape, to at most 280 visual tokens (sides in multiples of 48 px) with the processor's antialiased bicubic; video is sampled at one frame a second, at most 32 frames spread evenly, each frame at most 140 tokens; audio is read as 16 kHz mono (the mean of up to 8 channels), 25 tokens a second. EXIF orientation, a video's display rotation and a PDF page's crop box and rotation are applied.

MCP: `arsenal_run(model:"embeddinggemma-2", input:{"input":["what causes the northern lights?"], "task":"query"})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/embeddinggemma-2/ops/embed" -H 'Content-Type: application/json' -d '{"input":[{"type":"image","image":{"path":"/absolute/path/photo.jpg"}}]}'`. The OpenAI-style `/v1/embeddings` route also takes `image_url` and `input_audio` parts.

**Limits.**
- 8,192 tokens per input, counted before any media is encoded: an image up to 280, a video frame up to 140 (so about 58 one-second frames would fit, but at most 32 are sampled), a second of audio 25, a PDF page 280 (at most 29 pages).
- Audio past 30 seconds is refused unless `truncate` is set; then its first 30 seconds are used. This is what the maker's processor does (its feature extractor cuts at 480,000 samples), although the card speaks of minutes of audio.
- A video's soundtrack is not used; embed it as a separate audio part.
- Polish: Google states 100+ languages without listing them. A Polish probe scored a Polish aurora text above every unrelated item for an English aurora query (0.68 against 0.55 for a pasta text); Polish retrieval quality was not measured.
- CMYK JPEGs go through macOS colour management, while the maker's Pillow converts them arithmetically, so their vectors may differ slightly; a video's display rotation is applied, which the maker's reader was not checked for.
- A long video is decoded up to its last sampled frame, so very long clips take long to embed.
- No speed figures are claimed.

**Parity result.** Checked on 2 Nov 2026 against the makers' own code: the card's sentence-transformers path (sentence-transformers 6.2, Transformers 5.22.1, PyTorch 2.15 on MPS), captured once in FP16 and once in BF16 on fixed inputs (nine texts with the card's prefixes including one Polish, a 480×320 photo, a text-page image, a 2.9 s spoken clip, a 4 s video and one interleaved photo + audio + text item). Full numbers: `gate/golden/golden.json`.
- Without weights, in the default test suite: the chat-template layouts, resize sizes, the resized patch bytes (identical), EXIF orientation, frame sampling, the log-mel features (largest difference under 0.001), audio token counts, the audio tower's blocked key mask and relative shift, and the vision RoPE table and pooling match the maker's functions.
- Token ids are identical to the maker's processor for all 14 inputs.
- Each part on the maker's own input tensors, Arsenal FP16 against the maker's FP16 run: vision tower 0.999997 or better, audio tower 0.999997, text encoder per-token outputs 0.999994 or better and every checked layer 0.999998 or better.
- End to end through `embed` with Arsenal's own tokenizer, decoding, resizing and features, against the maker's FP16 vectors: texts 0.999998 to 1.0, images 1.0, audio 0.999999, the interleaved item 0.999999, video 0.99993 (FFmpeg and AVFoundation round a few pixels differently, at most 2 levels). Against the maker's BF16 vectors, the card's recommended precision: 0.99988 or better.
- Each query's best match is the right item: aurora query to the aurora text (0.87), "a rocky ocean coastline" to the photo (0.76), "a dog barking" to the spoken clip (0.78), "how to cook pasta" to the recipe page image (0.74), "a red ball rolling across a field" to the video (0.75).
- FP16 is the default, although the card says not to use it: in the maker's own code FP16 stayed finite and within cosine 0.9998 of BF16 on every input tried here, including an 11,409-token text, all-white and all-black images and 20 s of full-scale noise. In Arsenal the largest text hidden state was 84; the vision pooler's scaled output (up to 28,489, near FP16's 65,504 on a white page) is kept in float32 until the embedder's norm, as the maker computes it.

## How Arsenal runs it

Checkpoint: `google/embeddinggemma-2@f9972ac22e696a105c482202646979e33e4b5a1e`, one BF16 safetensors file (1.49 GB) that Get casts to FP16 (`cast:embed-gemma2-v2`). Runner: the Embed family of the bundled native worker (`Workers/Embed/Sources/EmbedWorker/EmbeddingGemma2*.swift`), sequential only, macOS 26 or later. The text encoder is adapted from mlx-swift-lm's Gemma code and the vision tower follows its Gemma 4 port; the audio conformer and the Gemma 4 feature extractor are ported from Transformers. Port notes: `log/003-final-picks-1102.md`.

## Licence and credit

Apache-2.0 per the pinned model card (Google's Gemma 4 licence page); commercial use allowed. Use must also follow Google's Gemma Prohibited Use Policy. Made by Google DeepMind (Henrique Schechter Vera, Sahil Dua and the EmbeddingGemma team). Paper: "EmbeddingGemma 2: Unified and Efficient Multimodal Representations", arXiv 2610.37364.

## Links

- Model card: https://huggingface.co/google/embeddinggemma-2
- Code: https://github.com/huggingface/transformers (models/embedding_gemma2, models/gemma4)
- Paper: https://arxiv.org/abs/2610.37364
