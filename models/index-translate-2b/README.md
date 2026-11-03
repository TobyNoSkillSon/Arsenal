# Index Translate 2B

Index Translate 2B is the compact 2-billion-parameter translation model of the Index-Translate family from bilibili's Index team, built on Qwen3.5 (2.2B parameters in Arsenal's count, including the vision encoder). It translates text across 150 languages, translates the text inside images without a separate OCR step, and follows translation instructions: glossaries, protected formats, tone, domain and context. Its strength is instruction-following translation in little memory: the makers report it ahead of Hy-MT2-1.8B on every main translation metric, with an instTrans instruction-adherence score of 0.7569, above Hy-MT2-30B-A3B (0.6415).

## What it is for

The makers trained it for constrained, instruction-following translation across many language pairs, including low-resource ones, through multilingual mid-training followed by specialist fine-tuning and reinforcement learning for general translation, instruction following and meme and community-slang translation.

- Translating documents, articles, subtitles and UI strings with a fixed glossary of terms
- Translating JSON, CSV, Markdown or code while leaving keys, markup, code and placeholders such as `{variable}` intact
- Adapting tone (formal, casual, social media) and resolving domain-specific meanings with context
- Translating screenshots, signs, menus and scanned pages from an image
- Low-resource language directions, where the makers report 3.97% off-target outputs on their low-resource instruction set
- Translation on Macs with less memory, or alongside other loaded models; `index-translate-9b` is more accurate
- Batches of up to 128 strings with the same settings

## Inputs and outputs

### `translate` (default)

Translates a batch of texts, or the text in one image, into `target_lang`, following any glossary and constraints given.

| Field | Type | Required | Notes |
|---|---|---|---|
| `target_lang` | string | yes | 2–32 characters. A language code (`en`, `de`, `pl`, `zh`, `zh-Hant`, `ja`, …) or a full language name (`English`, `英语`). |
| `text` | array of strings | no | Send `text`, `image` or both. 1–128 items, each 1–131072 characters. Each item is translated separately with the same settings. With `image`, the strings are joined and passed as reference text instead. |
| `image` | file object `{"path": string}` | no | Accepts `image/png`, `image/jpeg`, `image/webp`, `image/bmp`, `image/gif`, up to 16 megapixels. |
| `source_lang` | string | no | 2–32 characters, a code or name as for `target_lang`. Omit it or send `auto` to leave the source language unnamed in the prompt. |
| `context` | string | no | Up to 32768 characters. Background for the translation, added to the prompt as a constraint. |
| `max_tokens` | integer | no | 1–32768, default 1024. Output budget per item. |
| `glossary` | object | no | Source term → target term, all values strings. Added as a hard terminology constraint. |
| `tone` | string | no | Style the translation must take, for example "formal business email". |
| `format` | string | no | `json`, `csv`, `markdown` or `code`: keep that structure and translate only user-facing text. |
| `genre` | string | no | Kind of source text (for example "academic paper", "novel"); used in the constrained prompt. |
| `instructions` | string | no | Any other instruction for the translation. |
| `hard_constraints` | array of strings | no | Rules the translation must follow. |
| `soft_constraints` | array of strings | no | Preferences the translation should follow. |

Returns JSON: `translations`, an array with one entry per `text` item (one entry for an image), each `{"text": string, "detected_source_language": string, "finish_reason": "stop" | "length" | "repetition"}`. `detected_source_language` echoes `source_lang`, or `und` when it was omitted or `auto`; the model does not report the language it saw. `finish_reason` is `length` when the item hit `max_tokens`, and `repetition` when Arsenal stopped a looping output early.

Prompts: Arsenal writes the makers' own Chinese prompt templates, so callers give fields, not prompts. Without constraints a text uses the plain translation prompt; with any of `glossary`, `format`, `tone`, `context`, `instructions`, `hard_constraints` or `soft_constraints` it uses the makers' instTrans constraint format, with hard constraints, format and glossary as 【硬性要求】 lines and the rest as 【注意】 lines. Language codes map to the makers' Chinese language names; codes they do not list are named by macOS, and a full name is passed as written. An image uses the makers' image-translation prompt, with the instruction text before the image.

Images: the vision encoder works on 32-pixel merged patches. Arsenal accepts images up to 16 megapixels and resizes them so the model sees between 65,536 and 4,194,304 pixels (about 64 to 4,096 image tokens). Small text in a very large image may become unreadable at that budget, so crop to the region first *(unverified)*.

Length: the prompt and output of each item together must fit in 32,768 tokens, the makers' serving preset, though the checkpoint's configuration allows 262,144. Longer items are refused, not truncated; split them. Raise `max_tokens` for long translations; an item that hits it ends with `finish_reason: "length"`.

## Run it

MCP: `arsenal_run(model:"index-translate-2b", input:{"text":["Carbon fibre improves the stiffness of concrete."],"target_lang":"pl","glossary":{"carbon fibre":"włókno węglowe"}})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/index-translate-2b/ops/translate" -H 'Content-Type: application/json' -d '{"text":["Carbon fibre improves the stiffness of concrete."],"target_lang":"pl","glossary":{"carbon fibre":"włókno węglowe"}}'`

An image path may name any file on this Mac; Arsenal clones it into its inbox before the model reads it. `arsenal_upload` is only for a file that is not on this Mac. Decoding is greedy with thinking off, as the makers' client does. Precision (FP16 or int8) is chosen per model in Arsenal, not per request. In a quick check on an M5 Max (22 Oct 2026, Mac in use, short fixed test input), a warm request took 0.22 s at FP16 and 0.14 s at int8, with peaks of 6.08 and 3.40 GB.

## Limits

- Quality varies by language and task. The makers report that combined constraints and low-resource directions remain harder than core-language translation, and that a successful example does not guarantee every instruction will be followed.
- 32,768 tokens of prompt plus output per item; long documents must be split. The makers point to their Index-NativeLong models for full-document translation, and to Index-Homura for translation to a target syllable count; neither is in Arsenal.
- One image per request, and no PDF input; use a document model to extract the text of a PDF first.
- It is a translator, not a chat model; the makers note that translation training changes the base model's general capabilities.
- Less accurate than `index-translate-9b` on nearly every benchmark the makers report (they tie only on GuoFeng web novels), most of all on judged quality (WMT26 Judge 60.26 against 75.35) and low-resource directions.
- Checked live on 20 Oct at FP16 and int8: Polish correct; German dropped one word in one sentence. int4 failed that check and is not offered.

## Licence and credit

Apache-2.0. Commercial use, modification and redistribution are allowed; keep the licence and any NOTICE text, and mark changed files. The licence also grants a patent licence from contributors. Made by the Index LLM Team at bilibili, on Qwen3.5 from Alibaba's Qwen team. The makers ask for this citation:

```bibtex
@techreport{indextranslate2026,
  author={Tianjiao Li and Mengran Yu and Chenyu Shi and Lusheng Zhang and
          Qisi Chen and Yanshan Zhou and Ji Qi and Jingying Liu and
          Yuang Feng and Ziang Cui and Tianxing Yan},
  title={Index-Translate: A Multilingual Translation Model Family --- Text, Speech, Controlled Dubbing, and Long-Document Translation},
  institution={Index LLM Team},
  year={2026},
  month={September},
  url={https://arxiv.org/abs/2609.40181}
}
```

## Links

- Model card: https://huggingface.co/IndexTeam/Index-Translate-2B/tree/9bc08445315723eb9b77d1b93505915eb03c3f01
- Code: https://github.com/bilibili/Index-Translate
- Paper: https://arxiv.org/abs/2609.40181

## How Arsenal runs it

Source: `IndexTeam/Index-Translate-2B@9bc08445315723eb9b77d1b93505915eb03c3f01`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 derived locally. Int4 is removed: its live quality check failed (role text and reasoning leaked into the translation, and a verb was mistranslated). Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors were downloaded during integration.

Serving follows the maker card: prompt plus output within 32,768 tokens (`--max-model-len 32768`) and 1,024 output tokens per item by default; long text prompts prefill 1,024 tokens at a time. The image processor is Qwen3-VL's (patch 16, merge 2, temporal patch 2, mean and std 0.5), narrowed by `runners/native/language.json` to at most 4,194,304 pixels. Answers pass through the makers' `strip_think`, which drops any `<think>` block.

Parameter label: 2.2B, derived from the complete default text/image graph inventory. Source checkpoint: 2,274,069,824 stored values; imported graph: 2,213,241,664 values.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match (graph tensors 617; ignored source tensors 15); source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing (45/45 matrices) pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Measured: checked live on 20 Oct at FP16 and int8: Polish correct; German dropped one word in one sentence; int4 failed and was removed. The 22 Oct quick check (one Load and two Translate requests on a short fixed input, Mac in use) is in `results/`: load 1.38 s and 1.63 s; warm request 0.217 s and 0.143 s; peak 5,794 and 3,247 MiB at FP16 and int8. Research log: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: One user message, no system prompt. Constrained prompts use 【源文】 and 【约束要求】 blocks. [model card, first-hand]
- Prompts, reference audio or voice settings, if relevant: None apply. [model card]
- Parameters people recommend: Greedy decoding, `temperature=0`, thinking off. `max_tokens` 1,024 is a client default. [model card, first-hand]
- Arsenal call: `arsenal_run(model:"index-translate-2b", operation:"translate", input:{"text":["Keep the invoice."],"target_lang":"de","glossary":{"invoice":"Rechnung"}})`

**Good and bad**
- Good at: instTrans IFscore 0.7569, above Hy-MT2-30B-A3B's 0.6415 (maker's table). [model card, first-hand]
- Bad at: 3.97% off-target on the maker's low-resource set. One tester rated it 2 of 5 on game scripts, breaking escape codes and JSON. [model card; issue #9, first-hand]
- Against the obvious alternatives: Below the 9B on every column of the maker's main table. [model card, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Hosted API: a 1.2-times cap rendered as an increase; "release paused" as "development paused". Unanswered. Fix or workaround: Not found in sources checked. [issue #18, 2026-10-10, first-hand]
- Complaint or known bug: Broken JSON and escape codes on game scripts. Fix or workaround: Reported by one user, not confirmed: 3 to 4 dialogue lines per request, validated. [issue #9, 2026-10-04, first-hand]
- Complaint or known bug: A paragraph repeated 8 times came back 16 times, no end marker. Fix or workaround: Not found in sources checked. [discussion #3, 2026-10-31 UTC, first-hand, no reply]
- Complaint or known bug: Q&A text answered, not translated. Fix or workaround: Not found in sources checked. [discussion #2, 2026-10-30 UTC, first-hand, no reply]

**Evidence**
- Strength: thin. Maker figures, one tester, one hosted-API user, two discussions.
- Sources:
  - Model card, huggingface.co/IndexTeam/Index-Translate-2B, first-hand (maker)
  - Issues #9 and #18, github.com/bilibili/Index-Translate
  - Discussions #2 and #3, huggingface.co/IndexTeam/Index-Translate-2B
