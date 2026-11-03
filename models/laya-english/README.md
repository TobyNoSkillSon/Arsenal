# Laya · English

Laya is a non-autoregressive "System 1" decision model by Convai Innovations; this is its English checkpoint, run from aac6fef's native MLX FP16 conversion. It is a fully fine-tuned ModernBERT-large encoder (395M) with a decision head trained from scratch (two transformer layers, an option-marker scorer and an act/escalate head), 421M parameters in all. Every option of a question is scored at its own `[MASK]` token and softmaxed over that question's options, so the answer space is set per request and every question in a call is answered in one forward pass. It returns typed answers with probabilities and never generates text.

## What it is for

The makers trained it with reinforcement learning against strictly proper scoring rules (RLCD), so that reporting honest probabilities maximises reward, and built it for fast routing, classification, scoring and guardrail decisions in English. Arsenal suggests it as the decisions model for English text: on the makers' English tasks it is more accurate than Laya · Multilingual, which is faster.

- Routing English tickets, emails or JSON records to a team or intent.
- Yes/no guardrails and moderation checks ("does the user threaten to cancel?").
- Rating urgency or severity on a short ordered rubric.
- Shortlisting many search hits, files or log lines with `judge` before reading them in full.

## Inputs and outputs

### `decide` (default)

Answers any number of named, typed questions about one state in one call.

| Field | Type | Required | Notes |
|---|---|---|---|
| `state` | string, object or array | yes | The content every question is about. Objects and arrays are read in the key order sent. Objects with `image`, `images`, `audio`, `video` or `videos` keys are refused (text only). |
| `questions` | object | yes | Question name → question, at least 1 (`minProperties` 1). Names come back as the answers' keys; they are not shown to the model. Each question is described below. |
| `model` | string | no | `minLength` 1. `"auto"` (laya-english when the text's letters are ≥ 99.5 % ASCII, else laya-multilingual) or a decisions model id. Omit it on this model's own op endpoint: this model answers. |
| `bits` | integer or null | no | One of `0`, `4`, `8`, `16`, `null`. The precision this request requires (0 = 16). A request never changes the loaded precision: another value is a 409. Laya ships 16-bit only, so `0`, `16` or `null` pass. |
| `merge` | boolean | no | Verdict extension, accepted for compatibility. Arsenal never merges requests, so every answer is the one the request gets alone. |

A question is an object with these fields (unknown fields are ignored):

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | `noul` = yes/no (answer P(yes)); `choice` = one of named options; `score` = an ordered level. |
| `instructions` | string, object or array | no | The question itself, as text or structured JSON. |
| `criteria` | depends on `type` | `choice` and `score`: yes | `noul`: optional object with only `true` and `false`, each text, JSON or null. `choice`: object, option label → description (text, JSON, or null for the bare label), at least 1; labels are compared as exact strings and must be unique. `score`: array of level descriptions, lowest first, at least 1; levels are answered as `"0"`, `"1"`, …. |

Returns JSON `{"answers": {...}, "model": "laya-english", "usage": {"input_tokens": n, "output_tokens": 0}}`. Each answer has `type` and:

- `noul`: `noul`, the probability that the statement holds, 0 to 1.
- `choice`: `choice` (the label with the highest probability), `probabilities` (label → probability, summing to 1 over the options you sent) and `confidence`.
- `score`: `score`, the expected level (the sum of level index × probability, so 1.4 means between level 1 and 2, nearer 1), `probabilities` (`"0"`, `"1"`, … → probability), `confidence`, and `legend` (`"0"`, `"1"`, … → the level descriptions exactly as sent).

How Laya reads a question: each question becomes a header `<type> question: <instructions>` followed by its options, each behind its own marker. A choice option is written `label: description`, or the bare label when the description is null; a score level is `level 0: <description>`, `level 1: …`; a noul has two options, `false: <false description>` and `true: <true description>`, which default to "no, the statement does not hold" and "yes, the statement holds". The state follows the header: a string as is, an object or array as its JSON text with keys in the order sent. So write the instructions as the question, put what each option means in its description, and keep labels short ids for your code. The header and its options share a budget of 192 tokens (`head_max_len`); each option is cut to 48 tokens, options are shortened further when many of them crowd the budget, and a question whose options cannot all fit is refused. The state gets what remains of the 512-token context: a state that needs more than the context together with the longest question is refused ("Item needs about N tokens … Shorten it or split it"), as a 422 in `decide` and a per-item error in `judge`. A literal `[MASK]` inside the state is replaced by a space.

Probabilities are a softmax of the option logits divided by the checkpoint's fitted temperature for that question type and option count (from `rl_agent_config.json`), rounded to 4 decimals. `confidence` for choice and score is 1 minus the distribution's entropy over its maximum (ln k for k options): 0 when the options are equally likely, 1 when one has all the probability. The makers report the shipped checkpoint as over-confident (see Limits).

### `judge`

Asks the same questions about many items in one call; each item is its own state.

| Field | Type | Required | Notes |
|---|---|---|---|
| `items` | array | yes | At least 1. Strings, or any JSON value (objects are read in the key order sent). Media items get a per-item error. |
| `questions` | object | yes | As in `decide`. One batch extension: a `choice` question's `criteria` may also be a plain list of labels (strings). |
| `model` | string | no | As in `decide`. With `"auto"`, non-English items are sent to laya-multilingual and the results come back in item order. |
| `bits` | integer or null | no | As in `decide`. |

Returns JSON `{"results": [...]}`, one result per item in item order. A result is `{"answers": {...}, "model": "laya-english", "ms": n}`, where the answers carry the same `noul`, `choice`, `score`, `probabilities` and `confidence` values as `decide` without `type` and `legend` (noul answers here also carry `confidence`, the larger of P(yes) and P(no)), and `ms` is the batch's time divided by its items. An item that cannot be answered (too long, media) gets `{"error": "...", "model": ..., "ms": 0}` while the others are still answered. Use `judge` for lists; use `decide` for one state.

## Run it

MCP: `arsenal_run(model:"laya-english", input:{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}})` (add `operation:"judge"` with `items` for a batch).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/laya-english/ops/decide" -H 'Content-Type: application/json' -d '{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

TypeSafe System One route (also at `/providers/typesafe/v1/systemone`; batches at `/v1/judge`), which needs `model`: `curl -s "$ARSENAL_URL/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"laya-english","state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

The only setting is Kernels (Tuned or Stock MLX, under More; setting `stock_mlx`, applied at Load): Stock MLX runs plain MLX without the fast tokenizer and kernels, with the same answers as Exact, slower, for diagnosis. Measured on an M5 Max at FP16 in a quick check on a busy Mac: 5.51 ms per warm decision, 1.19 GB peak worker memory (`results/intake-m5-max-fp16-native-fast.json`).

## Limits

- English only. On the makers' 51-language MASSIVE test this checkpoint collapses outside English while staying confident (Khmer 0.000 accuracy at 0.952 confidence), so confidence cannot catch it; use Laya · Multilingual, or `model:"auto"`, for other languages.
- Text only: image, audio and video state is refused.
- 512 tokens per question, shared by the question, its options and the state; the question and options get at most 192 of them.
- Keep `choice` questions under about 20 options: options share the 192-token budget and accuracy falls off sharply with many labels. Split into a coarse choice and then a fine one.
- Ordinal `score` questions are its weakest type (SST-5 0.372 in the makers' tests).
- Ships over-confident: the makers measured mean ECE falling from 0.466 to 0.081 after refitting one temperature per question type and option count, and advise doing that on your own data before trusting the probabilities. Arsenal uses the shipped temperatures.
- Near chance on the typed-decisions benchmark zero-shot (0.362); Laya · Typed decisions is the checkpoint fine-tuned for those workflows.

## Licence and credit

Apache-2.0: free to use, modify and redistribute, including commercially, provided the licence and `NOTICE` are kept and changes are stated; it includes a patent grant. Original Laya models and code by Convai Innovations and contributors; MLX conversion by aac6fef (laya-mlx). The makers publish no citation; credit "Laya, Convai Innovations".

## Links

- Model card: https://huggingface.co/aac6fef/laya-mlx/blob/20aed815fc6acde75733882e7ec0e3f28aeb9717/README.md
- Original model card: https://huggingface.co/convaiinnovations/laya
- Code: https://github.com/NandhaKishorM/laya
- MLX port code: https://github.com/mizorewww/laya-mlx

## How Arsenal runs it

Source: `aac6fef/laya-mlx` at `20aed815fc6acde75733882e7ec0e3f28aeb9717`, FP16 MLX safetensors (download 0.85 GB; exact files, sizes and hashes in `model.toml`). The card states it converts `convaiinnovations/laya` at `c5d78730f3493e4fe16d61507ef4b78eef7318cf` (upstream code `NandhaKishorM/laya` at `6a5819129eb220570792e417e49723d697efd76f`) by renaming parameters and keeping the FP16 weights, with no retraining or further quantization. Arsenal ships 16-bit only since 26 October 2026: the 16-bit weights are 0.846 GB, under the 1 GB line below which int8 and int4 are not offered (`log/009-sixteen-bit-only.md`).

Runner: the native Swift decisions worker, Verdict's engine at `f8177ed` copied verbatim (Laya network, prompt, fast byte-BPE tokenizer, windowed attention, small-M GEMM route; `Workers/Decisions/PORTED-FROM.md`), linked into the one native worker binary (`log/004-native-link.md`) and driven over ArsenalWire. Two folder kernels serve the K > N Linears (`kernels/manifest.toml`, site `linear.down`): `splitk-exact` in Exact on every NAX chip, MLX's own split-K GEMM in one dispatch and bit-identical to the stock path (`log/005-folder-kernel-splitk-exact.md`: single items 7.91 → 7.50 ms median, −5.1 % over 4 alternated pairs, 202/202 answers byte-identical); `smallm-tile` in Fast on M5 at FP16, Verdict's qualified small-M tile (`log/004-folder-kernel-smallm-tile.md`: 7.42 → 6.89 ms median, −7.1 %, 0 changed answers, max |Δp| 0.0066). The MLX buffer cache is bounded at 1024 MB as in Verdict; without it the footprint reached 10 GB after 202 items, with it 1.35 GB (`log/002-derived-views-and-cache.md`).

Gate (`gate/gate.toml`, `log/001-verdict-port-parity.md`, 2 October 2026, M5 Max): Verdict's Laya fixture regression, 202 oracle items per set, three sets, each judged alone over the wire. Exact: 0 mismatches at FP16, int8 and int4, byte-identical answers. Fast at FP16: 197/202 items differ in the 4th decimal, max |Δp| 0.0066, 0 changed answers, inside Verdict's noise ruling (≤ 0.5 % changed, |Δp| ≤ 0.01). Single-item median under the GPU lock (not a benchmark): FP16 7.28 ms; load 0.14 s; MLX peak 1792 MB at FP16. Typed accuracy 4/4 on the decisions suite; the quick suite run (`results/m5-max.json`, contended Mac) answered all four cases correctly at 486 batched questions per second.

Precision intake (`log/006`, `log/007`, `log/008`, 15 October 2026, M5 Max, busy Mac): with the default Load kernels, FP16 measured 5.533 ms per warm decision and 1.185 GB peak; the since-withdrawn int8 passed (max relative L2 0.0074, max |Δp| 0.0038, 0.744 GB) and int4 failed its component bounds (0.1151, 0.0408, 0.529 GB). The schema 2 folder (decide/judge ops from the kind's shared schemas, `stock_mlx` setting, `suggested`) is recorded in `log/003-v2-folder.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: judge takes `items` and `questions`; 512-token context. [judge.input.json; model card]
- Prompts, reference audio or voice settings, if relevant: True/false and yes/no labels flipped a positive review to negative; positive/negative, A/B, 1/2 did not. [issue #156, 2026-09-22, first-hand]
- Parameters people recommend: Refit temperatures on your own data. [model card, first-hand]
- Arsenal call: `arsenal_run(model:"laya-english", operation:"judge", input:{"items":["Refund my order"],"questions":{"refund":{"type":"choice","criteria":{"A":"Asks for refund","B":"Does not"}}}})`

**Good and bad**
- Good at: 0.817 accuracy under 600 characters on 741 solicitations. [issue #450, 2026-09-24, first-hand]
- Bad at: 0.655 at 600 to 1,200 characters; 0.333 on SST-5 scores. [issues #450 and #555, first-hand]
- Against the obvious alternatives: 1,240 decisions, 9 suites: laya 0.686, Jev 1.13.0 0.907, Qwen2.5-1.5B PCD 0.605. [issue #555, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: `act_probability` is about 1.0 for every input. Fix or workaround: None; maintainer says document it. Confidence scored 0.77 on those items. [issue #185, 2026-09-22; maintainer, 2026-10-11, first-hand]
- Complaint or known bug: True/false labels bias noul answers. Fix or workaround: Maintainer: retrained checkpoint is the real fix. Interim: a two-option choice with neutral keys, or the `labels` override (PR #163, merged 2026-09-23; one user found it worse than default on laya-typed-decisions). [issue #156, 2026-09-22 to 24, first-hand]
- Complaint or known bug: Multi-row states get one answer for the whole state; 12 of 20 records cut off at 20 rows (maintainer). Fix or workaround: Per-record answers need `states`; truncation warning is PR #1426, open. [issue #1425, 2026-10-31, first-hand]

**Evidence**
- Strength: thin. One outside reproduction (#776); two evals favour Jev.
- Sources:
  - Model card, huggingface.co/convaiinnovations/laya (repo created 2026-09-18, inferred)
  - Issues #156, #185, #450, #555, #776, #1425 at github.com/NandhaKishorM/laya, plus PRs #163, #1426
