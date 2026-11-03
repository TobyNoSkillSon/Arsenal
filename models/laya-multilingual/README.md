# Laya · Multilingual

Laya is a non-autoregressive "System 1" decision model by Convai Innovations; this is its multilingual checkpoint for 100+ languages, run from aac6fef's native MLX FP16 conversion. It is an mmBERT-base encoder (307M, 22 layers, hidden size 768, 256k vocabulary) with a decision head trained from scratch (two transformer layers, an option-marker scorer and an act/escalate head), 322M parameters in all. Every option of a question is scored at its own `[MASK]` token and softmaxed over that question's options, so the answer space is set per request and every question in a call is answered in one forward pass. It returns typed answers with probabilities and never generates text.

## What it is for

The makers trained it from scratch with reinforcement learning against strictly proper scoring rules (RLCD) as the Laya checkpoint for anything that is not English. It is also the fastest Laya: the smaller 768-wide, 22-layer encoder is cheaper per token, about 2× faster than the English checkpoint in the makers' tests. `model:"auto"` sends non-English text here.

- Routing tickets, emails or JSON records written in languages other than English (Polish, German, Hindi, Japanese, …).
- Yes/no checks and moderation on mixed-language input.
- Intent classification across languages.
- The fastest pass over a large batch with `judge`, when its English accuracy is enough.

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

Returns JSON `{"answers": {...}, "model": "laya-multilingual", "usage": {"input_tokens": n, "output_tokens": 0}}`. Each answer has `type` and:

- `noul`: `noul`, the probability that the statement holds, 0 to 1.
- `choice`: `choice` (the label with the highest probability), `probabilities` (label → probability, summing to 1 over the options you sent) and `confidence`.
- `score`: `score`, the expected level (the sum of level index × probability, so 1.4 means between level 1 and 2, nearer 1), `probabilities` (`"0"`, `"1"`, … → probability), `confidence`, and `legend` (`"0"`, `"1"`, … → the level descriptions exactly as sent).

How Laya reads a question: each question becomes a header `<type> question: <instructions>` followed by its options, each behind its own marker. A choice option is written `label: description`, or the bare label when the description is null; a score level is `level 0: <description>`, `level 1: …`; a noul has two options, `false: <false description>` and `true: <true description>`, which default to "no, the statement does not hold" and "yes, the statement holds". The state follows the header: a string as is, an object or array as its JSON text with keys in the order sent. So write the instructions as the question, put what each option means in its description, and keep labels short ids for your code. The header and its options share a budget of 256 tokens (`head_max_len`); each option is cut to 48 tokens, options are shortened further when many of them crowd the budget, and a question whose options cannot all fit is refused. The state gets what remains of the 1,024-token context: a state that needs more than the context together with the longest question is refused ("Item needs about N tokens … Shorten it or split it"), as a 422 in `decide` and a per-item error in `judge`. A literal `<mask>` (this tokenizer's mask token) inside the state is replaced by a space.

Probabilities are a softmax of the option logits divided by the checkpoint's temperature (from `rl_agent_config.json`; this checkpoint ships 1.0 for every question type, with no per-option-count values), rounded to 4 decimals. `confidence` for choice and score is 1 minus the distribution's entropy over its maximum (ln k for k options): 0 when the options are equally likely, 1 when one has all the probability. The makers report the shipped checkpoint as uncalibrated and over-confident (see Limits).

### `judge`

Asks the same questions about many items in one call; each item is its own state.

| Field | Type | Required | Notes |
|---|---|---|---|
| `items` | array | yes | At least 1. Strings, or any JSON value (objects are read in the key order sent). Media items get a per-item error. |
| `questions` | object | yes | As in `decide`. One batch extension: a `choice` question's `criteria` may also be a plain list of labels (strings). |
| `model` | string | no | As in `decide`. With `"auto"`, items whose letters are ≥ 99.5 % ASCII are sent to laya-english and the results come back in item order. |
| `bits` | integer or null | no | As in `decide`. |

Returns JSON `{"results": [...]}`, one result per item in item order. A result is `{"answers": {...}, "model": "laya-multilingual", "ms": n}`, where the answers carry the same `noul`, `choice`, `score`, `probabilities` and `confidence` values as `decide` without `type` and `legend` (noul answers here also carry `confidence`, the larger of P(yes) and P(no)), and `ms` is the batch's time divided by its items. An item that cannot be answered (too long, media) gets `{"error": "...", "model": ..., "ms": 0}` while the others are still answered. Use `judge` for lists; use `decide` for one state.

## Run it

MCP: `arsenal_run(model:"laya-multilingual", input:{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}})` (add `operation:"judge"` with `items` for a batch).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/laya-multilingual/ops/decide" -H 'Content-Type: application/json' -d '{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

TypeSafe System One route (also at `/providers/typesafe/v1/systemone`; batches at `/v1/judge`), which needs `model`: `curl -s "$ARSENAL_URL/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"laya-multilingual","state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

The only setting is Kernels (Tuned or Stock MLX, under More; setting `stock_mlx`, applied at Load): Stock MLX runs plain MLX without the fast tokenizer and kernels, with the same answers as Exact, slower, for diagnosis. Measured on an M5 Max at FP16 in a quick check on a busy Mac: 3.47 ms per warm decision, 1.33 GB peak worker memory (`results/intake-m5-max-fp16-native-fast.json`).

## Limits

- Weaker on English than Laya · English (0.619 against 0.684 macro across the makers' English suites); route English text there, or use `model:"auto"`.
- Low-resource languages stay weak: in the makers' MASSIVE intent test Swahili scored 0.210, Tamil 0.250, Amharic 0.110 (20 options, random 0.050).
- Text only: image, audio and video state is refused.
- 1,024 tokens per question, shared by the question, its options and the state; the question and options get at most 256 of them.
- Keep `choice` questions under about 20 options: options share the 256-token budget and accuracy falls off sharply with many labels.
- Ordinal `score` questions are its weakest type (SST-5 0.282 in the makers' tests).
- Ships uncalibrated (temperature 1.0) and systematically over-confident: the makers measured mean ECE falling from 0.314 to 0.106 after refitting one temperature per question type and option count, and advise doing that on your own data before trusting the probabilities. Arsenal uses the shipped temperatures.
- Near chance on the typed-decisions benchmark zero-shot (0.342).

## Licence and credit

Apache-2.0: free to use, modify and redistribute, including commercially, provided the licence and `NOTICE` are kept and changes are stated; it includes a patent grant. Original Laya models and code by Convai Innovations and contributors; MLX conversion by aac6fef (laya-mlx). The makers publish no citation; credit "Laya, Convai Innovations".

## Links

- Model card: https://huggingface.co/aac6fef/laya-multilingual-mlx/blob/f2b4faf51023039425946074e2cf1361d2db11d5/README.md
- Original model card: https://huggingface.co/convaiinnovations/laya-multilingual
- Laya family: https://huggingface.co/convaiinnovations/laya
- Code: https://github.com/NandhaKishorM/laya
- MLX port code: https://github.com/mizorewww/laya-mlx

## How Arsenal runs it

Source: `aac6fef/laya-multilingual-mlx` at `f2b4faf51023039425946074e2cf1361d2db11d5`, FP16 MLX safetensors (download 0.68 GB; exact files, sizes and hashes in `model.toml`). The card states it converts `convaiinnovations/laya-multilingual` at `052592a15d198d9ad47da779604259b10b47b7aa` (upstream code `NandhaKishorM/laya` at `6a5819129eb220570792e417e49723d697efd76f`) by renaming parameters and keeping the FP16 weights; the porter reports FP16 agreeing with upstream PyTorch FP32 on 63/63 argmaxes, max calibrated probability difference 0.0012887. Arsenal ships 16-bit only since 26 October 2026: the 16-bit weights are 0.678 GB, under the 1 GB line below which int8 and int4 are not offered (`log/007-sixteen-bit-only.md`).

Runner: the native Swift decisions worker, Verdict's engine at `f8177ed` copied verbatim (Laya network, prompt, fast Metaspace-BPE tokenizer, windowed attention, small-M GEMM route; `Workers/Decisions/PORTED-FROM.md`), driven over ArsenalWire. Laya English's two folder kernels serve this model unchanged (`kernels/manifest.toml`, `log/003-folder-kernels.md`): in Exact, `splitk-exact` engages only for the head's linear2 (MLX does not split the encoder's 1152→768 projection) and passed the load-time gate bit for bit, single items 4.69 → 4.50 ms median, 202/202 answers identical; in Fast, `smallm-tile` on M5 at FP16, 4.47 → 4.10 ms, 0 changed answers, max |Δp| 0.0053. One pair per mode, so the size of the gain is not established. On the stock-MLX diagnosis path the library tokenizer encodes an empty string as one token where the fast tokenizer gives none; nothing else differs on 1,324 fixture texts (`log/004-native-link.md` in Laya English).

Gate (`gate/gate.toml`, `log/001-verdict-port-parity.md`, 2 October 2026, M5 Max): Verdict's Laya fixture regression, 202 oracle items per set, three sets, each judged alone over the wire. Exact: 0 mismatches at FP16, int8 and int4. Fast at FP16: max |Δp| 0.0053, 0 changed answers. Single-item median under the GPU lock (not a benchmark): FP16 4.41 ms; load 0.59 s; MLX peak 1673 MB at FP16. Typed accuracy 4/4 on the decisions suite; the quick suite run (`results/m5-max.json`, contended Mac) answered all four cases correctly at 1,070 batched questions per second.

Precision intake (`log/004`, `log/005`, `log/006`, 15 October 2026, M5 Max, busy Mac): with the default Load kernels, FP16 measured 5.744 ms per warm decision and 1.332 GB peak; the since-withdrawn int8 passed (max relative L2 0.0145, max |Δp| 0.0006, 0.836 GB) and int4 failed its component bounds (0.1015, 0.0095, 0.578 GB). The table figure above (3.47 ms) is the later intake result in `results/intake-m5-max-fp16-native-fast.json`. The schema 2 folder is recorded in `log/002-v2-folder.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: text with typed questions; the card says it is built on mmBERT-base. [Laya family model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: the card suggests `max_len=8192`, which Arsenal lacks; split long input. [Laya family model card, undated, first-hand]
- Arsenal call: `arsenal_run(model:"laya-multilingual", operation:"judge", input:{"items":["Bitte erstatten Sie meine Bestellung"],"questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}})`

**Good and bad**
- Good at: non-English intent; the card reports 0.451 across 13 MASSIVE languages against 0.306 for the English checkpoint. [Laya family model card, undated, first-hand]
- Bad at: negation. In issue #377, "My manager told me to close the account. I do not agree. Please do not cancel it." was answered cancel_account at 0.9998. [Laya issue #377, opened 2026-09-24, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: negated wording can flip to the destructive answer at high confidence (2 of 5 cancellation states wrong here, at 0.9501 and 0.9998). [Laya issue #377, opened 2026-09-24, first-hand] Fix or workaround: none confirmed; the maintainer advises validating negated wording and routing irreversible actions through review (2026-09-24).
- Complaint or known bug: more false alarms on benign and Spanish inputs since v0.4.0. The maintainer confirms v0.4.0 changed the Router default from english to multilingual for undetected-language states, but found fewer false alarms on their own test. Fix or workaround: not confirmed. The maintainer suggests `LAYA_DEFAULT_MODEL=english` or `"model":"english"` per request, which should restore 0.3.29 results if the diagnosis holds. [Laya issue #1058, opened 2026-10-08, first-hand; maintainer comment 2026-10-11]

**Evidence**
- Strength: thin. One maker card; the rest are user runs and issues.
- Sources re-opened: the family card (huggingface.co/convaiinnovations/laya, undated); issues #377, #555 and #1058 (github.com/NandhaKishorM/laya).
