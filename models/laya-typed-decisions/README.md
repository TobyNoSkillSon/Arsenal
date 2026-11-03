# Laya · Typed decisions

Laya is a non-autoregressive "System 1" decision model by Convai Innovations; this is its typed-decisions checkpoint, the English Laya fine-tuned for four business workflows, run from aac6fef's native MLX FP16 conversion. It is a ModernBERT-large encoder with Laya's decision head (two transformer layers, an option-marker scorer and an act/escalate head), 421M parameters in all, with a 1,024-token context. Every option of a question is scored at its own `[MASK]` token and softmaxed over that question's options, so the answer space is set per request and every question in a call is answered in one forward pass. It returns typed answers with probabilities and never generates text.

## What it is for

The makers fine-tuned the English Laya on the 1,200-case training split of the typed-decisions benchmark (agent-trace observability, customer service, invoice processing and security incidents) with RLCD, a strictly proper scoring-rule reward, alongside soft cross-entropy against a teacher. On that benchmark's test split it scores 0.766 accuracy with a Brier score of 0.062, against 0.362 and 0.316 for the base English checkpoint. It is a specialist: outside those workflows the makers expect base-Laya behaviour or worse.

- Invoice processing decisions (the makers' best workflow, 0.804 accuracy).
- Security incident triage (0.766).
- Customer service routing and checks (0.764).
- Agent-trace observability: judging what an agent did in a trace (0.730).

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

Returns JSON `{"answers": {...}, "model": "laya-typed-decisions", "usage": {"input_tokens": n, "output_tokens": 0}}`. Each answer has `type` and:

- `noul`: `noul`, the probability that the statement holds, 0 to 1.
- `choice`: `choice` (the label with the highest probability), `probabilities` (label → probability, summing to 1 over the options you sent) and `confidence`.
- `score`: `score`, the expected level (the sum of level index × probability, so 1.4 means between level 1 and 2, nearer 1), `probabilities` (`"0"`, `"1"`, … → probability), `confidence`, and `legend` (`"0"`, `"1"`, … → the level descriptions exactly as sent).

How Laya reads a question: each question becomes a header `<type> question: <instructions>` followed by its options, each behind its own marker. A choice option is written `label: description`, or the bare label when the description is null; a score level is `level 0: <description>`, `level 1: …`; a noul has two options, `false: <false description>` and `true: <true description>`, which default to "no, the statement does not hold" and "yes, the statement holds". The state follows the header: a string as is, an object or array as its JSON text with keys in the order sent. So write the instructions as the question, put what each option means in its description, and keep labels short ids for your code. The header and its options share a budget of 256 tokens (`head_max_len`); each option is cut to 48 tokens, options are shortened further when many of them crowd the budget, and a question whose options cannot all fit is refused. The state gets what remains of the 1,024-token context: a state that needs more than the context together with the longest question is refused ("Item needs about N tokens … Shorten it or split it"), as a 422 in `decide` and a per-item error in `judge`. A literal `[MASK]` inside the state is replaced by a space.

Probabilities are a softmax of the option logits divided by the checkpoint's temperature for that question type and option count (from `rl_agent_config.json`), rounded to 4 decimals. The per-option-count temperatures were inherited from the base checkpoint and take precedence over the per-type temperatures fitted for this model, as in the makers' runtime. `confidence` for choice and score is 1 minus the distribution's entropy over its maximum (ln k for k options): 0 when the options are equally likely, 1 when one has all the probability. The makers report it as still over-confident (see Limits).

### `judge`

Asks the same questions about many items in one call; each item is its own state.

| Field | Type | Required | Notes |
|---|---|---|---|
| `items` | array | yes | At least 1. Strings, or any JSON value (objects are read in the key order sent). Media items get a per-item error. |
| `questions` | object | yes | As in `decide`. One batch extension: a `choice` question's `criteria` may also be a plain list of labels (strings). |
| `model` | string | no | As in `decide`. `"auto"` routes to laya-english or laya-multilingual, never to this model; omit `model` to have this model answer. |
| `bits` | integer or null | no | As in `decide`. |

Returns JSON `{"results": [...]}`, one result per item in item order. A result is `{"answers": {...}, "model": "laya-typed-decisions", "ms": n}`, where the answers carry the same `noul`, `choice`, `score`, `probabilities` and `confidence` values as `decide` without `type` and `legend` (noul answers here also carry `confidence`, the larger of P(yes) and P(no)), and `ms` is the batch's time divided by its items. An item that cannot be answered (too long, media) gets `{"error": "...", "model": ..., "ms": 0}` while the others are still answered. Use `judge` for lists; use `decide` for one state.

## Run it

MCP: `arsenal_run(model:"laya-typed-decisions", input:{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}})` (add `operation:"judge"` with `items` for a batch).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/laya-typed-decisions/ops/decide" -H 'Content-Type: application/json' -d '{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

TypeSafe System One route (also at `/providers/typesafe/v1/systemone`; batches at `/v1/judge`), which needs `model`: `curl -s "$ARSENAL_URL/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"laya-typed-decisions","state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

The only setting is Kernels (Tuned or Stock MLX, under More; setting `stock_mlx`, applied at Load): Stock MLX runs plain MLX without the fast tokenizer and kernels, with the same answers as Exact, slower, for diagnosis. Measured on an M5 Max at FP16 in a quick check on a busy Mac: 5.50 ms per warm decision, 1.19 GB peak worker memory (`results/intake-m5-max-fp16-native-fast.json`).

## Limits

- A specialist: fine-tuned on four synthetic workflows; on other tasks expect the base English checkpoint's behaviour or worse. Use Laya · English for general English decisions.
- English only; use Laya · Multilingual for other languages.
- Text only: image, audio and video state is refused.
- 1,024 tokens per question, shared by the question, its options and the state; the question and options get at most 256 of them.
- Keep `choice` questions under about 20 options: options share the 256-token budget and accuracy falls off sharply with many labels.
- Still over-confident: ECE 0.213 on the typed-decisions test split (the base English checkpoint: 0.175), because the inherited per-option-count temperatures override this model's own; the makers advise refitting on held-out data before relying on the probabilities. Arsenal uses the shipped temperatures.
- Its probability distributions match the teacher's less well than its top answers do (soft accuracy 0.471).
- `model:"auto"` never selects it; name it explicitly.

## Licence and credit

Apache-2.0: free to use, modify and redistribute, including commercially, provided the licence and `NOTICE` are kept and changes are stated; it includes a patent grant. Original Laya models and code by Convai Innovations and contributors; MLX conversion by aac6fef (laya-mlx). The makers publish no citation; credit "Laya, Convai Innovations".

## Links

- Model card: https://huggingface.co/aac6fef/laya-typed-decisions-mlx/blob/f9e501c2080cc57c13d6887820329758f5351125/README.md
- Original model card: https://huggingface.co/convaiinnovations/laya-typed-decisions
- Laya family: https://huggingface.co/convaiinnovations/laya
- Benchmark dataset: https://huggingface.co/datasets/LocalLLaMA/typed-decisions
- Code: https://github.com/NandhaKishorM/laya
- MLX port code: https://github.com/mizorewww/laya-mlx

## How Arsenal runs it

Source: `aac6fef/laya-typed-decisions-mlx` at `f9e501c2080cc57c13d6887820329758f5351125`, FP16 MLX safetensors (download 0.85 GB; exact files, sizes and hashes in `model.toml`). The card states it converts `convaiinnovations/laya-typed-decisions` at `f9ab0b228f0fc0f14d873dbc99038f135c2da1b2` (upstream code `NandhaKishorM/laya` at `6a5819129eb220570792e417e49723d697efd76f`) by renaming parameters and keeping the FP16 weights, with no retraining or further quantization. Arsenal ships 16-bit only since 26 October 2026: the 16-bit weights are 0.846 GB, under the 1 GB line below which int8 and int4 are not offered (`log/007-sixteen-bit-only.md`).

Runner: the native Swift decisions worker, Verdict's engine at `f8177ed` copied verbatim (Laya network, prompt, fast byte-BPE tokenizer, windowed attention, small-M GEMM route; `Workers/Decisions/PORTED-FROM.md`), driven over ArsenalWire. Laya English's two folder kernels serve this model unchanged, since the encoder and head shapes are the same (`kernels/manifest.toml`, `log/003-folder-kernels.md`): `splitk-exact` in Exact on every NAX chip, bit-identical at the load gate, single items 7.77 → 7.30 ms median, 202/202 answers identical; `smallm-tile` in Fast on M5 at FP16, 7.38 → 6.87 ms, 0 changed answers, max |Δp| 0.0033.

Gate (`gate/gate.toml`, `log/001-verdict-port-parity.md`, 2 October 2026, M5 Max): Verdict's Laya fixture regression, 202 oracle items per set, three sets, each judged alone over the wire. Exact: 0 mismatches at FP16, int8 and int4. Fast at FP16: max |Δp| 0.0033, 0 changed answers. Single-item median under the GPU lock (not a benchmark): FP16 6.98 ms; load 0.13 s; MLX peak 1792 MB at FP16. Typed accuracy 4/4 on the decisions suite.

Precision intake (`log/004`, `log/005`, `log/006`, 15 October 2026, M5 Max, busy Mac): with the default Load kernels, FP16 measured 8.406 ms per warm decision and 1.184 GB peak; the since-withdrawn int8 passed (max relative L2 0.0111, max |Δp| 0.0012, 0.744 GB) and int4 failed its component bounds (0.1806, 0.0352, 0.537 GB). The table figure above (5.50 ms) is the later intake result in `results/intake-m5-max-fp16-native-fast.json`. The schema 2 folder is recorded in `log/002-v2-folder.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: a state plus typed questions, for four business workflows: agent traces, customer service, invoices and security incidents. [Laya typed-decisions model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: keep `choice` questions under about 20 options. [Laya typed-decisions model card, undated, first-hand]
- Arsenal call: `arsenal_run(model:"laya-typed-decisions", input:{"state":{"invoice":"INV-204","amount":1280.5,"currency":"EUR","vendor_known":false},"questions":{"approve":{"type":"noul","instructions":"Should this invoice be approved for payment?"},"queue":{"type":"choice","instructions":"Which queue handles it?","criteria":{"ap":"accounts payable","fraud":"possible fraud","hold":"hold for review"}}}})`

**Good and bad**
- Good at: the card's 400-case test split gives accuracy 0.766 and Brier 0.062; by workflow, invoices 0.804, security 0.766, customer service 0.764, traces 0.730. [Laya typed-decisions model card, undated, first-hand]
- Bad at: not found in the card figures read. The base checkpoint scores 0.362 zero-shot, against a 0.318 random-guess row. [Laya typed-decisions model card, undated, first-hand]
- Against the obvious alternatives: a third-party Jev 1.13 row (accuracy 0.727) is ahead on soft accuracy (0.580 against 0.471) and ECE (0.144 against 0.213); the card calls these indicative. [Laya typed-decisions model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: `choice` questions with 11 or more options use the shipped temperature 0.1006, which the runtime clamps to 0.5 at load, replacing that bucket's calibration. Fix or workaround: none shipped. The owner says the configs need re-publishing with refitted values; 0.4.10 adds `temperature_clamped` to `fit_temperature_map`. A contributor's held-out fit put this checkpoint near 0.39. Arsenal's runner was not checked for this clamp. [Laya issue #1245, opened 2026-10-15, first-hand; owner comments 2026-10-17]

**Evidence**
- Strength: thin. Card figures and one bug thread; the newest comment read is 23 Oct 2026, and the re-publish was still pending.
- Sources re-opened: the typed-decisions card (huggingface.co/convaiinnovations/laya-typed-decisions, undated); issue #1245 (github.com/NandhaKishorM/laya).
