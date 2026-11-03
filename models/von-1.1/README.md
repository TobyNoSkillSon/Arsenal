# Von · 1.1

Von 1.1 is the earlier release of Victor Hugo Panisa's (wfzyx) open, non-autoregressive "System One" decision model. It is a 395M-parameter ModernBERT-large encoder with an Option-Marker scoring head: the state and every candidate option are packed into one sequence, each option gets a `[MASK]` marker that attends to the state and to the other options, and one bidirectional forward pass scores all options jointly. It generates no text. Because options see each other, its answers can change when the same options are listed in a different order; Von 1.2 removes that.

## What it is for

The maker built it for fast, calibrated structured decisions (pick an option, judge a condition, rate a level) at the cost of a single encoder pass, whatever the number of options. It is English only and strongest on short, well-posed decisions. Von 1.2 keeps the same design without the option-order dependence.

- Routing and triage of short English messages (which team, which intent).
- Yes/no checks on a text or a JSON record ("does the writer ask for a refund?").
- Rating a text on an ordered rubric (urgency, severity, sentiment level).
- Reproducing results from an existing Von 1.1 pipeline.

## Inputs and outputs

### `decide` (default)

Answers any number of named, typed questions about one state in one call.

| Field | Type | Required | Notes |
|---|---|---|---|
| `state` | string, object or array | yes | The content every question is about. Objects and arrays are read in the key order sent. Objects with `image`, `images`, `audio`, `video` or `videos` keys are refused (text only). |
| `questions` | object | yes | Question name → question, at least 1 (`minProperties` 1). Names come back as the answers' keys; they are not shown to the model. Each question is described below. |
| `model` | string | no | `minLength` 1. `"auto"` (laya-english when the text's letters are ≥ 99.5 % ASCII, else laya-multilingual) or a decisions model id. Omit it on this model's own op endpoint: this model answers. |
| `bits` | integer or null | no | One of `0`, `4`, `8`, `16`, `null`. The precision this request requires (0 = 16). A request never changes the loaded precision: another value is a 409. Von ships 16-bit only, so `0`, `16` or `null` pass. |
| `merge` | boolean | no | Verdict extension, accepted for compatibility. Arsenal never merges requests, so every answer is the one the request gets alone. |

A question is an object with these fields (unknown fields are ignored):

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | `noul` = yes/no (answer P(yes)); `choice` = one of named options; `score` = an ordered level. |
| `instructions` | string, object or array | no | The question itself, as text or structured JSON. |
| `criteria` | depends on `type` | `choice` and `score`: yes | `noul`: optional object with only `true` and `false`, each text, JSON or null. `choice`: object, option label → description (text, JSON, or null for the bare label), at least 1; labels are compared as exact strings and must be unique. `score`: array of level descriptions, lowest first, at least 1; levels are answered as `"0"`, `"1"`, …. |

Returns JSON `{"answers": {...}, "model": "von-1.1", "usage": {"input_tokens": n, "output_tokens": 0}}`. Each answer has `type` and:

- `noul`: `noul`, the probability that the condition holds, 0 to 1.
- `choice`: `choice` (the label with the highest probability), `probabilities` (label → probability, summing to 1 over the options you sent) and `confidence`.
- `score`: `score`, the expected level (the sum of level index × probability, so 1.4 means between level 1 and 2, nearer 1), `probabilities` (`"0"`, `"1"`, … → probability), `confidence`, and `legend` (`"0"`, `"1"`, … → the level descriptions exactly as sent).

How Von reads a question: the instructions and the state are joined into one premise (`instructions + " " + state`); a string state is used as is, an object becomes one `key: value` line per top-level field in the order sent. Then each option follows behind its own marker: a choice option is its description, or its label when the description is null; a score level is its description; a noul has two options, the `true` and `false` descriptions, which default to "Yes, condition holds true." and "No, condition is false.". A noul with no criteria also gets the von-sdk zero-shot correction: the bare question is scored once without the state and 0.7 × that bias is removed. So write the instructions as the question, put the meaning of each option in its description, and keep labels short ids for your code. Probabilities pass through the maker's calibration map, a bounded temperature that depends on the option distribution's entropy, the state length and the option count; it changes how confident the answer is, never which option wins. `confidence` for choice and score is the top probability minus the second, bounded to 0–1: 0 on a tie, 1 when one option has all of it. Probabilities are rounded to 4 decimals, `score` to 2 and `confidence` to 3.

The state, instructions and options of one question must fit in 2,048 tokens together; a longer request is refused with a 422 naming the token count. A literal `[MASK]` in the state, instructions or options is refused, because Von reserves it for option markers.

### `judge`

Asks the same questions about many items in one call; each item is its own state.

| Field | Type | Required | Notes |
|---|---|---|---|
| `items` | array | yes | At least 1. Strings, or any JSON value (objects are read in the key order sent). Media items get a per-item error. |
| `questions` | object | yes | As in `decide`. One batch extension: a `choice` question's `criteria` may also be a plain list of labels (strings), each used as its own description. |
| `model` | string | no | As in `decide`. |
| `bits` | integer or null | no | As in `decide`. |

Returns JSON `{"results": [...]}`, one result per item in item order. A result is `{"answers": {...}, "model": "von-1.1", "ms": n}`, where the answers carry the same `noul`, `choice`, `score`, `probabilities` and `confidence` values as `decide` without `type` and `legend` (noul answers here also carry `confidence`, the larger of P(yes) and P(no)), and `ms` is the batch's time divided by its items. An item that cannot be answered (too long, media) gets `{"error": "...", "model": ..., "ms": 0}` while the others are still answered. Use `judge` for lists (search hits, files, tickets); use `decide` for one state.

## Run it

MCP: `arsenal_run(model:"von-1.1", input:{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}})` (add `operation:"judge"` with `items` for a batch).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/von-1.1/ops/decide" -H 'Content-Type: application/json' -d '{"state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

TypeSafe System One route (also at `/providers/typesafe/v1/systemone`; batches at `/v1/judge`), which needs `model`: `curl -s "$ARSENAL_URL/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"von-1.1","state":"Please refund my order","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}}'`

The only setting is Kernels (Tuned or Stock MLX, under More; setting `stock_mlx`, applied at Load): Stock MLX runs plain MLX without the fast tokenizer and kernels, with the same answers, slower, for diagnosis. Measured on an M5 Max at BF16 in a quick check on a busy Mac: 6.65 ms per warm decision, 1.18 GB peak worker memory (`results/intake-m5-max-bf16-native-fast.json`).

## Limits

- English only.
- Text only: image, audio and video state is refused.
- Answers can change when choice options are reordered: on the JevBench option-order diagnostic in the maker's Von 1.2 card, 49.5 % of hard-tier answers changed across four orderings (Von 1.2: 0 %).
- 2,048 tokens per question (state, instructions and options together), from the pinned `config.json` (`max_position_embeddings` 2048).
- Near chance on long multi-hop legal or policy reasoning (JevBench hard tier 35.1 %); the maker advises against unsupervised use for high-stakes contract decisions.
- The calibration map was fitted on the 231 public JevBench items, so the maker's calibration figures are in-sample and may drift on very different domains.
- It selects and scores; it does not write text or explain an answer.
- The maker has archived the code repository and says the weights will not be updated.

## Licence and credit

Apache-2.0: free to use, modify and redistribute, including commercially, provided the licence and notices are kept and changes are stated; it includes a patent grant. Made by Victor Hugo Panisa (wfzyx). The maker asks for this citation:

```bibtex
@software{von2026,
  title  = {Von: An Open-Source System One Decision Model},
  author = {Panisa, Victor Hugo},
  year   = {2026},
  url    = {https://github.com/wfzyx/von}
}
```

## Links

- Model card: https://huggingface.co/wfzyx/von/blob/d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd/README.md
- Code: https://github.com/wfzyx/von

## How Arsenal runs it

Source: `wfzyx/von` at `d8bb5e0745d8ee1fb65d536d6d4892d54d5a93fd`, the vendor's FP32 PyTorch zip `option_marker.pt` plus config, tokenizer and `marker_calibration.json` (download 1.58 GB; exact sizes and hashes in `model.toml`). The repository's `model.safetensors` is not the file von-sdk reads, so it is not pinned or used. Get casts the FP32 file once to a BF16 view (`cast:von-bf16-2`); no FP32 learned tensor runs at inference. Arsenal ships 16-bit only since 26 October 2026: the 16-bit weights are 0.792 GB, under the 1 GB line below which int8 and int4 are not offered (`log/004-sixteen-bit-only.md`).

Runner: the native Swift decisions worker, a port of Verdict's Von engine at `f8177ed` (ModernBERT encoder and option-marker scorer, the PyTorch-zip reader, fast tokenizer and calibration; `Workers/Decisions/PORTED-FROM.md`), driven over ArsenalWire. Fast and Exact give the same answers on Von: no chip kernel qualified, so Fast equals Exact, and Exact equals the stock MLX path.

Gate (`gate/gate.toml`, `log/001-verdict-port-parity.md`, 2 October 2026, M5 Max): Verdict's pinned von-sdk 1.1.1 fixture (5 items × 3 questions) through the wire runner. FP32 max |Δp| 0.0001 against the SDK (lab only, not offered), FP16 0.0101, int8 0.0184, int4 0.9518 (Verdict's known 4-bit drift on near-ties), 0 violations at every precision; MLX peaks 2752, 1502, 1203 and 1031 MB; loads 0.21–0.27 s. Typed accuracy 4/4 on the decisions suite.

Half-precision intake (`log/003-half-intake.md`, 15–16 October 2026, M5 Max, busy Mac, quick check): FP16 failed the shared FP32-source policy, so BF16 is the native tier. BF16 6.651 ms per warm decision, 1.177 GB peak. The since-withdrawn tiers measured int8 max relative L2 0.0104 (pass, 6.555 ms, 0.735 GB) and int4 0.1145 (fail, 7.439 ms, 0.543 GB). The schema 2 folder (decide/judge ops from the kind's shared schemas, `stock_mlx` setting) is recorded in `log/002-v2-folder.md`: the schemas accept all 39 golden System One requests and 3 × 202 judge items and reject 8 malformed ones.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: a text state with choices, as in the card's `von.decide(state, choices)`. English only. Arsenal also takes noul and score questions. [Von 1.1 card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: for batch judging, pass criteria or describe each choice. [Von main card, undated, first-hand]
- Parameters people recommend: refit calibration with `benchmarks/fit_calibration.py` if you depend on confidence [Von 1.1 card, undated, first-hand]; the main card's `von calibrate labels.jsonl` refits on your own labels [Von main card, undated, first-hand].
- Arsenal call: `arsenal_run(model:"von-1.1", operation:"decide", input:{"state":"...","questions":{"refund":{"type":"noul","instructions":"Does the writer ask for money back?"}}})` [input table above, first-hand]

**Good and bad**
- Good at: short, well-posed English decisions. 93.8% on 48 easy JevBench items, the same items the calibration map was fitted on. [Von 1.1 card, undated, first-hand]
- Bad at: long multi-clause policy text, 35.1% on 111 hard items. [Von 1.1 card, undated, first-hand]
- Bad at: option order. The 1.2 card says 49.5% of hard-tier answers changed under reordering. [Von 1.2 card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: answers change with option order. Fix or workaround: Von 1.2, order-invariant (0 flips on the same items). [Von 1.2 card, undated, first-hand]
- Complaint or known bug: confidence may drift on other domains. Fix or workaround: refit with `benchmarks/fit_calibration.py`. [Von 1.1 card, undated, first-hand]

**Evidence**
- Strength: thin. Maker's own numbers on public items. The 1.1 card gives 35.1% on the hard split; the 1.2 card gives 38.7% for 1.1.
- The main card says the project is archived and the weights will not be updated.
