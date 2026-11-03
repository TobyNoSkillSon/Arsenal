# Arsenal decisions: writing questions

Typed decisions (`decide` for one state, `judge` for many items) over the decision models. Each model's README (`arsenal_model(model:ID, docs:true)`) has its exact question format, context limit and languages; this page is how to ask well.

Use for a fairly long instruction plus a few defined answers, repeated over thousands of text items (or images with d1-3B): shortlist, route, classify or score. Give each item the evidence needed for one focused judgment. Use code for exact matching, numeric requirements, dates and counts; read sources or use a reasoning model for cross-item arguments, missing context and consequential verification. Typed answers are an interface guarantee, not a truth guarantee.

### Pick a decision model

| Model ID | Role and limits |
|---|---|
| `d1-3b` | Text/JSON or images; 32,768 tokens, 16 advertised languages including Polish. Vendor bf16, 3.12B, on MLX. Pick over Laya for images or context above 8,192 tokens. Fresh Polish job Noul: 24/32 versus Laya 17/32; no general quality advantage (see below). |
| `laya-multilingual` | Non-English or mixed-language text, including Polish written without diacritics; 1,024-token context. First candidate for semantic filtering of Polish ads. |
| `laya-english` | General English decisions; 512 tokens. |
| `laya-typed-decisions` | English workflow decisions (tickets, invoices, incidents); 1,024 tokens. Evaluate on your data before replacing another model. |
| `von-1.2` | English alternative, scores options independently of their order; 8,192 tokens. |
| `von-1.1` | Earlier English alternative; 2,048 tokens and option-order sensitivity. Prefer 1.2 for new workflows. |

Context includes state and questions. Oversize items receive errors rather than truncation. HTTP/Python `model:"auto"` requires both Laya English and Multilingual installed; the worker chooses English when at least 99.5% of the state's letters are ASCII, otherwise Multilingual. This is a character heuristic, not language detection: select Multilingual explicitly for ASCII Polish. Jev is TypeSafe's hosted model, not an Arsenal model ID. No local English-model superiority is established by the Polish validation below.

### Write the questions

These rules adapt TypeSafe AI's MIT agent skill and its primitives/confidence/cookbooks guidance:

- Put source, query, constraints and current facts in named JSON state fields. Put the judgment in `instructions` and answer meanings in `criteria`. Question IDs only identify answers for code; include the full meaning in the question.
- Ask one coherent judgment per question. A long instruction may define context, distinctions and exclusions; independent dimensions belong in separate questions. Combine their answers in code.
- `Choice` selects among competing options. Describe the distinctions and include a no-match option in every Choice (`other`, `none` or `unclear`). Keep the option set small enough to test; neither option count nor schema acceptance proves accuracy.
- `Noul` returns P(yes) from 0 to 1, optionally with `criteria:{"true":...,"false":...}`. It measures probability of a proposition, not intensity. It is not inherently more reliable than Choice.
- `Score` returns the weighted level index, 0 through `len(criteria)-1`, plus probabilities and a legend. Describe each ordered level as a concrete situation that stands on its own. Normalize different-length rubrics before combining them in code.
- Ask independent questions about the same state together. They cannot read each other's answers. A second request is needed when an earlier result supplies new evidence or options.
- Preserve probabilities. Choice/Score `confidence` summarizes distribution concentration, not correctness or permission. Arsenal's legacy Noul replies may also include `confidence`; use `noul` itself for P(yes), not that field as a separate quality estimate.
- Code owns top-k selection, weights and thresholds. For semantic shortlisting, try ranking by the relevant option's probability and read the shortlist. Validate cutoff/ranking on labelled examples from the user's data, including negatives and boundary cases; a universal 0.5, 0.85 or 0.9 cutoff has no evidence here. `calibrate()` fits the provided examples: assess a chosen cutoff on separate held-out cases before relying on it.

### Calls and answers

Single-item MCP example; `criteria` gives the meanings, not just labels:

```text
arsenal_run(model:"laya-multilingual", input:{
  "state":{"stanowisko":"Serwisant","obowiazki":"Diagnostyka i naprawa maszyn u klientów"},
  "questions":{"kind":{"type":"choice","instructions":"Jaki to rodzaj pracy?",
    "criteria":{"serwis":"serwis, naprawa, konserwacja lub instalacja urządzeń",
                "produkcja":"produkcja nowych wyrobów lub obsługa maszyn produkcyjnych",
                "other":"inna praca albo brak informacji pozwalającej dopasować kategorię"}}}})
```

For many items, use `arsenal_run(model:"laya-multilingual", operation:"judge", input:{"items":[STATE1,STATE2],"questions":QUESTIONS})`; the answer has one line per item. Reuse the same questions for the whole pile. Scripts use the raw batch endpoint:

```sh
curl --fail-with-body -sS "$ARSENAL_URL/v1/judge" -H 'Content-Type: application/json' \
  -d '{"model":"laya-multilingual","items":[{"opis":"Eksport nie działa"},{"opis":"Dziękuję, problem rozwiązany"}],"questions":{"bug":{"type":"noul","instructions":"Czy opis zgłasza nierozwiązany problem?"}}}'
```

Single state: `POST /v1/systemone` with `{"model":ID,"state":STATE,"questions":QUESTIONS}`. Both also exist under `/providers/typesafe/v1/...`; TypeSafe SDK clients use base `$ARSENAL_URL/providers/typesafe` with dummy key `local`. Universal equivalents: `POST /arsenal/v1/models/ID/ops/decide` or `/ops/judge`, omitting `model` inside the input so the endpoint's model answers.

Single responses contain `answers`, `model`, `usage`. A Noul answer has `noul`; a Choice has `choice`, `probabilities`, `confidence`; a Score has `score`, `probabilities`, `confidence`, `legend`. Batch responses have `results` in input order, each with `answers`, `model`, `ms`, or `error`. Keep errors separate from low scores; verify one result per item before applying decisions. A request with `bits` requires that loaded precision and returns 409 on mismatch; it never changes the shared dtype.

d1-3B image input is `images:[{"path":"/absolute/picture.png"}]` beside `state` and `questions`; use `state:null` for an image alone. Batch `images` is a list of image lists matching the order and count of `items`. No image URLs or audio/video. For reproducible cross-interface probabilities, use consistently ordered inputs: MCP sorts object keys, while raw HTTP preserves received order; the vendor renders that order into its prompt. Arrays retain order. This model uses LFM Open License v1.0: commercial use is subject to the US$10M annual-revenue threshold; inspect the pinned terms. It is a decision model and does not generate prose.

### Measured weak spots

d1-3B runs a Swift port of LiquidAI's System One prompt and readout over an MLX bf16 forward, pictures included, in the native worker. Its historical 8 Oct gate compared it with pinned vendor code run in float32 on the CPU: maximum absolute probability difference 0.0039 and 24/24 answer argmax agreement across Noul, Choice, Score, batch and image inputs (M5 Max, 8 Oct 2026). That retained result proves fidelity on those cases, not job-ad accuracy. New work follows the quality gate in the repository's `AGENTS.md` (Precision and the quality gate): independent GPU half-precision references or vendor-published outputs; the historical CPU/FP32 route is not a procedure to repeat. The earlier torch MPS runner and its reported 0.0 drift (MPS compared with itself) are superseded: torch 2.12's MPS causal attention is wrong in bf16 beyond ~100 tokens, which moved some answers by up to 0.73 in probability.

Fresh comparison, same chip/date: d1 bf16 on MLX versus Laya Multilingual fp16 Fast. Inputs and manual labels were saved before inference: 32 pracuj.pl ads (16 service/installation), plus 24 sections mentioning SEP (11 mandatory). Both used the exact questions retained in the probe. Polish job Noul got 24/32 versus Laya 17/32 (the historical vendor FP32 reference got 25/32; an earlier 27/32 came from the faulty torch MPS kernel); six-way Choice winning labels got 24/32 versus 23/32, finding 15/16 versus 12/16 service jobs. Ranked top 16 by Choice found 12/16 versus 11/16. Packaged HTTP timing on M5 Max, 8 Oct 2026, Mac in use, not a quiet benchmark: one decision (one ad, three questions) about 48 ms (the earlier torch runner: 123 ms); 32 ads × 3 questions 40 ms per ad (103); 32 ads × 1 question 24 ms per ad (80); a one-line state with one question about 21 ms; Laya about 12.6 ms per ad. d1's worker holds about 6.5 GB; a first call that has to load it takes about 6.4 s.

For full requirements, SEP presence got 24/24 versus Laya 20/24, but every section mentioned SEP, so false positives were not tested. Mandatory-SEP Noul got 20/24 versus 11/24, detecting 8/11 mandatory cases versus 9/11. SEP Choice got 13/24 versus 19/24 and found 0/11 mandatory cases versus 9/11. Excerpts did not rescue d1's Choice (0/11); its mandatory Noul got 12/24 and found 4/11. Use d1 for images/longer context and evaluate focused quality alternatives; keep exact text checks and source reading for requirements. This diagnostic sample does not establish a general quality winner.

Validation (pre-release build): `laya-multilingual` fp16 Fast, M5 Max, 7 Oct 2026, real Polish pracuj.pl ads. A manually labelled 32-ad subset had 16 service/installation jobs and 16 other jobs. English Noul at 0.5 got 16/32 right; Polish Noul got 17/32: both answered yes for 30–31 of 32 ads, so a yes/no question here sorted nothing. Polish six-way Choice with a no-match outcome got 26/32 right by winning label. Sorting its service probability put 14/16 true service jobs in the top 16; a fixed P(service)>0.5 retained 10/16 and classified 25/32 correctly. This supports Choice plus source review for this shortlist, not universal calibration or perfect exclusion.

On 24 full requirement sections, a SEP-name presence Noul got 20/24 right. Distinguishing mandatory SEP from preferred/obtainable/absent SEP was much worse: Noul 2/24 (yes for all 24; answering "no" every time would have scored 22/24), Choice 8/24; on shorter excerpts, 3/24 and 7/24. There were only 2 mandatory positives. Language change and shorter text are candidate tests, not guaranteed fixes. Use exact text checks for presence and inspect the wording for mandatory versus optional requirements. The Typed model was not installed at validation, so no Multilingual-versus-Typed quality comparison was run.

Other traps: negation, sarcasm, niche terminology, manipulated source text and knowledge outside the item can produce confident errors. Inspect representative kept/dropped items and boundary cases before closing a triage task. Keep enough candidates for review; do not silently discard an item that failed inference.

Sources: the 7 Oct validation inputs, labels and question definitions were a private sample of public job listings and are not distributed; those historical counts are separate from the fresh comparison above. A 200-listing triage run on the same app, model, chip and date took 0.6–1.3 seconds; that is task evidence, not a benchmark or latency promise for other inputs. New correctness calls used a GPU lock, not a quiet performance measurement. TypeSafe guidance: https://docs.typesafe.ai/primitives · https://docs.typesafe.ai/confidence · https://docs.typesafe.ai/cookbooks/parallel_questions · https://docs.typesafe.ai/cookbooks/rerank_typesafe. Arsenal-specific quality results do not transfer to hosted Jev.

## TypeSafe attribution

Decision-question guidance adapted from TypeSafe AI's official `typesafe-ai` agent skill (MIT), Copyright (c) 2026 TypeSafe AI. https://docs.typesafe.ai/agent-skill

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
