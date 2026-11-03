+++
hypothesis = "The Verdict port can carry the v2 folder contract (settings, typed ops, selection, browser data) without touching its qualified numerical path."
change = "schema 2: decide/judge ops with TypeSafe System One / Verdict judge JSON Schemas (kinds/decisions/ops), stock_mlx load setting (config.settings.stock_mlx = Verdict's Standard recipe), [selection] default native, facets text→text, suggested, gate/runner-check.json samples."
measurement = "arsenal-catalog validate passes; the input schemas accept all 39 golden System One requests and Verdict's 3 × 202-item judge fixtures, the output schemas accept their responses, and 8 malformed requests are rejected (docs/evidence/v2-decisions/op-schema-check.txt, jsonschema 4.25.1 and Arsenal's own validator)."
verdict = "kept"
numbers = { golden_requests = 39, judge_items = 606, rejected_negatives = 8, dtype_changes = 0, performance_claims = 0 }
links = ["../../kinds/decisions/ops/decide.input.json", "../../kinds/decisions/ops/judge.input.json", "gate/runner-check.json", "../../docs/evidence/v2-decisions/op-schema-check.txt"]
+++

# V2 folder

Ops are declared in the folder and reference the kind's one shared copy of the schemas (`kinds/decisions/ops/`). The first real app cycle found that Get validated the installed folder without the kinds directory; per-folder copies bridged that until the service passed the verified shared kinds root (606e8ae), after which they were removed again.
The schemas describe valid input for the universal op endpoint and OpenAPI; Verdict's own validator in the worker still produces the
byte-exact FastAPI 422/409 bodies, so the System One and judge dialect routes pass requests through unvalidated (key order and
Unicode-distinct labels matter to the model; see docs/notes/decisions.md).

Settings: Fast/Exact and dtype are the app's own controls, so the only per-model load setting is Verdict's third recipe, Standard
(plain MLX, library tokenizer, dense attention): same answers as Exact by Verdict's definition, slower, useful for diagnosis.
Verdict's request extensions (`bits`, `merge`) stay request fields; Arsenal runs one request per pass, so `merge` changes nothing.

`suggested = true`: Laya · English is the decisions pick for the browser (best English accuracy on Verdict's suite; Verdict's own default for English text).
