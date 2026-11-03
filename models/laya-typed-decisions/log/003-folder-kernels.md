+++
hypothesis = "Laya English's folder kernels serve Laya Typed Decisions unchanged (same encoder and head shapes)."
change = "kernels/ copied from laya-english (identical sources and manifest entries); the decisions runner no longer carries the tile kernel's source."
measurement = "laya-english/kernels/probe.py --model laya-typed-decisions --rounds 1 under scripts/mlock gpulock, M5 Max, fp16, 202 Verdict fixture items judged alone; one pair per mode (quick check, shared machine)."
verdict = "kept"
numbers = { exact_single_median_ms = "7.77 -> 7.30", exact_wire_answers_identical = "202/202", exact_load_gate = "bit-identical, 9 outputs", fast_single_median_ms = "7.38 -> 6.87", fast_changed_answers = 0, fast_max_dp = 0.0033, fast_load_gate_logits_rel = 0.0015 }
links = ["../kernels/manifest.toml", "../../laya-english/log/004-folder-kernel-smallm-tile.md", "../../laya-english/log/005-folder-kernel-splitk-exact.md", "../../../docs/evidence/mb-v2-kernels/probe-laya-typed-decisions.json"]
+++

# Folder kernels (from laya-english)

Same sites and shapes as Laya English. One pair per mode: −6 % Exact, −7 % Fast on single items, wire answers
byte-identical in Exact, 0 changed answers in Fast. See laya-english log 004/005 for the kernels' history.
