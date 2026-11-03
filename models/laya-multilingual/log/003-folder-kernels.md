+++
hypothesis = "Laya English's two folder kernels (smallm-tile Fast, splitk-exact Exact) serve Laya Multilingual unchanged: the small-M tile kernel was already its Fast default (compiled in), and MLX splits K for its head linear2."
change = "kernels/ copied from laya-english (identical sources and manifest entries); the decisions runner no longer carries the tile kernel's source, so without this folder Multilingual Fast would lose it."
measurement = "laya-english/kernels/probe.py --model laya-multilingual --rounds 1 under scripts/mlock gpulock, M5 Max, fp16, 202 Verdict fixture items judged alone; one pair per mode (quick check, shared machine)."
verdict = "kept"
numbers = { exact_single_median_ms = "4.69 -> 4.50", exact_wire_answers_identical = "202/202", exact_load_gate = "bit-identical, 9 outputs", fast_single_median_ms = "4.47 -> 4.10", fast_changed_answers = 0, fast_max_dp = 0.0053, fast_load_gate_logits_rel = 0.0033 }
links = ["../kernels/manifest.toml", "../../laya-english/log/004-folder-kernel-smallm-tile.md", "../../laya-english/log/005-folder-kernel-splitk-exact.md", "../../../docs/evidence/mb-v2-kernels/probe-laya-multilingual.json"]
+++

# Folder kernels (from laya-english)

The encoder's mlp.Wo (1152→768) is not split by MLX (K < 2·768), so in Exact mode the kernel serves the head's
linear2 only; the load-time Exact gate saw it engage and pass bit for bit. One pair per mode: the direction matches
Laya English (−4 % Exact, −8 % Fast on single items), the size is not established by one pair.
