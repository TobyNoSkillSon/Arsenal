# Maker reference for elephant-1: OneJev qev @81ce62f + the checkpoint's serve.patch, run as the maker serves it
# (DecisionEngine, sequential branches = the exact path), on the GPU (MPS) at half precision. Captured once.
# usage: python -I -B maker_capture.py <onejev> <snapshot> <fixtures.json> <out_dir> <dtype: bfloat16|float16>
import json
import sys
import time
from pathlib import Path

onejev, snapshot, fixtures, out, dtype = sys.argv[1:6]
sys.path.insert(0, onejev)

import torch  # noqa: E402
from qev import readout as ro  # noqa: E402
from qev.engine import DecisionEngine  # noqa: E402
from qev.prompt import render_question  # noqa: E402
from qev.schema import SystemOneRequest  # noqa: E402

out = Path(out)
out.mkdir(parents=True, exist_ok=True)
t0 = time.time()
engine = DecisionEngine(snapshot, device="mps", dtype=dtype, fork_mode="sequential")
print("loaded", time.time() - t0, engine.info, file=sys.stderr)
model = engine.model
layers = model.model.layers

# Per-layer residual stream: max |h| over every token (FP16 range check) and the last token's hidden state.
stats: dict = {}


def hook(i):
    def f(module, args, output):
        h = output[0] if isinstance(output, tuple) else output
        s = stats.setdefault("pass", [])
        s.append((i, float(h.float().abs().max()), h[0, -1].float().cpu()))
    return f


handles = [layer.register_forward_hook(hook(i)) for i, layer in enumerate(layers)]
emb_handle = model.model.embed_tokens.register_forward_hook(
    lambda m, a, o: stats.setdefault("emb", []).append(o[0, -1].float().cpu()))
norm_handle = ro.final_norm(model).register_forward_hook(
    lambda m, a, o: stats.setdefault("norm", []).append(o[0, -1].float().cpu()))

cases = json.loads(Path(fixtures).read_text())
records = []
tensors = {}
for case in cases:
    request = SystemOneRequest(state=case["state"], questions=case["questions"])
    rendered = [render_question(k, q, style=engine.style) for k, q in request.questions.items()]
    prefix_ids, suffixes, spans = engine._split(request.state, rendered)
    prompts = [engine._render_prompt(request.state, r.suffix) for r in rendered]
    stats.clear()
    branches = engine._run_sequential(request.state, rendered)
    passes = stats["pass"]
    n_layers = len(layers)
    assert len(passes) == n_layers * (1 + len(rendered)), (len(passes), n_layers, len(rendered))
    maxabs = [max(p[1] for p in passes[j::n_layers]) for j in range(n_layers)]
    stats.clear()
    resp, meta = engine.decide(request, debug=True)
    rec = {
        "id": case["id"],
        "prompts": prompts,
        "prefix_ids": prefix_ids,
        "suffix_ids": suffixes,
        "slot_logits": [b.logits for b in branches],
        "allowed_mass": [b.allowed_mass for b in branches],
        "answers": {k: a.model_dump() for k, a in resp.answers.items()},
        "usage": resp.usage.model_dump(),
        "meta": {k: v for k, v in meta.items() if k not in ("timing", "diagnostics")},
        "layer_maxabs": maxabs,
    }
    records.append(rec)
    print(case["id"], json.dumps(rec["answers"])[:300], file=sys.stderr)

# Component fingerprints on the first fixture's prefix + first branch: embedding, every layer and the final norm at the
# last token of each pass (prefix pass, then the suffix pass).
case = cases[0]
request = SystemOneRequest(state=case["state"], questions=case["questions"])
rendered = [render_question(k, q, style=engine.style) for k, q in request.questions.items()]
stats.clear()
engine._run_sequential(request.state, rendered[:1])
for p, (i, _, h) in enumerate(stats["pass"]):
    tensors[f"pass{p // len(layers)}.layer{i}"] = h.tolist()
for p, h in enumerate(stats["emb"]):
    tensors[f"pass{p}.embed"] = h.tolist()
for p, h in enumerate(stats["norm"]):
    tensors[f"pass{p}.norm"] = h.tolist()

for h in handles + [emb_handle, norm_handle]:
    h.remove()

env = {"torch": torch.__version__, "dtype": dtype, "device": "mps", "slot_ids": engine.slot_ids, "max_slots": engine.max_slots,
       "readout": engine.info.extra, "fork_mode": engine.info.fork_mode, "onejev": "81ce62f1597c91e46767d4d02d6ac2e18534fe94+serve.patch",
       "snapshot": snapshot}
import transformers  # noqa: E402
env["transformers"] = transformers.__version__
(out / f"reference-{dtype}.json").write_text(json.dumps({"env": env, "cases": records}, ensure_ascii=False))
(out / f"components-{dtype}.json").write_text(json.dumps(tensors))
print("done", time.time() - t0, file=sys.stderr)
