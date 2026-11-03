# The maker's own DecisionEngine.decide (OneJev qev @81ce62f + serve.patch) with the model forward replaced by fixed slot
# logits, so the branch plan (including the chunked path for a choice wider than the slots) and the answer assembly
# can be checked without weights. usage: python -I -B maker_decide_mapping.py <onejev> <out.json>
import json, math, sys
sys.path.insert(0, sys.argv[1])
from qev.calibrate import Calibration
from qev.engine import BranchResult, DecisionEngine
from qev.prompt import DEFAULT_STYLE
from qev.schema import SystemOneRequest

request_text = json.dumps({"state": {"note": "fixed"}, "questions": {
    "yes": {"type": "noul", "instructions": "Is it?", "criteria": {"true": "it is", "extra": 1}},
    "wide": {"type": "choice", "instructions": "Which?", "criteria": {f"o{i}": (None if i % 2 else f"option {i}") for i in range(7)}},
    "level": {"type": "score", "criteria": ["low", None, {"x": 1}]},
    "pair": {"type": "choice", "criteria": {"b": None, "a": "first"}}}}, ensure_ascii=False)
engine = DecisionEngine.__new__(DecisionEngine)
engine.max_slots, engine.calibration, engine.style = 4, Calibration(), DEFAULT_STYLE
engine.max_branch_tokens, engine.max_request_tokens = 32768, 65536
engine.info = type("Info", (), {"fork_mode": "sequential"})()
engine._render_prompt = lambda state, suffix: suffix
captured = {}
def split(state, rendered):
    captured["suffixes"] = [r.suffix for r in rendered]
    return [1, 2], [[3]] * len(rendered)
def run(state, rendered):
    out = []
    for b, r in enumerate(rendered):
        logits = [math.sin(1.7 * b + 0.9 * j) * (2 + b) for j in range(r.n_slots)]
        captured.setdefault("logits", []).append(logits)
        out.append(BranchResult(r.qid, r.kind, r.labels, r.descriptions, logits, 1.0, 1))
    return out
engine._prefix_and_suffix_ids, engine._run_sequential = split, run
request = SystemOneRequest.model_validate_json(request_text)
response, _ = engine.decide(request)
answers = {k: a.model_dump() for k, a in response.answers.items()}
json.dump({"request": request_text, "max_slots": 4, "suffixes": captured["suffixes"], "logits": captured["logits"],
           "answers": json.dumps(answers, ensure_ascii=False, sort_keys=True, separators=(",", ":"))}, open(sys.argv[2], "w"), ensure_ascii=False)
