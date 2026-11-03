# Maker reference for elephant-1's image path, as the maker trained and evaluates it: OneJev @81ce62f + the checkpoint's
# serve.patch, train.common.batch_scores for a multimodal row (train/eval_mm.py with the readout attached): the qev prompt
# with each <image:N> placeholder as a vision block (train.mm.render_prompt_mm), the Qwen3VLProcessor, one forward of the
# whole prompt with block ids [prefix incl. the image tokens | question suffix] in the full-attention layers, slot_lm
# readout from float32 slot rows. The unpatched qev mm_engine path (causal) is recorded as a diagnostic.
# usage: python -I -B capture_mm.py <onejev> <snapshot> <requests.json> <media dir> <out dir> <bfloat16|float16>
import json, sys, time
from pathlib import Path

onejev, snapshot, requests, media_dir, out, dtype = sys.argv[1:7]
sys.path.insert(0, onejev)
import torch  # noqa: E402
import transformers  # noqa: E402
from qev import readout as ro  # noqa: E402
from qev.prompt import build_messages, render_question, SLOT_LABELS, LETTERS  # noqa: E402
from qev.schema import SystemOneRequest  # noqa: E402
from train import mm  # noqa: E402

mm.set_root(media_dir)
out = Path(out); out.mkdir(parents=True, exist_ok=True)
dev = torch.device("mps")
t0 = time.time()
processor = transformers.AutoProcessor.from_pretrained(snapshot, local_files_only=True)
tok = processor.tokenizer
model = transformers.Qwen3_5ForConditionalGeneration.from_pretrained(snapshot, dtype=getattr(torch, dtype), device_map={"": "mps"},
                                                                      local_files_only=True, low_cpu_mem_usage=True).eval()
rcfg = ro.ReadoutConfig.load(snapshot)
assert rcfg.noncausal and not rcfg.head, rcfg
ro.install_noncausal(model)
head = model.get_output_embeddings().weight
def one(label):
    ids = tok.encode(label, add_special_tokens=False)
    return ids[0] if len(ids) == 1 and tok.decode(ids) == label else None
slot_ids = [one(l) for l in SLOT_LABELS]
assert all(i is not None for i in slot_ids)
slot_w = head.detach()[torch.tensor(slot_ids, device=head.device)].float()
counter = mm.TokenCounter(processor)
print("loaded", time.time() - t0, type(processor.image_processor).__name__, file=sys.stderr)

stats = []
def hook(i):
    def f(m, a, o):
        h = o[0] if isinstance(o, tuple) else o
        stats.append((i, h[0, -1].float().cpu()))
    return f
layers = model.model.language_model.layers
handles = [l.register_forward_hook(hook(i)) for i, l in enumerate(layers)]
vis = {}
vh = model.model.visual.register_forward_hook(lambda m, a, o: vis.setdefault("out", []).append(
    (o.pooler_output if hasattr(o, "pooler_output") else o[1] if isinstance(o, tuple) else o).float().cpu()))

records, comps = [], {}
for req in json.loads(Path(requests).read_text()):
    from PIL import Image
    media = []
    for m in req["media"]:
        with Image.open(Path(media_dir) / (m["path"] if m["type"] == "image" else m["frames"][0])) as im:
            media.append({**m, "w": im.width, "h": im.height})
    request = SystemOneRequest(state=req["state"], questions=req["questions"])
    rendered = [render_question(k, q) for k, q in request.questions.items()]
    rec = {"id": req["id"], "branches": []}
    for bi, r in enumerate(rendered):
        prompt = mm.render_prompt_mm(processor, {"enable_thinking": False}, build_messages, request.state, r.suffix, media)
        suffix_start = prompt.rindex(r.suffix)
        inputs, ends = mm.mm_inputs(processor, [prompt], [media], dev)
        ids, prefix, _ = ro.token_layout(tok, prompt, suffix_start, None)
        shift = counter.media_extra(media)
        assert len(ids) + shift == ends[0] + 1, (len(ids), shift, ends)
        P = prefix + shift
        width = inputs["input_ids"].shape[1]
        branch = {"prompt": prompt, "input_ids": inputs["input_ids"][0].tolist(), "prefix_tokens": P,
                  "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else [],
                  "video_grid_thw": inputs["video_grid_thw"].tolist() if "video_grid_thw" in inputs else []}
        grids = {k: inputs[k] for k in ("image_grid_thw", "video_grid_thw") if k in inputs}
        if "mm_token_type_ids" in inputs:
            pos, delta = model.model.get_rope_index(inputs["input_ids"], mm_token_type_ids=inputs["mm_token_type_ids"],
                                                    attention_mask=inputs["attention_mask"], **grids)
        else:
            pos, delta = model.model.get_rope_index(inputs["input_ids"], image_grid_thw=inputs["image_grid_thw"],
                                                    attention_mask=inputs["attention_mask"])
        branch["positions"] = pos[:, 0].tolist(); branch["rope_delta"] = int(delta.view(-1)[0])
        with torch.inference_mode():
            for mode in ("blocks", "causal"):
                stats.clear(); vis.clear()
                ctx = ro.blocks(ro.block_ids([P], [ends[0] + 1], width, dev)) if mode == "blocks" else ro.blocks(None)
                with ctx:
                    o = model.model(**inputs, use_cache=False, return_dict=True)
                h = o.last_hidden_state[0, -1].float()
                logits = (slot_w[: len(r.labels)] @ h).tolist()
                branch[f"slot_logits_{mode}"] = logits
                if mode == "blocks" and bi == 0:
                    key = req["id"]
                    for pk in ("pixel_values", "pixel_values_videos"):
                        if pk in inputs:
                            comps[f"{key}.{pk}"] = inputs[pk].float().cpu().flatten().tolist()
                    comps[f"{key}.vision"] = torch.cat(vis["out"], 0).flatten().tolist() if vis.get("out") else []
                    for i, v in stats:
                        comps[f"{key}.layer{i}"] = v.tolist()
                    comps[f"{key}.norm"] = h.cpu().tolist()
                    branch["max_abs_layers"] = max(float(v.abs().max()) for _, v in stats)
        rec["branches"].append(branch)
        print(req["id"], bi, "P", P, "of", width, "blocks", [round(x, 3) for x in branch["slot_logits_blocks"]],
              "causal", [round(x, 3) for x in branch["slot_logits_causal"]], file=sys.stderr)
    records.append(rec)
for h in handles + [vh]:
    h.remove()
if dtype == "bfloat16":
    # Diagnostic only: the vision tower in float32 on the GPU, on the processor's own pixels, per request.
    model.model.visual.float()
    with torch.inference_mode():
        for req in json.loads(Path(requests).read_text()):
            media = []
            for m in req["media"]:
                with Image.open(Path(media_dir) / (m["path"] if m["type"] == "image" else m["frames"][0])) as im:
                    media.append({**m, "w": im.width, "h": im.height})
            request = SystemOneRequest(state=req["state"], questions=req["questions"])
            r0 = render_question(*next(iter(request.questions.items())))
            prompt = mm.render_prompt_mm(processor, {"enable_thinking": False}, build_messages, request.state, r0.suffix, media)
            inputs, _ = mm.mm_inputs(processor, [prompt], [media], dev)
            parts = []
            if "pixel_values" in inputs:
                parts += list(model.model.get_image_features(inputs["pixel_values"].float(), inputs["image_grid_thw"], return_dict=True).pooler_output)
            if "pixel_values_videos" in inputs:
                parts += list(model.model.get_video_features(inputs["pixel_values_videos"].float(), inputs["video_grid_thw"], return_dict=True).pooler_output)
            comps[f"{req['id']}.vision_fp32"] = torch.cat(parts, 0).float().cpu().flatten().tolist()
env = {"torch": torch.__version__, "transformers": transformers.__version__, "dtype": dtype, "device": "mps",
       "image_processor": type(processor.image_processor).__name__, "slot_ids_head": slot_ids[:4],
       "onejev": "81ce62f1597c91e46767d4d02d6ac2e18534fe94+serve.patch", "path": "train.common.batch_scores (multimodal, noncausal, slot_lm)"}
(out / f"mm-reference-{dtype}.json").write_text(json.dumps({"env": env, "cases": records}, ensure_ascii=False))
(out / f"mm-components-{dtype}.json").write_text(json.dumps(comps))
print("done", time.time() - t0, file=sys.stderr)
