# Weight-free fixture for the image path from the maker capture (maker_capture_mm.py): per request the JSON text, per
# branch the rendered prompt, the processor's expanded ids, the prefix length, grids and M-RoPE positions, and every
# 97th normalised pixel value of the request's images (the processor's output). usage:
# python -I -B maker_mm_fixture.py <requests-mm.json> <mm-reference-bfloat16.json> <mm-components-bfloat16.json> <out.json>
import json, sys
reqs = json.load(open(sys.argv[1])); ref = {c["id"]: c for c in json.load(open(sys.argv[2]))["cases"]}; comps = json.load(open(sys.argv[3]))
out = []
for r in reqs:
    c = ref[r["id"]]
    pix = comps.get(f"{r['id']}.pixel_values", [])
    vpix = comps.get(f"{r['id']}.pixel_values_videos", [])
    out.append({"id": r["id"], "request": json.dumps({k: r[k] for k in ("state", "questions", "media")}, ensure_ascii=False),
                "branches": [{k: b.get(k, []) for k in ("prompt", "input_ids", "prefix_tokens", "image_grid_thw", "video_grid_thw", "positions", "rope_delta")} for b in c["branches"]],
                "slot_logits_bf16": [b["slot_logits_blocks"] for b in c["branches"]],
                "pixels": {"count": len(pix), "stride": 97, "values": pix[::97]},
                "video_pixels": {"count": len(vpix), "stride": 97, "values": vpix[::97]}})
json.dump({"source": "OneJev qev 81ce62f + serve.patch, train.common.batch_scores (multimodal, noncausal); transformers 5.19.0 Qwen3VLProcessor", "cases": out},
          open(sys.argv[4], "w"), ensure_ascii=False)
