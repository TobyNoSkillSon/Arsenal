#!/usr/bin/env python3
"""import-quick-checks.py: turn a night's peak.py measurements into Models-table quick checks.

    python3 -I scripts/import-quick-checks.py <peak-results.jsonl> [--repo DIR] [--run NAME] [--gpu-cores N] [--local-models DIR] [--dry-run]

Reads JSON lines written by lab/night-*/peak.py (model, dtype, date, chip, peak_footprint_mb = the workers' macOS
phys_footprint_peak in MiB, summed over all Arsenal worker processes (only the measured
model was loaded), across one Load and two requests, ops: {op: {seconds_warm, status, ...}}, error) and writes,
for each model folder under models/ or planned-models/ (kit folders, shipped or planned):

  results/quick-checks/<run>.json                       the rows used, as recorded (schema "quick-check-1"; not shipped)
  results/quick-<chip>-<dtype>-<runner>-<mode>.json     schema-2 table result, state quickCheck, bound to that record

Figures: peak memory for every kind; the warm request time as the speed figure for operation kinds (audio, embeddings,
perception, restoration, geometry) and as seconds per image for an image model whose operation is not generation.
A folder whose own kind declares no table figures (a kit kind: local.document, local.forecast, …) gets no table file:
its folder, installed under local-models/, is bound byte for byte to its owner's code approval. Its figures (peak memory
and the warm request time) go to Resources/figures/folder-quick-checks.json, which Arsenal ships and binds to the model
id, kind, weights revision, precision and runner (FolderReferenceFigures); the record stays in the repo's kit folder.
--local-models names a folder of installed models to read model.toml from (read only) for a model the repo has no
folder for; its entry cites the measurements file.
Rows with an error, without a peak, or without a completed (HTTP 200) request are skipped; the last qualifying row of a
(model, dtype) wins, so a later failed run never discards an earlier good one. A (model, dtype) whose other results
(any file this script did not write, in any mode) already carry a quick-check or measured figure for the same runner
is left alone, and a file of the same name that this script did not write is never overwritten. Rerunnable: the script
rewrites only its own files (those bound to results/quick-checks/), and a rerun under the same run merges its rows into
the existing record (a precision it measured again replaces its old row), so earlier results bound to it stay valid.
"""
import argparse, datetime, json, os, re, subprocess, sys, tomllib

FOLDER_FIGURES = os.path.join("Resources", "figures", "folder-quick-checks.json")
FOLDER_FIGURES_SCHEMA = "folder-figures-1"

OPERATION_KINDS = ["audio", "embeddings", "perception", "restoration", "geometry"]
KIND_METRICS = {
    "decisions": ["agreementPercent", "millisecondsPerDecision", "batchedDecisionsPerSecond", "peakMemoryGB", "batteryPercentPer1000Decisions"],
    "chat": ["nextTokenAgreementPercent", "nextTokenKL", "firstTokenSeconds", "tokensPerSecond", "peakMemoryGB", "loadPeakMemoryGB", "maximumContextTokens",
             "batteryPercentPer100000Tokens"],
    "images": ["psnrDB", "secondsPerImage", "peakMemoryGB", "batteryPercentPer100Images"],
    "voices": ["spectralDistance", "realTimeFactor", "firstAudioSeconds", "peakMemoryGB", "batteryPercentPerHourSpeech"],
    "sound": ["spectralDistance", "realTimeFactor", "peakMemoryGB", "batteryPercentPerHourAudio"],
    "enhancement": ["spectralDistance", "realTimeFactor", "peakMemoryGB", "batteryPercentPerHourAudio"],
    **{k: ["secondsPerRequest", "peakMemoryGB"] for k in OPERATION_KINDS},
}
# What peak.py sent (its fixtures and TEXTS): the hover names the input with the time.
AUDIO_OPS = {"transcribe", "diarize", "end_of_turn", "separate_audio"}
IMAGE_OPS = {"detect", "segment", "estimate_depth", "remove_background", "ground_screen", "upscale_image", "image_to_3d"}
VIDEO_OPS = {"interpolate_video", "upscale_video", "track_video"}


def input_words(op):
    if op in AUDIO_OPS: return "a 3.3 s speech-and-music clip"
    if op in IMAGE_OPS: return "a 1024×1280 photo"
    if op in VIDEO_OPS: return "a 2 s 640×360 video clip"
    if op == "embed": return "three short sentences"
    if op == "rerank": return "a query and four short documents"
    return "a short fixed test input"


def utc(text):
    """'2026-10-22T04:52:20+0200' → '2026-10-22T02:52:20Z'."""
    return datetime.datetime.strptime(text, "%Y-%m-%dT%H:%M:%S%z").astimezone(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def gpu_cores():
    out = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    m = re.search(r"Total Number of Cores:\s*(\d+)", out)
    return int(m.group(1)) if m else None


def has_figure(doc):
    stats = doc.get("stats") or {}
    return any(isinstance(c, dict) and c.get("state") in ("quickCheck", "measured") and c.get("value") is not None for c in (stats.get("metrics") or {}).values())


def front_matter(text):
    """A results/*.md file's +++ TOML header, as JSON-like data."""
    if not text.startswith("+++\n"): return None
    end = text.find("\n+++", 4)
    return tomllib.loads(text[4:end]) if end > 0 else None


def read_result(path):
    try:
        text = open(path, encoding="utf-8").read()
        return json.loads(text) if path.endswith(".json") else front_matter(text) if path.endswith(".md") else None
    except Exception:
        return None


def importer_owned(doc):
    """A table result this script wrote: bound to a quick-check record."""
    source = (((doc or {}).get("stats") or {}).get("availability") or {}).get("provenance", {}).get("source", "") if isinstance(doc, dict) else ""
    return source.startswith("results/quick-checks/")


def existing_figures(results, dtype, runner):
    """Another result file (not this script's) already gives this precision a quick-check or measured figure for the
    runner, in any mode."""
    if not os.path.isdir(results): return None
    for name in sorted(os.listdir(results)):
        path = os.path.join(results, name)
        if not os.path.isfile(path): continue
        doc = read_result(path)
        if not isinstance(doc, dict) or doc.get("schema") != 2 or importer_owned(doc): continue
        stats = doc.get("stats") or {}
        if stats.get("dtype") == dtype and stats.get("runner") == runner and has_figure(doc): return name
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("jsonl")
    ap.add_argument("--repo", default=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    ap.add_argument("--run", default="night-2026-10-22")
    ap.add_argument("--gpu-cores", type=int, default=None)
    ap.add_argument("--local-models", default=None, help="installed model folders to read model.toml from (never written)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    cores = args.gpu_cores or gpu_cores()
    if not cores: sys.exit("GPU core count unknown; pass --gpu-cores")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.-]*", args.run): sys.exit("run must be letters, digits, dots and dashes")

    def completed(row):
        return {op: o for op, o in (row.get("ops") or {}).items() if isinstance(o, dict) and o.get("status") == 200 and isinstance(o.get("seconds_warm"), (int, float))}

    def problem(row):
        if row.get("error"): return "error"
        if not isinstance(row.get("peak_footprint_mb"), (int, float)) or row["peak_footprint_mb"] <= 0: return "no peak"
        if not completed(row): return "no completed request"
        return None

    good, bad = {}, {}
    for line in open(args.jsonl, encoding="utf-8"):
        line = line.strip()
        if not line: continue
        try: row = json.loads(line)
        except json.JSONDecodeError: continue  # a line still being written
        if not (row.get("model") and row.get("dtype")): continue
        key = (row["model"], row["dtype"])
        why = problem(row)
        if why: bad[key] = why
        else: good[key] = row  # the last good row wins; a later failure never discards it

    by_model, skipped = {}, [f"{m} {d}: {why}" for (m, d), why in sorted(bad.items()) if (m, d) not in good]
    for (model, dtype), row in sorted(good.items()):
        by_model.setdefault(model, []).append((dtype, row, completed(row)))

    written, untouched = [], []
    store_path = os.path.join(args.repo, FOLDER_FIGURES)
    store = read_result(store_path) if os.path.isfile(store_path) else None
    store_entries = list(store.get("entries") or []) if isinstance(store, dict) and store.get("schema") == FOLDER_FIGURES_SCHEMA else []
    store_changed = False
    # Where an entry without a repo folder cites its measurements: a path that exists, relative to the repo when the file
    # is inside it, else from the home folder ("~/…"), never naming the user.
    source = os.path.realpath(args.jsonl)
    repo_root, home = os.path.realpath(args.repo), os.path.realpath(os.path.expanduser("~"))
    measured_at = (os.path.relpath(source, repo_root) if source.startswith(repo_root + os.sep)
                   else "~" + source[len(home):] if source.startswith(home + os.sep) else source)
    for model, rows in sorted(by_model.items()):
        # A shipped folder, else a kit folder; a model the repo has no folder for only by --local-models (read only).
        folder = next((os.path.join(args.repo, tree, model) for tree in ("models", "planned-models")
                       if os.path.isfile(os.path.join(args.repo, tree, model, "model.toml"))), None)
        installed = os.path.join(args.local_models, model) if args.local_models else None
        source_folder = folder or (installed if installed and os.path.isfile(os.path.join(installed, "model.toml")) else None)
        if not source_folder: skipped.append(f"{model}: no model folder"); continue
        manifest = tomllib.load(open(os.path.join(source_folder, "model.toml"), "rb"))
        kind = manifest["kind"]
        folder_kind = kind not in KIND_METRICS
        # A kit kind is the folder's own (kind_file) or, for a kit model in the signed catalog, Resources/kinds/<kind>.toml.
        kind_path = (os.path.join(source_folder, manifest["kind_file"]) if manifest.get("kind_file")
                     else os.path.join(args.repo, "Resources", "kinds", kind + ".toml"))
        if folder_kind and not os.path.isfile(kind_path): skipped.append(f"{model}: kind {kind} has no table figures"); continue
        if not folder_kind and not folder: skipped.append(f"{model}: no model folder"); continue
        revision = manifest["weights"]["revision"]
        runner = (manifest.get("selection") or {}).get("default") or manifest["runner"][0]["name"]
        spec = next((r for r in manifest.get("runner", []) if r.get("name") == runner), {})
        kernels = os.path.join(source_folder, "kernels", "manifest.toml")
        fast_kernels = os.path.isfile(kernels) and 'mode = "fast"' in open(kernels, encoding="utf-8").read()
        mode = "fast" if spec.get("fast_exact") or fast_kernels else "exact"
        declared = manifest.get("op") or []
        primary = None
        if folder_kind:
            # A kit kind declares its operations in the folder's own kind file.
            kind_spec = tomllib.load(open(kind_path, "rb"))
            declared = kind_spec.get("op") or []
            primary = kind_spec.get("primary_operation")
        titles = {o["name"]: o.get("title", o["name"]) for o in declared}
        order = ([primary] if primary else []) + [o["name"] for o in declared if o.get("default")] + [o["name"] for o in declared if not o.get("default")]
        results = os.path.join(folder, "results") if folder else None
        evidence_path = os.path.join("results", "quick-checks", args.run + ".json")
        evidence_rows = []
        for dtype, row, ok_ops in rows:
            if folder_kind:
                chip = row.get("chip") or "m5-max"
                date = utc(row["date"])
                op = next((o for o in order if o in ok_ops), sorted(ok_ops)[0])
                title = titles.get(op, op.replace("_", " ").capitalize())
                busy = row.get("machine") or "Mac in use"
                cited = (os.path.relpath(os.path.join(folder, evidence_path), args.repo) if folder else measured_at) + f"; {dtype}"
                common = dict(chip=chip, gpuCores=cores, date=date, source=cited, revision=revision, run=f"{args.run}-{dtype}")
                what = f"one ordinary Load, then two {title} requests (warm-up, timed) through Arsenal's API"
                caveat = f"{busy}. Quick check, not the controlled interleaved (ABBA) protocol."
                entry = {"model": model, "kind": kind, "revision": revision, "precision": dtype, "runner": runner,
                         "peakMemoryGB": {"state": "quickCheck", "value": row["peak_footprint_mb"] * 1_048_576 / 1e9,
                                          "basis": f"Load and two {title} requests on {input_words(op)}.", "provenance": dict(common, method=(
                             f"Sum of macOS phys_footprint_peak over all Arsenal worker processes (only the measured model was loaded) across {what}; "
                             f"MiB converted to GB (10^9 bytes). {caveat}"))},
                         "secondsPerRequest": {"state": "quickCheck", "value": ok_ops[op]["seconds_warm"], "basis": f"One warm {title} request on {input_words(op)}.",
                                               "provenance": dict(common, method=f"Wall time of the timed (second) {title} request, one sample, after {what}; includes the API round trip. {caveat}")}}
                store_entries = [e for e in store_entries if (e.get("model"), e.get("precision"), e.get("runner")) != (model, dtype, runner)] + [entry]
                store_changed = True
                if folder: evidence_rows.append(dict({k: v for k, v in row.items() if k not in ("model", "dtype")}, precision=dtype, date=date))
                written.append(f"{model} {dtype}: {FOLDER_FIGURES}" + (f" (record {os.path.relpath(os.path.join(folder, evidence_path), args.repo)})" if folder else ""))
                continue
            chip = row.get("chip") or "m5-max"
            name = f"quick-{chip}-{dtype}-{runner}-{mode}.json"
            blocker = existing_figures(results, dtype, runner)
            if blocker: untouched.append(f"{model} {dtype}: kept {blocker}"); continue
            target_file = os.path.join(results, name)
            if os.path.exists(target_file) and not importer_owned(read_result(target_file)):
                untouched.append(f"{model} {dtype}: kept {name} (not written by this script)"); continue
            date = utc(row["date"])
            op = next((o for o in order if o in ok_ops), sorted(ok_ops)[0])
            title = titles.get(op, op.replace("_", " ").capitalize())
            busy = row.get("machine") or "Mac in use"
            common = dict(chip=chip, gpuCores=cores, date=date, source=f"{evidence_path}; {dtype}", revision=revision, run=f"{args.run}-{dtype}")
            what = f"one ordinary Load, then two {title} requests (warm-up, timed) through Arsenal's API"
            caveat = f"{busy}. Quick check, not the controlled interleaved (ABBA) protocol."
            peak_gb = row["peak_footprint_mb"] * 1_048_576 / 1e9
            metrics = {m: {"state": "notMeasured"} for m in KIND_METRICS[kind]}
            metrics["peakMemoryGB"] = {"state": "quickCheck", "value": peak_gb, "basis": f"Load and two {title} requests on {input_words(op)}.",
                                       "provenance": dict(common, method=(
                f"Sum of macOS phys_footprint_peak over all Arsenal worker processes (only the measured model was loaded) across {what}; "
                f"MiB converted to GB (10^9 bytes). {caveat}"))}
            speed = "secondsPerRequest" if kind in OPERATION_KINDS else "secondsPerImage" if kind == "images" and op != "generate" else None
            if speed:
                metrics[speed] = {"state": "quickCheck", "value": ok_ops[op]["seconds_warm"], "basis": f"One warm {title} request on {input_words(op)}.",
                                  "provenance": dict(common, method=f"Wall time of the timed (second) {title} request, one sample, after {what}; includes the API round trip. {caveat}")}
            doc = {"schema": 2, "model": model, "mode": mode, "stats": {
                "kind": kind, "dtype": dtype, "runner": runner,
                "availability": {"state": "quickCheck", "provenance": dict(common, method=f"Loaded and answered {title} at {dtype}: {what}. {caveat}")},
                "metrics": metrics}}
            evidence_rows.append(dict({k: v for k, v in row.items() if k not in ("model", "dtype")}, precision=dtype, date=date))
            if not args.dry_run:
                os.makedirs(results, exist_ok=True)
                with open(os.path.join(results, name), "w", encoding="utf-8") as f: json.dump(doc, f, indent=2, sort_keys=True, ensure_ascii=False); f.write("\n")
            written.append(f"{model} {dtype}: {os.path.join('results', name)}")
        if evidence_rows and not args.dry_run:
            target = os.path.join(folder, evidence_path)
            os.makedirs(os.path.dirname(target), exist_ok=True)
            # A rerun under the same run merges into the record: rows of precisions this pass did not write stay, because
            # their table results still point at this file. A record for another revision is replaced whole.
            prior = read_result(target) if os.path.isfile(target) else None
            if isinstance(prior, dict) and prior.get("schema") == "quick-check-1" and prior.get("model") == model and prior.get("revision") == revision:
                fresh = {r["precision"] for r in evidence_rows}
                evidence_rows += [r for r in prior.get("rows") or [] if isinstance(r, dict) and r.get("precision") not in fresh]
            evidence_rows.sort(key=lambda r: r["precision"])
            evidence = {"schema": "quick-check-1", "model": model, "run": args.run, "revision": revision, "chip": evidence_rows[0].get("chip") or "m5-max",
                        "gpu_cores": cores, "source": "lab/" + os.path.basename(os.path.dirname(os.path.abspath(args.jsonl))) + "/" + os.path.basename(args.jsonl),
                        "method": "lab peak.py: per precision, one ordinary Load and two requests of each one-shot operation through the installed "
                                  "Arsenal's API; peak_footprint_mb is the sum of macOS phys_footprint_peak (MiB) over all Arsenal worker processes (only the measured model was loaded); seconds_warm is the second "
                                  "request's wall time. Busy Mac; quick check.",
                        "rows": evidence_rows}
            with open(target, "w", encoding="utf-8") as f: json.dump(evidence, f, indent=2, sort_keys=True, ensure_ascii=False); f.write("\n")
    if store_changed and not args.dry_run:
        os.makedirs(os.path.dirname(store_path), exist_ok=True)
        store_entries.sort(key=lambda e: (e["model"], e["precision"], e["runner"]))
        with open(store_path, "w", encoding="utf-8") as f:
            json.dump({"schema": FOLDER_FIGURES_SCHEMA, "entries": store_entries}, f, indent=2, sort_keys=True, ensure_ascii=False); f.write("\n")
    print(json.dumps({"written": written, "kept_existing": untouched, "skipped": skipped}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
