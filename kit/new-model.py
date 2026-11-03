#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Add a Hugging Face model to the installed Arsenal when no bundled runner serves it.

    kit/new-model.py plan OWNER/REPO                       read-only: licence, files, size, dtype, params
    kit/new-model.py scaffold OWNER/REPO --id ID --kind KIND --op OP [--input image|audio|text|numbers]
    kit/new-model.py try ID OP '{"param": ...}'            build if needed, run on the real weights outside Arsenal, measure
    kit/new-model.py build ID                              compile only (try does it for you)
    kit/new-model.py check ID|FOLDER [--load] [--json]     every static check, no model run (install does it for you)
    kit/new-model.py install ID                            check, drop the folder into Arsenal, refresh, add to the table
    kit/new-model.py clean ID                              delete this tool's working folder (after install)

Weights are downloaded once, into the Hugging Face cache (~/.cache/huggingface/hub); Arsenal's Get reuses that copy.

The procedure is docs/ADD-A-MODEL.md. Standard library only; no network except Hugging Face and the local Arsenal.
"""
import argparse, collections, datetime, fcntl, hashlib, json, os, platform, re, select, shutil, struct, subprocess, sys, threading, time, urllib.request

KIT = os.path.dirname(os.path.abspath(__file__))
# The kit also ships inside the signed app (Contents/Resources/kit): never write a __pycache__ beside it, or the app's
# seal breaks and the updater refuses every later version. Homebrew's python3 would write one there.
sys.dont_write_bytecode = True
sys.path.insert(0, KIT)
import folder_check  # noqa: E402  (static checks of a model folder)
WORK = os.environ.get("ARSENAL_KIT_WORK") or os.path.expanduser("~/Library/Caches/Arsenal-new-models")
BUILD = os.environ.get("ARSENAL_KIT_BUILD") or os.path.join(WORK, "build")
SUPPORT = os.environ.get("ARSENAL_SUPPORT_DIR") or os.path.expanduser("~/Library/Application Support/Arsenal")
# The one place Arsenal's Get looks for local copies (ArsenalCore LocalModelSources): a file here is cloned, not downloaded.
HF_HUB = os.path.expanduser("~/.cache/huggingface/hub")
MLX_PIN = "901941965d82e4a216d4d117231d847d194c563d"
# ARSENAL_APP points the kit at another copy of the app, e.g. one running a lab profile.
APPS = [os.environ["ARSENAL_APP"]] if os.environ.get("ARSENAL_APP") else [os.path.expanduser("~/Applications/Arsenal.app"), "/Applications/Arsenal.app"]
OPEN_LICENCES = {"apache-2.0", "mit", "bsd-2-clause", "bsd-3-clause", "isc", "cc0-1.0", "cc-by-4.0", "mpl-2.0", "unlicense", "zlib"}
SPDX = {"apache-2.0": "Apache-2.0", "mit": "MIT", "bsd-2-clause": "BSD-2-Clause", "bsd-3-clause": "BSD-3-Clause", "isc": "ISC",
        "cc0-1.0": "CC0-1.0", "cc-by-4.0": "CC-BY-4.0", "mpl-2.0": "MPL-2.0", "unlicense": "Unlicense", "zlib": "Zlib"}
DTYPES = {"F32": "fp32", "F16": "fp16", "BF16": "bf16"}
SMALL = (".json", ".txt", ".model", ".tiktoken")  # tokenizer, config and processor files
TODO = "TODO"


def die(message):
    print("STOP: " + message)
    sys.exit(1)


def signed_ids():
    """Model ids in the installed Arsenal's signed catalog: a local folder named like one is ignored."""
    try:
        with open(app() + "/Contents/Resources/catalog/catalog.json") as f:
            return {entry["manifest"]["id"] for entry in json.load(f)["models"]}
    except (OSError, ValueError, KeyError):
        return set()


def app():
    for path in APPS:
        if os.path.exists(path + "/Contents/MacOS/Arsenal"):
            return path
    die("Arsenal.app is not installed in ~/Applications or /Applications.")


def locked(name):
    """An exclusive lock held until this command exits: one kit build, and one model run on the GPU, at a time."""
    os.makedirs(WORK, exist_ok=True)
    path = os.path.join(WORK, name + ".lock")
    handle = open(path, "a+")
    try:
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"waiting for {path} (another build or model run)", flush=True)
        fcntl.flock(handle, fcntl.LOCK_EX)
    return handle


def folder(model_id):
    return os.path.join(WORK, "models", model_id)


def snapshot_dir(repo, revision):
    return os.path.join(HF_HUB, "models--" + repo.replace("/", "--"), "snapshots", revision)


def inside_hub(path):
    """True when path, symlinks resolved, stays inside the Hugging Face cache. A snapshot linked into another app's folder
    (another app once linked its own Drama 3 conversion into the Fish Audio snapshot) is not Arsenal's to write."""
    hub = os.path.realpath(HF_HUB)
    return os.path.realpath(path).startswith(hub + os.sep)


def writable_snapshot_dir(repo, revision):
    wdir = snapshot_dir(repo, revision)
    # Checked before makedirs: realpath also resolves the missing tail, so a linked repo folder or a dangling link stops here.
    if not inside_hub(wdir):
        die(f"{wdir} resolves to {os.path.realpath(wdir)}, outside the Hugging Face cache: another app owns that folder. "
            "Remove the link (never the files it points to) and scaffold again.")
    os.makedirs(wdir, exist_ok=True)
    return wdir


def weights_dir(model_id):
    toml = open(os.path.join(folder(model_id), "model.toml")).read()
    repo = re.search(r'^repo = "([^"]+)"', toml, re.M).group(1)
    revision = re.search(r'^revision = "([0-9a-f]{40})"', toml, re.M).group(1)
    wdir = snapshot_dir(repo, revision)
    if not inside_hub(wdir):
        die(f"{wdir} resolves to {os.path.realpath(wdir)}, outside the Hugging Face cache: those are another app's files, not this model's pins.")
    return wdir


CONVERTED_SUFFIXES = (".pth", ".pt", ".npy", ".npz")


def load_dir(model_id, process, wait):
    """The folder `try` loads from: the Hugging Face snapshot, or, when the pins include PyTorch or NumPy files that only
    Get's conversion turns into safetensors, the runner's own `convert` output (as Get makes it). ARSENAL_KIT_CONVERTED
    names an existing converted copy to reuse; otherwise it is made once under the kit's work folder."""
    files = folder_check.read_toml(os.path.join(folder(model_id), "model.toml")).get("weights", {}).get("files", [])
    if not any(name.endswith(CONVERTED_SUFFIXES) for name in files):
        return weights_dir(model_id)
    supplied = os.environ.get("ARSENAL_KIT_CONVERTED")
    target = supplied or os.path.join(WORK, "converted", model_id)
    if not os.path.exists(os.path.join(target, ".converted")):
        # A run that stopped half way leaves files the converter would refuse to overwrite: start from an empty folder.
        # The kit's own folder is cleared; a folder the user named is never deleted, only refused when not empty.
        if supplied:
            if os.path.isdir(target) and os.listdir(target):
                die(f"ARSENAL_KIT_CONVERTED={target} holds files but no .converted marker; empty it or name another folder.")
        else:
            shutil.rmtree(target, ignore_errors=True)
        source = weights_dir(model_id)
        staging = os.path.join(WORK, "converted", model_id + ".source")
        shutil.rmtree(staging, ignore_errors=True)
        for name in files:  # the pinned files only, as Get stages them (APFS clones of the cache)
            os.makedirs(os.path.dirname(os.path.join(staging, name)), exist_ok=True)
            subprocess.run(["cp", "-c", "-L", os.path.join(source, name), os.path.join(staging, name)], check=True)
        os.makedirs(target, exist_ok=True)
        print(f"converting the pinned files as Get does, into {target}", flush=True)
        send(process, {"id": 99, "op": "convert", "src_dir": staging, "dst_dir": target, "dtype": "fp16", "group_size": 64,
                       "recipe": "cast:arsenal-kit-half-v1", "assets": {}})
        reply = wait(99)
        shutil.rmtree(staging, ignore_errors=True)
        if not reply["ok"]:
            die(f"convert failed: {reply['error']['message']}")
        open(os.path.join(target, ".converted"), "w").write(json.dumps(reply["files"]))
    return target


def hf_token():
    """The user's Hugging Face read token for gated repositories: HF_TOKEN, else the file `hf auth login` writes. Never printed."""
    if os.environ.get("HF_TOKEN"):
        return os.environ["HF_TOKEN"].strip()
    path = os.path.join(os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "token")
    return open(path).read().strip() if os.path.exists(path) else None


def hf_open(url, headers=None):
    """GET from Hugging Face with the user's token; a gated or private repository stops with what to do, not a traceback."""
    request = urllib.request.Request(url, headers=headers or {})
    token = hf_token()
    if token:
        request.add_header("Authorization", "Bearer " + token)
    try:
        return urllib.request.urlopen(request, timeout=60)
    except urllib.error.HTTPError as error:
        if error.code in (401, 403):
            path = url.split("huggingface.co/")[-1].split("?")[0]
            repo = "/".join((path[len("api/models/"):] if path.startswith("api/models/") else path).split("/")[:2])
            die(f"{repo} is gated or private on Hugging Face (HTTP {error.code}). The user must open https://huggingface.co/{repo}, "
                "accept its terms with their own account, and give a read token (HF_TOKEN, or `hf auth login`)"
                + (": the current token has no access to it." if token else "; none is set."))
        raise


def http_json(url, data=None, token=None, method=None):
    request = urllib.request.Request(url, data=data, method=method)
    if token:
        request.add_header("Authorization", "Bearer " + token)
    if data is not None:
        request.add_header("Content-Type", "application/json")
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.loads(response.read() or b"null")


def repo_id(text):
    """owner/repo from what a user pastes: owner/repo, or a Hugging Face link to the repo or any page inside it."""
    match = re.fullmatch(r"(?:https?://)?(?:www\.)?(?:huggingface\.co|hf\.co)/(?!api/|datasets/|spaces/)([\w.-]+/[\w.-]+)(?:/.*)?/?", text.strip())
    repo = match.group(1) if match else text.strip().strip("/")
    if not re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        die(f"'{text}' is not a Hugging Face model: give owner/repo or its https://huggingface.co/owner/repo link.")
    return repo


def accepted(licence, args):
    """An open licence, or the exact non-open one the user accepted for their own Mac (--accept-licence). Never shipped by Arsenal."""
    if licence in OPEN_LICENCES:
        return True
    given = (getattr(args, "accept_licence", None) or "").strip().lower()
    if given and given == (licence or "none"):
        return True
    if given:
        die(f"--accept-licence {given} does not match this repository's licence '{licence or 'none'}'. Read the licence, ask the user again.")
    return False


def hf_info(repo):
    try:
        return json.loads(hf_open(f"https://huggingface.co/api/models/{repo}?blobs=true").read())
    except Exception as error:
        die(f"Hugging Face does not answer for {repo}: {error}")


def safetensors_header(repo, revision, name):
    url = f"https://huggingface.co/{repo}/resolve/{revision}/{name}"
    size = struct.unpack("<Q", hf_open(url, {"Range": "bytes=0-7"}).read())[0]
    return json.loads(hf_open(url, {"Range": f"bytes=8-{7 + size}"}).read())


CORE = ("config.json", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "preprocessor_config.json")


def inventory(info):
    """Pinned files: the base safetensors (those its index lists, when there is one) and the config/tokenizer files."""
    weights, small, other = [], [], []
    siblings = info.get("siblings", [])
    index = next((s for s in siblings if s["rfilename"] == "model.safetensors.index.json"), None)
    listed = None
    if index:
        try:
            url = f"https://huggingface.co/{info['id']}/resolve/{info['sha']}/model.safetensors.index.json"
            listed = set(json.loads(hf_open(url).read())["weight_map"].values())
        except Exception:
            listed = None
    for sibling in siblings:
        name, size = sibling["rfilename"], sibling.get("size") or 0
        base = os.path.basename(name).lower()
        if name.endswith(".safetensors") and "/" not in name and not any(word in base for word in ("adapter", "lora", "peft")):
            (weights if listed is None or name in listed else other).append(sibling)
        elif name.endswith(".safetensors.index.json") or name in CORE:
            small.append(sibling)
        elif base.endswith(SMALL) and size < 20_000_000 and not base.startswith(("readme", "license", "notice")) \
                and not name.startswith(("onnx/", "openvino/", ".", "assets/")):
            small.append(sibling)
        else:
            other.append(sibling)
    if listed is None and len(weights) > 1:
        # Without an index, two root checkpoints are variants of one model (model.safetensors and model.fp16.safetensors):
        # keep the plain one so its keys are not loaded twice.
        plain = [w for w in weights if w["rfilename"] == "model.safetensors"]
        if plain:
            other += [w for w in weights if w not in plain]
            weights = plain
    return weights, small, other


def licence_of(info):
    card = info.get("cardData") or {}
    value = card.get("license") or next((t[8:] for t in info.get("tags", []) if t.startswith("license:")), "")
    if isinstance(value, list):
        value = value[0] if len(value) == 1 else ""
    return str(value).lower()


def inspect_weights(repo, revision, weights):
    """Parameter count and on-disk float type from the safetensors headers; refuses quantized checkpoints."""
    params, counts = 0, {}
    for sibling in weights:
        for key, tensor in safetensors_header(repo, revision, sibling["rfilename"]).items():
            if key == "__metadata__":
                continue
            count = 1
            for dim in tensor["shape"]:
                count *= dim
            params += count
            counts[tensor["dtype"]] = counts.get(tensor["dtype"], 0) + count
    floats = sum(v for k, v in counts.items() if k in DTYPES)
    if floats < 0.5 * max(params, 1):
        die(f"{repo} stores most weights as {', '.join(sorted(k for k in counts if k not in DTYPES))}: a quantized release. "
            "Find the maker's 16-bit or FP32 safetensors release of the same model and plan that repo instead.")
    kinds = {DTYPES[k] for k in counts if k in DTYPES}
    return params, "fp32" if "fp32" in kinds else ("bf16" if "bf16" in kinds else "fp16")


def plan(args):
    args.repo = repo_id(args.repo)
    info = hf_info(args.repo)
    revision = info["sha"]
    licence = licence_of(info)
    extra = ["--accept-licence", licence] if licence not in OPEN_LICENCES and accepted(licence, args) else []
    bundled = subprocess.run([app() + "/Contents/MacOS/Arsenal", "catalog", "add-local", args.repo, "--plan", *extra], capture_output=True, text=True)
    if bundled.returncode == 0:
        print(bundled.stdout.strip())
        print("\nA bundled runner serves this model: follow Path A in docs/ADD-A-MODEL.md (catalog add-local). Do not use the kit.")
        return
    if bundled.returncode != 3:  # 3 is the only "no bundled runner" refusal; the others are about this repo or this Mac
        print((bundled.stdout + bundled.stderr).strip())
        die("Arsenal refused this model for the reason above (not a missing runner). Tell the user; do not write a runner for it.")
    if not accepted(licence, args):
        die(f"licence '{licence or 'none'}' is not on Arsenal's open list ({', '.join(sorted(OPEN_LICENCES))}), so Arsenal will never ship this model. "
            f"The user may still add it to their own Mac on its own terms: show them the licence (https://huggingface.co/{args.repo}), "
            f"and only if they accept it, rerun with --accept-licence {licence or 'none'}.")
    weights, small, other = inventory(info)
    if not weights:
        names = ", ".join(s["rfilename"] for s in other[:8])
        die(f"{args.repo} has no .safetensors weights (files: {names}). Search Hugging Face for a safetensors release of the same model "
            "(often '<org>/<name>-hf' or a transformers port), check it is the same weights, and plan that repo instead.")
    params, source = inspect_weights(args.repo, revision, weights)
    download = sum(s.get("size") or 0 for s in weights + small)
    config = next((s for s in small if s["rfilename"] == "config.json"), None)
    print(f"repo       {args.repo} at {revision}")
    print(f"licence    {SPDX[licence]} (open)" if licence in OPEN_LICENCES else
          f"licence    {licence}: NOT open; accepted by the user for this Mac only. Arsenal never ships or shares it.")
    print(f"task       {info.get('pipeline_tag') or 'not tagged'}; library {info.get('library_name') or 'none'}")
    print(f"params     {params:,} ({label(params)}), counted from the safetensors headers")
    print(f"weights    {', '.join(s['rfilename'] for s in weights)}: {source.upper()} on disk; Arsenal serves FP16" +
          (" (cast once at Get)" if source != "fp16" else ""))
    print(f"also pins  {', '.join(s['rfilename'] for s in small) or 'nothing'}{'' if config else '  (no config.json: read sizes from the maker code)'}")
    print(f"download   {download / 1e6:.0f} MB, once, into the Hugging Face cache; Arsenal's Get reuses it")
    reason = (bundled.stdout + bundled.stderr).strip().split(" Nothing was written.")[0]
    print(f"runner     none bundled ({reason}); you write one with the kit (Path B in docs/ADD-A-MODEL.md)")
    print("\nTell the user in one message: the repo, licence, size, that you will write a small MLX runner for it, and that Arsenal "
          "will ask them once to approve its code. Wait for their yes. Then: kit/new-model.py scaffold " + args.repo + " --id ID --kind KIND --op OP")


def label(params):
    return f"{params / 1e9:.1f}B" if params >= 1e9 else f"{params / 1e6:.1f}M"


def toml_string(value):
    return json.dumps(value, ensure_ascii=False)


def toml_list(values):
    return "[" + ", ".join(toml_string(v) for v in values) + "]"


INPUTS = {
    "image": ({"image": {"type": "object", "description": "The picture.", "x-arsenal-file": {"accept": ["image/png", "image/jpeg", "image/heic"]},
                         "x-arsenal-control": "image-picker", "required": ["path"], "additionalProperties": False,
                         "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 4096}}}}, ["image"], "image"),
    "audio": ({"audio": {"type": "object", "description": "The recording.", "x-arsenal-file": {"accept": ["audio/wav", "audio/mpeg", "audio/x-m4a", "audio/flac"]},
                         "required": ["path"], "additionalProperties": False,
                         "properties": {"path": {"type": "string", "minLength": 1, "maxLength": 4096}}}}, ["audio"], "audio"),
    "text": ({"text": {"type": "string", "description": "The text.", "minLength": 1, "maxLength": 100000}}, ["text"], "text"),
    "numbers": ({"values": {"type": "array", "description": "The numbers, oldest first.", "minItems": 1, "maxItems": 100000,
                            "items": {"type": "number"}}}, ["values"], "text"),
}


def scaffold(args):
    model_id = args.id
    if not re.fullmatch(r"[a-z0-9][a-z0-9.-]*", model_id):
        die("--id must be lowercase letters, digits, dots and dashes, e.g. vitpose-base-simple.")
    target = folder(model_id)
    if os.path.exists(target):
        die(f"{target} exists. Edit it, or delete it to start over.")
    if os.path.exists(os.path.join(SUPPORT, "local-models", model_id)):
        die(f"Arsenal already has a local model {model_id}.")
    if model_id in signed_ids():
        die(f"{model_id} is a model in Arsenal's catalog; a local folder with that id would be ignored. Pick another --id.")
    local = args.kind.startswith("local.")
    resources = app() + "/Contents/Resources/kinds"
    if local:
        if not re.fullmatch(r"local\.[a-z0-9][a-z0-9.-]*", args.kind):
            die("a local kind is local.<name>, lowercase, e.g. local.pose.")
        if not args.input:
            die("a local kind needs --input image|audio|text|numbers (what the op takes).")
        shared = os.path.join(SUPPORT, "local-models")
        for other_model in (os.listdir(shared) if os.path.isdir(shared) else []):
            if os.path.exists(os.path.join(shared, other_model, "kinds", args.kind + ".toml")):
                die(f"{args.kind} is already used by the installed model {other_model}; folders sharing a local kind must be identical. "
                    "Pick another name, e.g. " + args.kind + "-2.")
    else:
        source_ops = os.path.join(resources, args.kind, "ops")
        if not os.path.exists(os.path.join(source_ops, f"{args.op}.input.json")):
            kinds = sorted(k[:-5] for k in os.listdir(resources) if k.endswith(".toml"))
            ops = sorted({f.split(".")[0] for f in os.listdir(source_ops)}) if os.path.isdir(source_ops) else []
            die(f"kind {args.kind} has no op {args.op}. Kinds: {', '.join(kinds)}. Its ops: {', '.join(ops) or 'none'}. "
                "If none fits, use --kind local.<name> --input ...")
    if not re.fullmatch(r"[a-z][a-z0-9_]*", args.op):
        die("--op is lowercase letters, digits and underscores, e.g. estimate_pose.")
    args.repo = repo_id(args.repo)
    info = hf_info(args.repo)
    revision = info["sha"]
    licence = licence_of(info)
    if not accepted(licence, args):
        die(f"licence '{licence or 'none'}' is not open. Only with the user's acceptance of its terms: rerun with --accept-licence {licence or 'none'}.")
    weights, small, _ = inventory(info)
    if not weights:
        die("no .safetensors weights; run plan for the reason.")
    params, source = inspect_weights(args.repo, revision, weights)

    # One download, into the Hugging Face cache, where Arsenal's Get finds and clones it (the user said yes at plan).
    # A file already there with the right hash is reused, so nothing is downloaded twice.
    wdir = writable_snapshot_dir(args.repo, revision)
    pins = {}
    for sibling in weights + small:
        name = sibling["rfilename"]
        path = os.path.join(wdir, name)
        if not inside_hub(os.path.dirname(path)):
            die(f"{name} would land outside the Hugging Face cache ({os.path.realpath(os.path.dirname(path))}).")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        lfs = (sibling.get("lfs") or {}).get("sha256")
        if os.path.exists(path) and (not lfs or sha256(path) == lfs):
            print(f"reusing {name} from the Hugging Face cache")
        else:
            print(f"downloading {name} ({(sibling.get('size') or 0) / 1e6:.1f} MB)", flush=True)
            part = os.path.join(HF_HUB, "models--" + args.repo.replace("/", "--"), ".kit-" + sha256_text(name) + ".part")
            if os.path.lexists(path):
                os.remove(path)
            try:
                with hf_open(f"https://huggingface.co/{args.repo}/resolve/{revision}/{name}") as response, open(part, "wb") as out:
                    shutil.copyfileobj(response, out, 8 << 20)
            except BaseException:
                if os.path.exists(part):
                    os.remove(part)
                raise
            if lfs and sha256(part) != lfs:
                os.remove(part)
                die(f"{name} does not match its Hugging Face hash; scaffold again.")
            os.rename(part, path)
        pins[name] = (sha256(path), os.path.getsize(path))

    os.makedirs(target)
    for sub in ("ops", "log", "gate", "bench", "runners/swift"):
        os.makedirs(os.path.join(target, sub), exist_ok=True)
    shutil.copy(os.path.join(KIT, "runner", "Package.swift"), os.path.join(target, "runners/swift/Package.swift"))
    shutil.copytree(os.path.join(KIT, "runner", "Sources"), os.path.join(target, "runners/swift/Sources"))
    model_swift = os.path.join(target, "runners/swift/Sources/Runner/Model.swift")
    text = open(model_swift).read()
    text = text.replace('let modelKind = "local.example"', f"let modelKind = {toml_string(args.kind)}")
    text = text.replace('let modelOps = ["predict"]', f"let modelOps = [{toml_string(args.op)}]")
    open(model_swift, "w").write(text)

    modalities_in, modalities_out = [TODO], [TODO]
    if local:
        properties, required, modality = INPUTS[args.input]
        modalities_in, modalities_out = [modality], [TODO]
        kind_dir = os.path.join(target, "kinds", args.kind, "ops")
        os.makedirs(kind_dir)
        schema = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": False,
                  "required": required, "properties": properties}
        json.dump(schema, open(os.path.join(kind_dir, f"{args.op}.input.json"), "w"), indent=2)
        output = {"$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": True,
                  "description": TODO + ": list the fields run() returns under properties, with types", "properties": {}}
        json.dump(output, open(os.path.join(kind_dir, f"{args.op}.output.json"), "w"), indent=2)
        title = args.op.replace("_", " ").capitalize()
        open(os.path.join(target, "kinds", args.kind + ".toml"), "w").write(
            f'id = {toml_string(args.kind)}\ntitle = {toml_string(TODO + ": plural kind title, e.g. Poses")}\n'
            f'title_singular = {toml_string(TODO + ": e.g. Pose model")}\nicon = "sparkles"\norder = 20\n'
            f'primary_operation = {toml_string(args.op)}\n\n[[op]]\nname = {toml_string(args.op)}\ntitle = {toml_string(title)}\n'
            f'input = "ops/{args.op}.input.json"\noutput = "json"\noutput_schema = "ops/{args.op}.output.json"\n'
            'stream = false\ndefault = true\ntimeout_s = 300\nfirst_response_timeout_s = 300\n')
    else:
        # The kind decides whether the op answers with JSON or with files (depth maps, audio): copy its `output`.
        kind_toml = open(os.path.join(resources, args.kind + ".toml")).read()
        block = kind_toml.split(f'name = "{args.op}"', 1)[1].split("[[op]]", 1)[0]
        found = re.search(r'^output = "([a-z]+)"', block, re.M)
        kind_output = found.group(1) if found else "json"
        for suffix in ("input", "output"):
            shutil.copy(os.path.join(source_ops, f"{args.op}.{suffix}.json"), os.path.join(target, "ops", f"{args.op}.{suffix}.json"))

    owner = args.repo.split("/")[0]
    created = (info.get("createdAt") or "2026")[:4]
    lines = [
        "schema = 3", f"id = {toml_string(model_id)}",
        f"name = {toml_string(TODO + ': full official name with family, size and version, e.g. ViTPose Base Simple')}",
        f"kind = {toml_string(args.kind)}",
    ]
    if local:
        lines.append(f'kind_file = "kinds/{args.kind}.toml"')
    lines += [
        f"summary = {toml_string(TODO + ': one sentence, what it does for the user')}",
        f"description = {toml_string(TODO + ': 2-4 sentences from the model card: what it is, who made it, what it was trained on, its benchmark or speed claim')}",
        f"purpose = {toml_string(TODO + ': a few words, under 40 characters, e.g. Find body keypoints')}",
        f"publisher = {toml_string(owner)}", f"released = {created}", f"params = {toml_string(label(params))}",
        f"tasks = {toml_list([info.get('pipeline_tag') or TODO])}", f"languages = {toml_list([TODO + ': ISO codes the model really reads or writes, e.g. en; mul if it has no language'])}",
        f"good_for = {toml_list([TODO + ': what it is best at'])}",
        f"limits = {toml_list([TODO + ': where it fails; say whether it handles Polish when it reads or writes language', 'Parity with the maker code not run'])}",
        "suggested = false", "",
        "[links]", f"homepage = {toml_string('https://huggingface.co/' + args.repo)}", "",
        "[licence]", f"spdx = {toml_string(SPDX.get(licence, 'LicenseRef-' + (licence or 'none')))}",
        f"name = {toml_string(SPDX.get(licence, licence or 'none') + ('' if licence in OPEN_LICENCES else ', accepted by this Mac' + chr(39) + 's owner'))}",
        f"commercial = {'true' if licence in OPEN_LICENCES else 'false'}",
        f"url = {toml_string(f'https://huggingface.co/{args.repo}/blob/{revision}/README.md')}", "",
        "[facets]", f"modalities_in = {toml_list(modalities_in)}", f"modalities_out = {toml_list(modalities_out)}",
        "# modalities: each of text, audio, image, video", "",
        "[weights]", f"repo = {toml_string(args.repo)}", f"revision = {toml_string(revision)}", f"dtype = {toml_string(source)}",
        f"files = {toml_list(list(pins))}", "", "[weights.sha256]",
    ]
    lines += [f"{toml_string(name)} = {toml_string(sha)}" for name, (sha, _) in pins.items()]
    lines += ["", "[weights.bytes]"] + [f"{toml_string(name)} = {size}" for name, (_, size) in pins.items()]
    if source != "fp16":  # FP32 and BF16 checkpoints are cast to FP16 once, at Get, by the kit runner's convert
        lines += ["", "[derive]", 'fp16 = "cast:arsenal-kit-half-v1"']
    lines += ["", "[selection]", 'default = "kit"', "", "[[runner]]", 'name = "kit"', 'language = "swift"', 'entry = "runners/swift/bin/runner"',
              "protocol = 1", 'chips = ["*"]', 'min_macos = "26.0"', "fast_exact = false", 'features = ["sequentialOnly"]']
    if not local:
        lines += ["", "[[op]]", f"name = {toml_string(args.op)}", f"title = {toml_string(args.op.replace('_', ' ').capitalize())}",
                  f'input = "ops/{args.op}.input.json"', f"output = {toml_string(kind_output)}", f'output_schema = "ops/{args.op}.output.json"',
                  "stream = false", "timeout_s = 300", "first_response_timeout_s = 300"]
    lines += ["", "[gate]", 'recipe = "gate/gate.toml"', "", "[bench]", 'recipe = "bench/bench.toml"', ""]
    open(os.path.join(target, "model.toml"), "w").write("\n".join(lines))
    open(os.path.join(target, "gate/gate.toml"), "w").write(
        f'schema = 1\nid = "{model_id}-kit-v1"\nreference_runner = "pinned-maker-implementation"\nprecision = "fp16"\n'
        'independent_vendor_qualified = false\npassing_requires = ["Parity with the maker code on fixed inputs (not run)"]\n')
    open(os.path.join(target, "bench/bench.toml"), "w").write(
        f'schema = 1\nsuite = "{model_id}-kit-v1"\nversion = 1\nprecision = "fp16"\nrunner = "kit"\nstatus = "deferred-by-user"\n'
        'metrics = ["seconds_per_request", "peak_ram_mb"]\n')
    open(os.path.join(target, "README.md"), "w").write(
        f"# {TODO}: full official name\n\n{TODO}: what it does, its inputs and outputs, one example request, and its limits.\n\n"
        f"Source: [{args.repo}](https://huggingface.co/{args.repo}/tree/{revision}), {SPDX.get(licence, licence + ' (not open; added by this Mac' + chr(39) + 's owner on its terms)')}. Runs as FP16 through the Arsenal kit runner (MLX).\n")
    reference = gather_reference(args.repo, revision, info, wdir, weights, model_id)
    print(f"Folder:    {target}")
    print(f"Weights:   {wdir} (Hugging Face cache; Arsenal's Get clones these, no second download)")
    print(f"Reference: {reference} (model card, the maker's Python source as text, tensors.txt; read them, never run them)")
    print("\nNext:")
    print(f"  1. Write {model_swift}: port the maker's code in {reference}; module key names must match tensors.txt.")
    print(f"  2. kit/new-model.py try {model_id} {args.op} '{{...}}'   (builds first; run it with the card's own example)")
    print(f"  3. Replace every {TODO} in {target} (model.toml, README.md, ops/ or kinds/) from README.md in {reference}")
    print(f"  4. kit/new-model.py install {model_id}")


def gather_reference(repo, revision, info, wdir, weights, model_id):
    """The card, the maker's model code and the checkpoint's tensor list in one folder outside the model folder."""
    reference = os.path.join(WORK, "reference", model_id)
    os.makedirs(reference, exist_ok=True)
    lines = []
    for sibling in weights:
        with open(os.path.join(wdir, sibling["rfilename"]), "rb") as handle:
            size = struct.unpack("<Q", handle.read(8))[0]
            header = json.loads(handle.read(size))
        lines += [f"{key} {tensor['shape']} {tensor['dtype']}" for key, tensor in header.items() if key != "__metadata__"]
    open(os.path.join(reference, "tensors.txt"), "w").write("\n".join(sorted(lines)) + "\n")

    def fetch(url, name):
        try:
            data = hf_open(url).read()
        except Exception:
            return False
        open(os.path.join(reference, name), "wb").write(data)
        return True

    fetch(f"https://huggingface.co/{repo}/raw/{revision}/README.md", "README.md")
    for sibling in info.get("siblings", []):  # the repository's own code (custom architectures), as text
        name = sibling["rfilename"]
        if name.endswith(".py") and (sibling.get("size") or 0) < 2_000_000:
            fetch(f"https://huggingface.co/{repo}/raw/{revision}/{name}", "repo-" + name.replace("/", "-"))
    model_type = ((info.get("config") or {}).get("model_type") or "").replace("-", "_")
    if model_type:  # a transformers architecture: its modeling and preprocessing files
        base = f"https://raw.githubusercontent.com/huggingface/transformers/main/src/transformers/models/{model_type}/"
        for part in ("modeling", "configuration", "image_processing", "feature_extraction", "processing"):
            fetch(base + f"{part}_{model_type}.py", f"transformers-{part}_{model_type}.py")
    write_outline(reference)
    write_tensor_tree(reference, lines)
    card = open(os.path.join(reference, "README.md"), errors="ignore").read() if os.path.exists(os.path.join(reference, "README.md")) else ""
    links = sorted(set(re.findall(r"https://github\.com/[\w.-]+/[\w.-]+", card)))
    if links:
        open(os.path.join(reference, "github-links.txt"), "w").write("\n".join(links) + "\n")
    return reference


def write_outline(reference):
    """outline.txt: every class and function of the maker's Python files with its line number (parsed as text, never run)."""
    import ast
    out = []
    for name in sorted(os.listdir(reference)):
        if not name.endswith(".py"):
            continue
        out.append(f"== {name}")
        try:
            tree = ast.parse(open(os.path.join(reference, name), errors="ignore").read())
        except SyntaxError:
            out.append("   (could not parse)")
            continue
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                bases = ", ".join(ast.unparse(b) for b in node.bases)
                out.append(f"{node.lineno:6}  class {node.name}({bases})")
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        out.append(f"{item.lineno:6}      def {item.name}({', '.join(a.arg for a in item.args.args)})")
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                out.append(f"{node.lineno:6}  def {node.name}({', '.join(a.arg for a in node.args.args)})")
    open(os.path.join(reference, "outline.txt"), "w").write("\n".join(out) + "\n")


def write_tensor_tree(reference, lines):
    """tensors-tree.txt: tensor names with every layer number folded into its range ({0-11}), so a 12-layer model reads as one."""
    groups = collections.OrderedDict()
    for line in lines:
        name, rest = line.split(" ", 1)
        parts = name.split(".")
        pattern = tuple("{n}" if part.isdigit() else part for part in parts)
        numbers = [int(part) for part in parts if part.isdigit()]
        groups.setdefault((pattern, rest), []).append(numbers)
    out = []
    for (pattern, rest), seen in sorted(groups.items()):
        position, folded = 0, []
        for part in pattern:
            if part == "{n}":
                values = [numbers[position] for numbers in seen]
                low, high = min(values), max(values)
                folded.append(str(low) if low == high else f"{{{low}-{high}}}")
                position += 1
            else:
                folded.append(part)
        out.append(".".join(folded) + " " + rest)
    open(os.path.join(reference, "tensors-tree.txt"), "w").write("\n".join(out) + "\n")


def build(args):
    target = folder(args.id)
    package = os.path.join(target, "runners/swift")
    if not os.path.exists(os.path.join(package, "Package.swift")):
        die(f"no runner at {package}; scaffold first.")
    if open(os.path.join(package, "Package.swift")).read() != open(os.path.join(KIT, "runner/Package.swift")).read():
        die("runners/swift/Package.swift must stay the kit's; put all code in Sources/Runner/Model.swift (more .swift files there are fine).")
    app_pin = app_mlx_pin()
    if not app_pin:
        print("note: this Arsenal.app does not record its mlx-swift revision; the GPU self-test is the only compatibility check.")
    if app_pin and app_pin != MLX_PIN:
        die(f"this Arsenal.app was built with mlx-swift {app_pin[:12]}, the kit pins {MLX_PIN[:12]}: its MLX shaders would not match the runner. "
            "Use the kit from the Arsenal release that matches the installed app.")
    build_lock = locked("build")
    shared = BUILD
    pkg = os.path.join(shared, "pkg")
    os.makedirs(pkg, exist_ok=True)
    shutil.copy(os.path.join(package, "Package.swift"), os.path.join(pkg, "Package.swift"))
    shutil.rmtree(os.path.join(pkg, "Sources"), ignore_errors=True)
    shutil.copytree(os.path.join(package, "Sources"), os.path.join(pkg, "Sources"))
    clt = "/Library/Developer/CommandLineTools"
    weak = os.path.join(shared, "weak-borrow.o")
    if not os.path.exists(weak):
        source = os.path.join(shared, "weak-borrow.c")
        open(source, "w").write("extern void swift_initBorrow(void) __attribute__((weak_import));\n"
                                "__attribute__((used)) void *const arsenal_weak_swift_initBorrow = (void *)&swift_initBorrow;\n")
        subprocess.run([clt + "/usr/bin/clang", "-target", "arm64-apple-macos26.0", "-c", source, "-o", weak], check=True)
    first = not os.path.exists(os.path.join(shared, "scratch/release/runner"))
    print("building" + (" (first build compiles MLX: about 10 minutes, once for every model)" if first else ""), flush=True)
    started = time.time()
    result = subprocess.run(
        ["nice", "-n", "19", clt + "/usr/bin/swift", "build", "--package-path", pkg, "--scratch-path", os.path.join(shared, "scratch"),
         "--build-system", "native", "-c", "release", "-j", "2", "-Xlinker", weak, "-Xlinker", "-weak_reference_mismatches", "-Xlinker", "weak"],
        capture_output=True, text=True, env=dict(os.environ, DEVELOPER_DIR=clt))
    if result.returncode != 0:
        errors = [line.replace(pkg + "/Sources", package + "/Sources") for line in (result.stdout + result.stderr).splitlines()
                  if ": error:" in line or ": warning: unused" in line]
        print("\n".join(dict.fromkeys(errors).keys()) or (result.stdout + result.stderr)[-3000:])
        die("the runner does not compile; fix the errors above in Model.swift and build again.")
    bin_dir = os.path.join(package, "bin")
    shutil.rmtree(bin_dir, ignore_errors=True)
    os.makedirs(bin_dir)
    shutil.copy(os.path.join(shared, "scratch/release/runner"), os.path.join(bin_dir, "runner"))
    # The app's MLX shaders, compiled from the same MLX pin; the runner finds them beside itself.
    shutil.copytree(app() + "/Contents/Helpers/mlx-swift_Cmlx.bundle", os.path.join(bin_dir, "mlx-swift_Cmlx.bundle"))
    shutil.copy(os.path.join(pkg, "Package.resolved"), os.path.join(package, "Package.resolved"))
    subprocess.run(["codesign", "--force", "--sign", "-", "--timestamp=none", os.path.join(bin_dir, "runner")], check=True, capture_output=True)
    try:
        gpu_selftest(os.path.join(bin_dir, "runner"))
    except SystemExit:
        shutil.rmtree(bin_dir, ignore_errors=True)  # never leave a runner that failed its self-test for try to reuse
        raise
    receipt(target)
    print(f"built {bin_dir}/runner in {time.time() - started:.0f} s; GPU self-test passed. Next: kit/new-model.py try {args.id} OP '{{...}}'")


def app_mlx_pin():
    """The mlx-swift revision the installed app was built with, as its third-party notices record it (None if not recorded)."""
    notices = app() + "/Contents/Resources/THIRD_PARTY_NOTICES.txt"
    if not os.path.exists(notices):
        return None
    match = re.search(r"\*\*mlx-swift\*\* revision `([0-9a-f]{40})`", open(notices, errors="ignore").read())
    return match.group(1) if match else None


def gpu_selftest(runner):
    """The copied shaders and this runner on the GPU against the CPU, on small fixed tensors (about a second)."""
    process = subprocess.Popen([runner], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    try:
        replies = frames(process, timeout=60)
        next(replies, None)
        send(process, {"id": 1, "op": "selftest"})
        reply = next(replies, None)
        while reply is not None and reply.get("id") != 1:
            reply = next(replies, None)
    finally:
        process.kill()
    if not reply or not reply.get("ok") or reply.get("max_relative_difference", 1) > 1e-2:
        detail = reply.get("error", {}).get("message") if reply and not reply.get("ok") else json.dumps(reply)
        die(f"the GPU self-test failed ({detail}): Arsenal.app's MLX shaders do not work with this runner. Use the kit matching the installed Arsenal.")


def sha256_text(text):
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def receipt(target):
    """Arsenal admits a local Swift runner only with a receipt hashing every file under runners/swift/."""
    package = os.path.join(target, "runners/swift")
    hashes = {}
    for root, _, files in os.walk(package):
        for name in files:
            path = os.path.join(root, name)
            relative = os.path.relpath(path, target)
            if relative != "runners/swift/build-receipt.json":
                hashes[relative] = sha256(path)
    swift = subprocess.run(["/Library/Developer/CommandLineTools/usr/bin/swift", "--version"], capture_output=True, text=True)
    json.dump({"command": "kit/new-model.py build (swift build -c release --build-system native, Arsenal kit Package.swift)",
               "toolchain": (swift.stdout + swift.stderr).strip().splitlines()[-1] + " (CommandLineTools); MLX shaders copied from Arsenal.app",
               "sha256": dict(sorted(hashes.items()))}, open(os.path.join(package, "build-receipt.json"), "w"), indent=2)


REPLY_SECONDS = 300  # the scaffold's first_response_timeout_s


def frames(process, timeout=REPLY_SECONDS):
    descriptor = process.stdout.fileno()

    def read_exact(count):
        data = b""
        while len(data) < count:
            ready, _, _ = select.select([descriptor], [], [], timeout)
            if not ready:
                process.kill()
                die(f"the runner sent nothing for {timeout} s (a hang, or an endless loop in Model.swift).")
            part = os.read(descriptor, count - len(data))
            if not part:
                return None
            data += part
        return data

    while True:
        header = read_exact(4)
        if header is None:
            return
        length = struct.unpack(">I", header)[0]
        if length > 16 * 1024 * 1024:
            process.kill()
            die(f"the runner sent a {length}-byte frame; Arsenal refuses frames over 16 MiB and stops the worker.")
        body = read_exact(length)
        if body is None or body[0] != 1:
            return
        yield json.loads(body[1:])


def send(process, message):
    body = b"\x01" + json.dumps(message).encode()
    process.stdin.write(struct.pack(">I", len(body)) + body)
    process.stdin.flush()


def schema_errors(value, schema, path="result"):
    """The parts of Arsenal's JSON Schema check a reply most often fails: type, required, unknown fields, list items."""
    kind = schema.get("type")
    types = {"object": dict, "array": list, "string": str, "boolean": bool, "integer": int, "number": (int, float)}
    if isinstance(kind, str) and kind in types:
        if not isinstance(value, types[kind]) or (kind in ("integer", "number") and isinstance(value, bool)):
            return [f"{path} must be {kind}, got {type(value).__name__}"]
    errors = []
    if isinstance(value, dict):
        properties = schema.get("properties", {})
        errors += [f"{path}.{key} is required" for key in schema.get("required", []) if key not in value]
        if schema.get("additionalProperties") is False:
            errors += [f"{path}.{key} is not in the schema" for key in value if key not in properties]
        for key, item in value.items():
            if key in properties:
                errors += schema_errors(item, properties[key], f"{path}.{key}")
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        for index, item in enumerate(value[:50]):
            errors += schema_errors(item, schema["items"], f"{path}[{index}]")
    return errors


def op_schema(target, op, suffix):
    kind = re.search(r'^kind = "([^"]+)"', open(os.path.join(target, "model.toml")).read(), re.M).group(1)
    for path in (os.path.join(target, "ops", f"{op}.{suffix}.json"), os.path.join(target, "kinds", kind, "ops", f"{op}.{suffix}.json")):
        if os.path.exists(path):
            return json.load(open(path))
    return {}


def short(value):
    if isinstance(value, list):
        if len(value) > 8:
            return [short(v) for v in value[:8]] + [f"... {len(value)} items"]
        return [short(v) for v in value]
    if isinstance(value, dict):
        return {k: short(v) for k, v in value.items()}
    return value


def needs_build(target):
    runner = os.path.join(target, "runners/swift/bin/runner")
    if not os.path.exists(runner):
        return True
    built = os.path.getmtime(runner)
    sources = os.path.join(target, "runners/swift/Sources")
    return any(os.path.getmtime(os.path.join(root, name)) > built for root, _, names in os.walk(sources) for name in names)


def try_model(args):
    target = folder(args.id)
    if not os.path.isdir(target):
        die(f"no folder for {args.id}; scaffold first.")
    if needs_build(target):
        build(args)
    runner = os.path.join(target, "runners/swift/bin/runner")
    requests = []
    for index, text in enumerate(args.params):
        try:
            params = json.loads(text)
        except json.JSONDecodeError as error:
            die(f"request {index + 1} is not JSON ({error}). Files are objects: '{{\"image\": {{\"path\": \"/abs/file.jpg\"}}}}'.")
        problems = schema_errors(params, op_schema(target, args.op, "input"), "params")
        if problems:
            die(f"request {index + 1} does not match the op's input schema: " + "; ".join(problems[:6]))
        requests.append(params)
    gpu_lock = locked("gpu")
    scratch = os.path.join(WORK, "try-support")
    os.makedirs(os.path.join(scratch, "jobs"), exist_ok=True)
    process = subprocess.Popen([runner], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=dict(os.environ, ARSENAL_SUPPORT_DIR=scratch, ARSENAL_CACHE_DIR=scratch))
    replies = frames(process)
    stderr = collections.deque(maxlen=200)
    threading.Thread(target=lambda: [stderr.append(line.decode(errors="replace")) for line in process.stderr], daemon=True).start()
    peaks = [0.0]

    def peak():
        return peaks[0]

    def wait(request_id):
        for frame in replies:
            if frame.get("event") == "status":
                peaks[0] = max(peaks[0], frame.get("memory", {}).get("footprint_mb", 0))
            if frame.get("id") == request_id and "ok" in frame:
                return frame
        time.sleep(0.2)
        die("the runner exited without replying. Its stderr:\n" + "".join(stderr)[-3000:])

    hello = next(replies, None)
    if not hello or hello.get("event") != "hello":
        time.sleep(0.2)
        die("the runner did not say hello: " + "".join(stderr)[-2000:])
    kind = re.search(r'^kind = "([^"]+)"', open(os.path.join(target, "model.toml")).read(), re.M).group(1)
    if kind not in hello.get("kinds", []) or hello.get("runner") != "kit":
        die(f"the runner's hello must list kind {kind} and runner kit; set modelKind = \"{kind}\" in Model.swift and build again.")
    load_seconds, timings, results = run_requests(process, wait, args, target, requests)
    process.kill()
    failed = False
    for index, result in enumerate(results):
        result = {k: v for k, v in result.items() if k != "id"}
        print(f"--- request {index + 1}")
        print(json.dumps(short(result), indent=1)[:4000])
        if not result.get("ok"):
            if index == 0 or result.get("error", {}).get("code") != "invalid_request":
                failed = True  # an awkward input may be refused cleanly, never crash or fail in the engine
            continue
        problems = schema_errors({k: v for k, v in result.items() if k != "ok"}, op_schema(target, args.op, "output"))
        if problems:
            die(f"request {index + 1}: the reply does not match the op's output schema, so Arsenal would refuse it: " + "; ".join(problems[:6]))
    if failed:
        die("a request failed (the first request must answer; others must answer or give invalid_request): fix Model.swift and try again.")
    print(f"\nload {load_seconds:.2f} s, first request {timings[0]:.2f} s, warm request {timings[-1]:.2f} s, peak memory {peak():.0f} MB")
    record(args.id, args.op, load_seconds, timings, peak())
    print("Saved to results/ and log/. Check that the output makes sense against the model card (its example inputs and outputs).")


def run_requests(process, wait, args, target, requests):
    """One load; the first request twice (first and warm timing), then each further request once."""
    def out_dir(request_id):
        """A fresh job folder per request, where a runner writes files (depth maps, audio), as Arsenal's supervisor gives it."""
        path = os.path.join(WORK, "try-support", "jobs", f"try-{os.getpid()}-{request_id}")
        os.makedirs(path, exist_ok=True)
        return path

    try:
        started = time.time()
        send(process, {"id": 1, "op": "load", "model_dir": target, "weights_dir": load_dir(args.id, process, wait), "dtype": "fp16", "fast": False,
                       "assets": {}, "config": {"settings": {}, "kernels": []}})
        loaded = wait(1)
        load_seconds = time.time() - started
        if not loaded["ok"]:
            die(f"load failed: {loaded['error']['message']}")
        timings, results = [], []
        for index in range(2):
            started = time.time()
            send(process, dict(requests[0], id=2 + index, op=args.op, out_dir=out_dir(2 + index)))
            result = wait(2 + index)
            timings.append(time.time() - started)
            if not result["ok"]:
                break
        results.append(result)
        for index, params in enumerate(requests[1:]):
            send(process, dict(params, id=10 + index, op=args.op, out_dir=out_dir(10 + index)))
            results.append(wait(10 + index))
        send(process, {"id": 9, "op": "unload"})
    except BaseException:
        process.kill()
        raise
    return load_seconds, timings, results


def record(model_id, op, load_seconds, timings, peak):
    target = folder(model_id)
    toml = open(os.path.join(target, "model.toml")).read()
    revision = re.search(r'revision = "([0-9a-f]{40})"', toml).group(1)
    chip = re.sub(r"[^a-z0-9]+", "-", subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True)
                  .stdout.strip().lower().replace("apple ", "")).strip("-") or "mac"
    cores = subprocess.run(["system_profiler", "SPDisplaysDataType"], capture_output=True, text=True).stdout
    gpu_cores = int(re.search(r"Total Number of Cores: (\d+)", cores).group(1)) if "Total Number of Cores" in cores else 0
    swap = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout.strip()
    now = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    run = "kit-" + now[:10]
    build = subprocess.run(["defaults", "read", app() + "/Contents/Info.plist", "CFBundleVersion"], capture_output=True, text=True).stdout.strip()
    peak_mib = round(peak * 1e6 / 1048576, 1)  # the runner reports MB; quick-check rows are MiB
    os.makedirs(os.path.join(target, "results/quick-checks"), exist_ok=True)
    runner_sha = sha256(os.path.join(target, "runners/swift/bin/runner"))
    ops = {op: {"seconds_first": round(timings[0], 3), "seconds_warm": round(timings[-1], 3), "status": 200}}
    previous = os.path.join(target, f"results/quick-checks/{run}.json")
    if os.path.exists(previous):  # another op of the same model and runner, tried earlier today: keep its figures
        row = (json.load(open(previous)).get("rows") or [{}])[0]
        if row.get("runner_sha256") == runner_sha:
            ops = dict(row.get("ops") or {}, **ops)
            peak_mib = max(peak_mib, row.get("peak_footprint_mb") or 0)
    json.dump({
        "schema": "quick-check-1", "model": model_id, "run": run, "revision": revision, "chip": chip, "gpu_cores": gpu_cores,
        "method": f"kit/new-model.py try: one Load and two requests of {op} through the kit runner outside the app, FP16; "
                  f"peak_footprint_mb is the runner's largest phys_footprint after a reply, in MiB; Arsenal build {build}.",
        "rows": [{"precision": "fp16", "date": now, "peak_footprint_mb": peak_mib, "load_seconds": round(load_seconds, 3),
                  "swap": swap, "runner_sha256": runner_sha, "ops": ops}],
    }, open(os.path.join(target, f"results/quick-checks/{run}.json"), "w"), indent=2, sort_keys=True)
    kind = re.search(r'^kind = "([^"]+)"', toml, re.M).group(1)
    if kind in ("audio", "embeddings", "perception", "restoration", "geometry"):
        # The Models table's figures: one warm request and peak memory, bound to the quick-check record above.
        def provenance(method):
            return {"chip": chip, "date": now, "gpuCores": gpu_cores, "method": method, "revision": revision, "run": run + "-fp16",
                    "source": f"results/quick-checks/{run}.json; fp16"}
        method = f"kit/new-model.py try: one Load, then two {op} requests (warm-up, timed) on the kit runner outside the app. Quick check."
        json.dump({"mode": "exact", "model": model_id, "schema": 2, "stats": {
            "availability": {"provenance": provenance(method), "state": "quickCheck"}, "dtype": "fp16", "kind": kind, "runner": "kit",
            "metrics": {
                "peakMemoryGB": {"basis": f"Load and two {op} requests.", "provenance": provenance("Runner phys_footprint after each reply, MB to GB. " + method),
                                 "state": "quickCheck", "value": peak_mib * 1048576 / 1e9},
                "secondsPerRequest": {"basis": f"One warm {op} request.", "provenance": provenance("Wall time of the timed request. " + method),
                                      "state": "quickCheck", "value": round(timings[-1], 3)}}}},
            open(os.path.join(target, f"results/quick-{chip}-fp16-kit-exact.json"), "w"), indent=2, sort_keys=True)
    repo = re.search(r'repo = "([^"]+)"', toml).group(1)
    open(os.path.join(target, "log/001-add.md"), "w").write(
        "+++\n"
        f'hypothesis = "The Arsenal kit runner serves {repo} at FP16."\n'
        'change = "Kit runner (kit/runner) with a Model.swift port of the maker code."\n'
        f'measurement = "kit/new-model.py try on {chip}, {now}: one Load and two {op} requests on the pinned weights, outside the app."\n'
        'verdict = "kept"\n'
        f"numbers = {{ load_s = {load_seconds:.3f}, first_s = {timings[0]:.3f}, warm_s = {timings[-1]:.3f}, peak_footprint_mb = {peak:.0f} }}\n"
        f'links = ["results/quick-checks/{run}.json"]\n'
        "+++\n"
        f"# 001 Added with the kit\n\nMeasured {now}, Arsenal build {build}, {chip}: load {load_seconds:.2f} s, {op} {timings[0]:.2f} s first and "
        f"{timings[-1]:.2f} s warm, peak {peak:.0f} MB, swap {swap}.\nNot checked: parity with the maker's code, benchmark quality, int8/int4.\n")


def check(args, quiet=False):
    """Every static check, then Arsenal's own validation. Nothing runs the model; --load also loads its weights into the runner."""
    target = os.path.abspath(args.id) if os.sep in args.id else folder(args.id)
    if not os.path.isdir(target):
        die(f"no folder {target}: give a model id from scaffold, or a folder path.")
    report, facts = folder_check.run(target, kinds_dir=app() + "/Contents/Resources/kinds", hub=HF_HUB)
    notes = [p for p in report.problems if p["code"] == "S6-local-only"]
    report.problems = [p for p in report.problems if p["code"] != "S6-local-only"]
    if getattr(args, "json", False):
        print(json.dumps({"folder": target, "ok": not report.problems, "problems": report.problems, "notes": notes, "weights": facts}, indent=1))
        sys.exit(0 if not report.problems else 1)
    problems = report.lines()
    if getattr(args, "load", False) and not problems:
        problems += load_check(target, facts)
    name = os.path.basename(target)
    if needs_build(target) and os.path.isdir(os.path.join(target, "runners/swift/Sources")) and os.path.exists(os.path.join(target, "runners/swift/bin/runner")):
        problems.append(f"S4-stale runners/swift/Sources: changed after the runner was built. Fix: kit/new-model.py try {name} OP '{{...}}' again.")
    records = [os.path.join(target, "results/quick-checks", f) for f in os.listdir(os.path.join(target, "results/quick-checks"))] \
        if os.path.isdir(os.path.join(target, "results/quick-checks")) else []
    runner = os.path.join(target, "runners/swift/bin/runner")
    if records and os.path.exists(runner) and not any(row.get("runner_sha256") == sha256(runner) for record in records for row in json.load(open(record))["rows"]):
        problems.append(f"S7-stale results/quick-checks: the runner changed since the last try. Fix: kit/new-model.py try {name} OP '{{...}}' again.")
    validate = subprocess.run([app() + "/Contents/MacOS/Arsenal", "catalog", "validate", target], capture_output=True, text=True)
    if validate.returncode != 0:
        problems.append("S9-arsenal catalog validate: " + (validate.stdout + validate.stderr).strip()[-2000:])
    for note in notes:
        print(f"note: {note['problem']}")
    if problems:
        print("\n".join("- " + p for p in problems))
        die(f"{len(problems)} problem(s) above; fix each, then check again.")
    if not quiet:
        print(f"ok: {os.path.basename(target)} passes every static check" + (" and its weights load into the runner" if getattr(args, "load", False) else "")
              + f". Next: kit/new-model.py install {os.path.basename(target)}")


def load_check(target, facts):
    """Start the runner and load the pinned weights into its modules: every key and shape must match. No forward pass runs."""
    runner = os.path.join(target, "runners/swift/bin/runner")
    if not facts.get("cached") or not os.path.exists(runner):
        return ["S5-load: the weights are not all in the Hugging Face cache or the runner is not built, so the load check could not run. "
                "Fix: scaffold downloads them; try builds the runner."]
    scratch = os.path.join(WORK, "check-support")
    os.makedirs(os.path.join(scratch, "jobs"), exist_ok=True)
    process = subprocess.Popen([runner], stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               env=dict(os.environ, ARSENAL_SUPPORT_DIR=scratch, ARSENAL_CACHE_DIR=scratch))
    try:
        replies = frames(process)
        next(replies, None)
        send(process, {"id": 1, "op": "load", "model_dir": target, "weights_dir": facts["snapshot"], "dtype": "fp16", "fast": False,
                       "assets": {}, "config": {"settings": {}, "kernels": []}})
        reply = next((f for f in replies if f.get("id") == 1 and "ok" in f), None)
    finally:
        process.kill()
    if not reply:
        return ["S5-load: the runner exited while loading. Fix: run kit/new-model.py try to see its error."]
    if not reply.get("ok"):
        return [f"S5-load: {reply['error']['message']} Fix: name modules after the checkpoint's keys (reference/tensors-tree.txt), "
                "transpose conv weights, or pass rename: to Weights.apply."]
    return []


def install(args):
    check(args, quiet=True)
    target = folder(args.id)
    local = os.path.join(SUPPORT, "local-models")
    os.makedirs(local, exist_ok=True)
    staging, final = os.path.join(local, ".staging-" + args.id), os.path.join(local, args.id)
    shutil.rmtree(staging, ignore_errors=True)
    shutil.copytree(target, staging, symlinks=False)
    if os.path.exists(final):
        shutil.rmtree(final)
    os.rename(staging, final)
    status = json.load(open(os.path.join(SUPPORT, "status.json")))
    token = open(os.path.join(SUPPORT, "mcp-token")).read().strip()
    base = f"http://127.0.0.1:{status['port']}/arsenal/v1"
    http_json(base + "/refresh", data=b"{}", token=token, method="POST")  # the same as the Refresh arrow in the Models table
    row = next((m for m in http_json(base + "/catalog", token=token) if m.get("id") == args.id), None)
    if not row:
        die("Arsenal did not list the model after Refresh. Run arsenal_find(query:\"" + args.id + "\") to read the problem, fix it, install again.")
    http_json(base + f"/models/{args.id}/add", data=b"{}", token=token, method="POST")
    digest = row.get("contentDigest")
    if not digest:
        die("Arsenal listed the model without a content digest; run arsenal_model(model:\"" + args.id + "\") and read its problem.")
    print(f"Installed {args.id} ({row.get('name')}) and added it to Arsenal's Models table.")
    print(f"Content digest: {digest}")
    print("\nAsk the user, in one message: may Arsenal run this model's code (the kit runner plus your Model.swift, digest " + digest[:12] + ")?")
    print("On their yes, call these Arsenal MCP tools in order:")
    print(f'  arsenal_settings(action:"approve_code", model:"{args.id}", content_digest:"{digest}")')
    print(f'  arsenal_download(model:"{args.id}", allow_download:true)     (the user agreed to the download at plan; arsenal_wait if it returns a job)')
    print(f'  arsenal_run(model:"{args.id}", ...)                           (one real request; it loads the model; check the answer against the model card)')
    print(f"Then: kit/new-model.py clean {args.id}")
    toml = open(os.path.join(target, "model.toml")).read()
    source = re.search(r'^dtype = "(fp32|bf16)"', toml, re.M)
    if source:
        print(f"Tell the user: the original {source.group(1).upper()} files stay in the Hugging Face cache "
              f"({weights_dir(args.id)}) after Arsenal casts its FP16 copy; offer to delete them, only with their yes.")


def clean(args):
    """The kit's own working folders. The weights stay in the Hugging Face cache: a shared store Arsenal cloned from."""
    for path in (folder(args.id), os.path.join(WORK, "reference", args.id)):
        if os.path.exists(path):
            shutil.rmtree(path)
            print("deleted " + path)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    plan_parser = commands.add_parser("plan")
    plan_parser.add_argument("repo", help="owner/repo or its Hugging Face link")
    scaffold_parser = commands.add_parser("scaffold")
    scaffold_parser.add_argument("repo", help="owner/repo or its Hugging Face link")
    for each in (plan_parser, scaffold_parser):
        each.add_argument("--accept-licence", metavar="LICENCE", help="add a model that is not openly licensed, on the user's own acceptance of its terms; local only")
    scaffold_parser.add_argument("--id", required=True)
    scaffold_parser.add_argument("--kind", required=True, help="an Arsenal kind (embeddings, perception, audio, ...) or local.<name>")
    scaffold_parser.add_argument("--op", required=True, help="the operation name, e.g. embed, detect, classify_voice")
    scaffold_parser.add_argument("--input", choices=sorted(INPUTS), help="for a local kind: what the op takes")
    for name in ("build", "install", "clean"):
        commands.add_parser(name).add_argument("id")
    check_parser = commands.add_parser("check", help="every static check of a folder (no model run); --load also loads its weights")
    check_parser.add_argument("id", help="a model id from scaffold, or the path of any model folder")
    check_parser.add_argument("--load", action="store_true", help="also load the pinned weights into the runner (no forward pass)")
    check_parser.add_argument("--json", action="store_true", help="the problems as JSON, for agents and tools")
    try_parser = commands.add_parser("try")
    try_parser.add_argument("id")
    try_parser.add_argument("op")
    try_parser.add_argument("params", nargs="*", default=["{}"], help="one or more JSON requests, run in one load")
    args = parser.parse_args()
    {"plan": plan, "scaffold": scaffold, "build": build, "try": try_model, "check": check, "install": install, "clean": clean}[args.command](args)


if __name__ == "__main__":
    main()
