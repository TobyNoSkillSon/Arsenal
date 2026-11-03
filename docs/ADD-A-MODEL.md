<!-- SPDX-License-Identifier: MIT -->
# Add a model

This is the whole procedure for adding a Hugging Face model to an installed Arsenal, written for an AI agent working for the Mac's owner. Read it top to bottom once, then follow the steps in order. You need nothing else: every rule, command and porting lesson is on this page.

What you will make is a **model folder**: a recipe (pinned weights, the operation's input and output schemas, a description) plus a small **MLX runner in Swift** that you write. Arsenal picks the folder up from `local-models/` with no rebuild or reinstall, checks it, asks the owner once to approve its code, downloads the pinned weights and shows the model in the Models table. Arsenal never adds it to its own catalogue and never shares it.

Two rules have no exceptions:

- **Every added model runs on MLX, in Swift, inside its runner.** Python never serves a model: no Python, PyTorch, Core ML, ONNX or other engine, no starting other programs, no network access from the runner. `check` refuses a folder that breaks this.
- **The owner decides.** Their yes comes before the download, before Arsenal runs the new code, and before adding a model whose licence is not open.

## 0. Before you start

- **Arsenal** is installed and running: `~/Applications/Arsenal.app` or `/Applications/Arsenal.app`. Its support folder is `~/Library/Application Support/Arsenal` (or `$ARSENAL_SUPPORT_DIR` when that is set). Arsenal's MCP tools are named `arsenal_*`.
- **The kit** (`kit/new-model.py`, `kit/runner/`). It ships inside the app: `cd` into `Arsenal.app/Contents/Resources` (in `~/Applications` or `/Applications`) and run the commands below from there; a checkout of the Arsenal repository has the same `kit/` at its root. The kit only reads that folder; its working folders live in `~/Library/Caches/Arsenal-new-models/`.
- **Swift**: Xcode Command Line Tools (`/Library/Developer/CommandLineTools/usr/bin/swift` must exist; if not, ask the owner to run `xcode-select --install`). **Python 3** for the kit (the system `/usr/bin/python3` is enough; the kit uses only the standard library).
- About 1.5 GB free for the kit's build folder (`~/Library/Caches/Arsenal-new-models/`). The first build compiles MLX once, about 4 to 10 minutes; later builds take seconds.
- Commands that may run longer than your shell allows (the first build): run them in the background and read their output when they finish.

## 1. Plan

The owner gives a link or `owner/repo`. Pass either one as it is:

```sh
python3 kit/new-model.py plan https://huggingface.co/owner/repo
```

The plan reads Hugging Face and writes nothing. It prints one of these:

- **`STOP: …`**: read the reason. The common ones:
  - *no `.safetensors` weights*: search Hugging Face for a safetensors release of the same weights (an official `-hf` sibling, a transformers port, or the bot's "Adding safetensors" pull request, which you pin by its commit). Plan that repository only if its card says it holds the same weights; say which one you used. When only the maker's own `torch.save` archive exists (`.pth`, `.pt`) or its data files are NumPy arrays (`.npy`, `.npz`), you may pin those files instead: with `[derive] fp16 = "cast:arsenal-kit-half-v1"` Get turns each one into `<name>.safetensors` through the runner's `convert`, which reads the pickle opcode by opcode and never executes it, and `check` counts its tensors the same way (it refuses an archive that would need code to load). Your `Model.swift` then loads `<name>.safetensors`.
  - *gated or private*: the owner must open the repository page, accept its terms with their own Hugging Face account, and give you a read token (`HF_TOKEN` in the environment, or `hf auth login`). Never ask for their password.
  - *licence not open*: see **Licences** below.
- **"A bundled runner serves this model"**: follow **Path A**.
- **The repository, licence, parameters, weights and download size**: follow **Path B**.

Then tell the owner, in one message: the repository, its licence, the download size, and for Path B that you will write a small MLX runner and that Arsenal will ask them once to approve its code. Wait for their yes. That yes covers this model's download.

### Licences

Arsenal's catalog lists openly licensed models (Apache-2.0, MIT, BSD, ISC, CC0, CC-BY-4.0, MPL-2.0, Unlicense, Zlib) and a few whose terms limit commercial use (non-commercial or a revenue cap), each showing those terms before Get. The owner may still add any other model **to their own Mac, on the terms of its licence**: show them the licence link and its main restrictions (non-commercial, research only, revenue caps, use restrictions) in one or two sentences. Only if they accept, rerun every kit command for this model with `--accept-licence LICENCE`, using the exact licence name the plan printed (for example `--accept-licence other`). The folder is then marked as the owner's own (`spdx = "LicenseRef-…"`, `commercial = false`); Arsenal keeps it on this Mac and never ships, lists or shares it. Never accept a licence on the owner's behalf.

## Path A: a model Arsenal already runs

1. `"$APP/Contents/MacOS/Arsenal" catalog add-local owner/repo --support "$SUPPORT"` writes `local-models/<id>/` (`$APP` and `$SUPPORT` as in step 0; add `--accept-licence …` as above).
2. Fill in the description in `local-models/<id>/model.toml` (step 6 lists the fields).
3. `python3 kit/new-model.py check "$SUPPORT/local-models/<id>"`, fix every line it prints, then continue at step 8.

## Path B: write an MLX runner with the kit

### 2. Choose the kind and the operation

A **kind** groups models that do the same job; an **operation** is one thing a model does, with an input and an output schema. Use an Arsenal kind when its operation means exactly what your model does, so Arsenal's tool windows and other apps can use it:

| Kind | Operations |
|---|---|
| `embeddings` | `embed` (texts or images to vectors), `rerank` |
| `perception` | `detect`, `segment`, `estimate_depth`, `ground_screen`, `track_video`, `segment_session`, `video_session` |
| `audio` | `transcribe`, `diarize`, `separate_audio`, `end_of_turn` |
| `enhancement` | `enhance` (clean up audio) |
| `restoration` | `upscale_image`, `upscale_video`, `interpolate_video`, `colorize`, `inpaint` |
| `voices` | `speak`, `voices`, `design_voice`, `clone_voice` |
| `sound` | `generate_sound`, `generate_music` |
| `images` | `generate` |
| `geometry` | `image_to_3d` |
| `decisions` | `decide`, `judge` |
| `chat` | `translate`, `read_document`, `extract` |

Read the kind's schemas before you choose: `"$APP/Contents/Resources/kinds/<kind>/ops/<op>.input.json` and `.output.json`. Your runner must accept exactly that input and return exactly that output.

When no operation fits (classifying an image or a sound, estimating pose, forecasting numbers), make your own kind: `--kind local.<name> --op <verb> --input image|audio|text|numbers`, for example `--kind local.image-labels --op classify_image --input image`. You then write the output schema yourself (step 6). Pick a kind name no other added model uses.

### 3. Scaffold

```sh
python3 kit/new-model.py scaffold owner/repo --id ID --kind KIND --op OP [--input TYPE]
```

The id is lowercase with dashes, e.g. `vit-base-patch16-224`. This downloads the pinned files once into the Hugging Face cache (`~/.cache/huggingface/hub`, reusing files already there; Arsenal's Get later copies them from there) and writes:

- the model folder `~/Library/Caches/Arsenal-new-models/models/ID/`, with every pin, hash, size and the licence filled in, and `TODO` where you must write;
- for a local kind, its operations in `kinds/local.<name>.toml` with schemas in `kinds/local.<name>/ops/`, not in `model.toml`; for an Arsenal kind, the `[[op]]` in `model.toml` and the kind's schemas copied to `ops/`;
- the reference folder `~/Library/Caches/Arsenal-new-models/reference/ID/`: the model card (`README.md`), the maker's Python source as text (never run it), `outline.txt` (every class and function of that source with its line number), `tensors.txt` (every checkpoint tensor with shape and dtype) and `tensors-tree.txt` (the same with layer numbers folded, `layers.{0-11}.…`).

### 4. Port the model to MLX

Edit only `runners/swift/Sources/Runner/Model.swift` in the model folder; you may add more `.swift` files beside it. Keep `Package.swift`, `Kit.swift`, `Wire.swift` and `main.swift` as the kit wrote them.

**Work in this order:**

1. Read the model card's usage example: which processor, which inputs, what output, what the example's answer is.
2. Read `tensors-tree.txt`: this is the network you must build, module by module.
3. In `outline.txt` find the classes the checkpoint uses and read only those, by line number, in the maker's source (`transformers-modeling_*.py`, `repo-*.py`, or the GitHub project in `github-links.txt`). A `timm` model (library `timm`, `outline.txt` empty) is defined in `timm/models/<architecture>.py` of github.com/huggingface/pytorch-image-models; read that file at the version the card names. Read the preprocessing too (`image_processing_*.py`, `feature_extraction_*.py`, `tokenization_*.py`, or the card's example), and the `config.json` values the code reads.
4. Write one Swift `Module` class per PyTorch module, with properties named after the checkpoint keys, then `init` (load) and `run` (one request).
5. `try` it (step 5) and fix what it reports.

**The template.** `Model.swift` starts like this; keep the shape:

```swift
import Foundation
import MLX
import MLXNN

let modelKind = "embeddings"          // exactly model.toml's kind
let modelOps = ["embed"]              // exactly model.toml's [[op]] names

final class Model {
    let net: Encoder
    let tokenizer: Tokenizer
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)                       // config.json as [String: Any]
        net = Encoder(config)
        try Weights.apply(try Weights.load(weightsDir), to: net)      // throws listing missing/unused keys and bad shapes
        tokenizer = try Text.tokenizer(weightsDir)
    }
    func run(op: String, params: Params) throws -> [String: Any] {
        // read inputs (params.string / numbers / file / double), compute, return the output schema's fields
        throw RunnerError.invalid("unknown op \(op)")
    }
}
```

**Module names follow the checkpoint.** A property `var layers: [Block]` matches `layers.0.…`, `layers.1.…`; a nested module matches a dotted prefix. When a Swift name must differ from the key, use `@ModuleInfo(key: "…") var x: Linear` (for submodules) or `@ParameterInfo(key: "…") var w: MLXArray` (for bare tensors such as `cls_token` or `position_embeddings`). Never give two properties the same key: the second silently takes the tensors and the first stays empty (only the "Unused" list in the load error shows it). When the names still differ, pass `rename:` to `Weights.apply` (`{ key in key.hasPrefix("pooler.") ? nil : key.replacingOccurrences(of: "bert.", with: "") }`; returning nil skips a tensor the model does not use). `Weights.apply` refuses missing keys, unused keys and shape mismatches, and names them: read its message, it tells you exactly what to fix.

**MLX conventions.**

- MLX is channels-last: images are `[N, H, W, C]`, audio sequences `[N, L, C]`.
- Convolution weights must be transposed before `Weights.apply`: PyTorch `Conv2d` `[out, in, kh, kw]` → `w.transposed(0, 2, 3, 1)`; `Conv1d` `[out, in, k]` → `w.transposed(0, 2, 1)`; `ConvTranspose2d` `[in, out, kh, kw]` → `w.transposed(1, 2, 3, 0)`; `Conv3d` `[out, in, t, h, w]` → `w.transposed(0, 2, 3, 4, 1)`. Transpose per named tensor (by key), never by guessing from the shape. `Linear` weights load unchanged.
- A depthwise conv is `groups: channels`; a patch embedding with stride = kernel is a `Conv2d` (or a reshape plus matmul).
- PyTorch `nn.MultiheadAttention` stores `in_proj_weight` as q, k and v stacked: split it into three with `split(w, parts: 3, axis: 0)`.
- `LayerNorm(dimensions:eps:)`, `RMSNorm(dimensions:eps:)`, `BatchNorm(featureCount:eps:)`, `GroupNorm(groupCount:dimensions:pytorchCompatible: true)`: copy each `eps` from config.json or the maker's code (BERT uses 1e-12, ViT 1e-12, most others 1e-5 or 1e-6).
- GELU: `gelu` is the exact (erf) form, `geluApproximate` the tanh form; use the one the config names (`"gelu"` is exact, `"gelu_new"`/`"gelu_pytorch_tanh"` are tanh).

**MLX Swift names.**

- layers: `Linear(inDim, outDim, bias: true)`, `Conv1d`/`Conv2d(inputChannels:outputChannels:kernelSize:stride:padding:dilation:groups:bias:)`, `ConvTransposed2d(…)`, `LayerNorm`, `RMSNorm`, `BatchNorm`, `GroupNorm`, `Embedding(embeddingCount:dimensions:)`, `Upsample(scaleFactor:mode:)`, `Dropout` (inactive at inference; leave it out)
- functions: `gelu`, `geluApproximate`, `relu`, `silu`, `sigmoid`, `tanh`, `softmax(x, axis: -1)`, `matmul`, `sqrt`, `rsqrt`, `exp`, `log`
- array ops: `concatenated([a, b], axis:)`, `stacked`, `split`, `x.reshaped([…])`, `x.transposed(0, 2, 1, 3)`, `x.swappedAxes(1, 2)`, `x.mean(axis:keepDims:)`, `x.sum(axis:)`, `argMax(x, axis:)`, `x.asType(.float32)`, `x[0..., 0]` (slicing), `expandedDimensions(x, axis:)`, `MLX.where(cond, a, b)`, `padded(x, widths:)`
- attention: `MLXFast.scaledDotProductAttention(queries:keys:values:scale:mask:)` with `[B, heads, L, headDim]` inputs; a padding mask is an additive `[B, 1, 1, L]` array (0 where kept, a large negative number where padded) cast to the queries' dtype.
- results: `list(array)` turns an MLXArray into `[Double]` for the reply.

**Kit helpers** (`Kit.swift`):

- inputs: `params.string("text")`, `params.strings("texts")`, `params.numbers("values")`, `params.file("image")` (a URL), `params.double("x", default: 0.5)`, `params.int`, `params.bool`, and `params.raw` for anything else (the request's JSON as `[String: Any]`)
- weights and config: `Weights.load(dir)` (every safetensors tensor as float16; a tensor beyond float16's range stays float32), `Weights.apply`, `Files.json(dir)` (config.json) and `Files.json(dir, "preprocessor_config.json")`
- media: `Images.load(url, width:, height:)` gives `[H, W, 3]` floats in 0…1, then `Images.normalize(pixels, mean:, std:)`; `Images.savePNG`; `Audio.load(url, sampleRate: 16000)` gives mono float samples
- text: `Text.tokenizer(dir)` then `.encode(text:)` gives token ids (with the model's special tokens) and `.decode(tokens:)`
- errors: `throw RunnerError.invalid("…")` for bad input (Arsenal answers `invalid_request`), `RunnerError.failed("…")` for a failure
- long work: `try params.checkCancelled()` between layers, chunks or frames, so Stop in Arsenal works

**Lessons from earlier ports.** Each of these broke a real model; check yours for each one.

*Numbers and precision*

- Weights arrive as float16 and the network computes in float16. Some intermediate values do not fit float16's range of ±65,504 even when every weight does: sums of squares (a scaler's variance, an RMS norm written by hand: squaring anything above 256 overflows), linear-attention normalisers, mel power spectra of loud audio, large position indices. Compute those in float32 (`x.asType(.float32)`), then cast back. A NaN or infinity in the output almost always means one of these.
- Compute softmax over many items, the final scores and any reply numbers in float32.
- One float32 constant promotes the whole network to float32 (slow, and not what the maker ran): build constants such as position tables `.asType(x.dtype)`. In Swift a plain `Float` literal operand is weak and keeps float16, but `MLXArray(Float)` is strong and promotes.
- `/` between integer MLXArrays is true division in Swift (Python's `//` floors): use `floorDivide` or integer Swift maths.
- `a += b` on an MLXArray changes every reference to that array: write `h = h + x`.
- `import MLX` adds array versions of `min`, `max`, `log`: for plain numbers write `Swift.min`, `Foundation.log`.

*Inputs and preprocessing* (most wrong answers come from here, not from the network)

- Copy the maker's preprocessing exactly: resize size and method, crop, rescale (often 1/255), mean and std, channel order (some models want BGR), and the order of steps. Read them from `preprocessor_config.json` and the processor class, not from memory.
- `Images.load` decodes to RGB in 0…1 and resizes with CoreGraphics; PIL's bilinear or bicubic resize gives slightly different pixels. That is fine for adding a model; note it in `limits` if outputs depend on exact pixels. A centre crop takes the floor of `(size - crop) / 2`.
- Text: check whether the maker adds special tokens (`[CLS]`/`[SEP]`, `<s>`, an end token) and which token or mean the model pools. Pooling the wrong position silently gives wrong vectors. For sentence embeddings, read the repository's `modules.json` and `1_Pooling/config.json` (mean pooling, CLS, last token) and whether it ends with a `Normalize` module.
- Audio: resample to the model's rate (`Audio.load(url, sampleRate:)`), and copy the feature extraction (frame length, hop, mel bins, log or not, normalisation) from `feature_extraction_*.py`.
- Classification labels are in `config.json` `id2label` (or `label_names` for timm): return the label text, not only the index. When the checkpoint has no label names (an ImageNet-1k timm model), put the dataset's standard list in the model folder as a JSON file and load it from `modelDir`; check two or three indices against the maker's list.

*Weights*

- Check `tensors.txt` against your modules: every tensor is used, or skipped on purpose with `rename:` returning nil. A head you skip (a pooler, a training-only head) must not be one the output needs.
- An mlx-community conversion may store tensors in MLX's layout already; the maker's checkpoint stores PyTorch's. Read the shapes in `tensors.txt` and transpose only what is in PyTorch's layout.
- The plan prints the stored dtype; trust the safetensors header, not the repository name.

*Operations that return files* (depth maps, masks, images, audio)

- An Arsenal kind whose operation has `output = "file"` in its kind file (`segment`, `estimate_depth`, every image and audio maker) answers with files. Each request carries `params.raw["out_dir"]`, an absolute folder made for that request: write the files there (`Images.savePNG`, or raw float tensors as `.safetensors` with `MLX.save(arrays:url:)`), close them, and return receipts: `{"files": [{"id": "view-0", "path": "/abs/out_dir/x.png", "mime": "image/png", "bytes": 1234, "sha256": "<64 hex>", "width": 640, "height": 480}], …}` with `Files.size(url)` and `Files.sha256(url)`. Copy every other field the kind's output schema lists. `try` passes an `out_dir` the same way. Never write outside `out_dir`.

*Runner hygiene*

- stdout carries Arsenal's protocol frames: never `print`. Write diagnostics to stderr: `FileHandle.standardError.write(Data("…\n".utf8))`.
- Cache what you can at load (tokenizer, filter banks, position tables), not per request.

### 5. Try it on a real input

```sh
python3 kit/new-model.py try ID OP '{"image": {"path": "/abs/cat.jpg"}}' '{"image": {"path": "/abs/tiny-33x17.png"}}'
```

`try` builds the runner when the source changed (fix the compiler errors it prints, then run it again), loads the real weights outside Arsenal, sends each JSON request, checks every reply against the operation's output schema, and saves the measured load time, request time and peak memory into `results/` and `log/`. Files are `{"path": "/absolute/path"}` objects; text is a plain string. The first request must succeed; later ones must succeed or answer `invalid_request`, never crash.

Use the model card's own example, with its expected answer, as the first request (the card's example image URL, the example sentences, the example clip); download a small example file into your working folder if needed. Then decide whether the answer is **correct**, not only whether it ran:

- a classifier: the card's example gets the card's label (or a plainly right one: a photo of two cats gets a cat label) with a clear top score;
- embeddings: related texts score clearly higher than unrelated ones (with normalised vectors, a paraphrase pair is usually above 0.6 and an unrelated pair below 0.2); where the card prints scores, yours match them to about two decimals;
- depth or segmentation: near things are near, the object's mask covers the object;
- speech or text output: the words are right;
- any output: no NaN, no infinity, the right number of results for the inputs.

If the answer is wrong, do not loop end-to-end. Find the first place it goes wrong: check the preprocessed input (print its shape, mean and range to stderr), then the output of the embeddings, then the first block, against what the maker's code would produce for the same input. Most bugs are a wrong normalisation, a wrong transpose, a missing scale (`1/sqrt(headDim)`), a wrong `eps`, the wrong GELU, or the wrong pooled token.

Add one awkward input as a further request: a very short or odd-sized one (a 33×17 image, a 1-second clip, one word, 7 numbers). It must answer or give `invalid_request`.

A comparison with the maker's own Python code is optional. It needs PyTorch and the maker's package (about 1 GB): offer it in your report and run it only with the owner's yes, outside the model folder. It is a check, never a way to serve the model.

### 6. Describe it

Replace every `TODO` in `model.toml`, `README.md` and, for a local kind, `kinds/`:

- `name`: the full official name with family, size and version, in the maker's words ("ViT Base Patch16 224", "All MiniLM L6 v2"). The name only.
- `summary` (one sentence, what it does for the owner), `description` (2 to 4 sentences: what it is, who made it, what it was trained on, a benchmark claim from the card), `purpose` (a few words, at most 40 characters), `good_for`, `limits` (where it fails; for language models, whether it handles Polish).
- `params`: the plan's count; check it against the card.
- `tasks`, and `languages` as ISO codes: the languages the model reads or writes (a classifier whose labels are English is `en`); `mul` only for a model with no language at all, such as depth.
- `[facets]`: `modalities_in` and `modalities_out`, each from text, audio, image, video.
- For a local kind, the output schema `kinds/local.<name>/ops/<op>.output.json`: list the fields `run()` returns with their types. `try` and Arsenal check every reply against it.

### 7. Check

```sh
python3 kit/new-model.py check ID
python3 kit/new-model.py check ID --load     # also loads the weights into the runner; runs no model
```

`check` runs every static check Arsenal can make without running the model, then Arsenal's own validation. Each problem is one line: a code, the file and field, what is wrong, and `Fix:` with what to do. Fix every line and check again until it prints `ok`. The checks:

| Code | Checks |
|---|---|
| S1 | `model.toml` parses; id matches the folder; required fields filled; no `TODO`; purpose length; facets; a log entry |
| S2 | every operation has input and output schemas; valid JSON; known types; every required field exists; file inputs are annotated |
| S3 | every pinned weight file exists with its pinned size and hash; safetensors headers read; dtype and parameter count match `model.toml` |
| S4 | the runner is the kit's MLX Swift runner, built, with MLX's shaders beside it; sources import MLX and contain no Python, other processes, run-time code loading, network access or other engines; `modelKind` and `modelOps` match `model.toml`; no scripts or other weight formats in the folder |
| S5 | (`--load`) the weights load into the runner's modules with every key and shape matching |
| S6 | the licence is open, or marked as the owner's own (then a note says it stays on this Mac) |
| S7 | every operation has a measured `try` with the current runner |
| S9 | Arsenal's own folder validation |

Add `--json` for the same result as JSON. `check` also takes a folder path, e.g. one already in `local-models/`.

### 8. Install

```sh
python3 kit/new-model.py install ID
```

It runs `check`, copies the folder into `local-models/`, refreshes Arsenal and prints the content digest and the calls that follow. Arsenal also watches `local-models/`, so a folder placed there shows up by itself within seconds. A folder with a problem shows as an error row: `arsenal_find(query:"ID")` and `arsenal_model(model:"ID")` print its problem.

### 9. Approve, download, run

Ask the owner once: "May Arsenal run this model's code (digest …)?" On their yes:

1. `arsenal_settings(action:"approve_code", model:"ID", content_digest:"…")`
2. `arsenal_download(model:"ID", allow_download:true)`, then `arsenal_wait(job:"…")` (it copies the Hugging Face cache files; no second download). The model now appears in the owner's Models table.
3. One real `arsenal_run(model:"ID", operation:"OP", input:{…}, format:"json")`, with the same example as in step 5. Check the answer again.

Report to the owner: what the model does, the example and its answer, the measured load time, request time and peak memory from step 5, and what was not checked (parity with the maker's code, benchmark quality). Then run `python3 kit/new-model.py clean ID`, only after `install` succeeded (the installed copy in `local-models/ID` is then the only copy of your code); it deletes the kit's working folders and keeps the weights in the Hugging Face cache. For an FP32 or BF16 checkpoint those originals are a second copy beside Arsenal's FP16 one: say so, and delete them only with the owner's yes.

If you change `Model.swift` after installing, run `try` and `install` again; Arsenal then asks for a new approval, because the code changed.

## When you are stuck

Stop and tell the owner which step failed and the last error, in one sentence. Do not work around the kit: never run the maker's Python to serve the model, never hand-edit pins, hashes or receipts, never edit the kit's `Package.swift`, `Kit.swift` or `Wire.swift`. To remove an installed folder: `rm -rf "$SUPPORT/local-models/ID"` (the exact path, after looking at it), and `arsenal_delete(model:"ID", remove:true, confirm:true)` if it was downloaded, with the owner's yes.

## What this does not cover

Parity with the maker's code, benchmark quality, speed work and int8/int4 tiers are not part of adding a model; the folder's log says "parity not run". The full qualification for Arsenal's own catalogue is [ADDING-A-MODEL.md](ADDING-A-MODEL.md).
