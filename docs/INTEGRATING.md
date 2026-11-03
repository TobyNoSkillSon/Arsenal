# Integrating a model without weights

A model is ready for the later weight trial when its declared operations work through an isolated native worker on random weights, its parameter inventory matches header-only checkpoint inspection, and all input/output schemas and folder checks pass. This establishes implementation wiring, shapes and precision rules. It does not qualify trained output, quality, speed or peak memory.

Read `MODEL-FOLDER.md`, `RUNNERS.md`, `DEPENDENCIES.md` and a complete working folder (`models/birefnet`). Never download weights during this phase. Config/tokenizer/processor files and exact HTTP Range reads of safetensors headers are allowed. Use `lab/` for private evidence and isolated `ARSENAL_SUPPORT_DIR`/`ARSENAL_CACHE_DIR`; leave the installed app alone.

## Files owned by each integrator

| Group | Registry / implementation package | Worker family |
|---|---|---|
| G1 language, vision chat, translation/OCR/extraction/grounding | `Workers/Language/Sources/LanguageWorker/Registry.swift` | `language` |
| G2 embeddings / reranking | `Workers/Embed/Sources/EmbedWorker/Registry.swift` | `embed` |
| G3 transcription / diarization / turn detection / separation | `Workers/Audio/Sources/AudioWorker/Registry.swift` | `audio` |
| G4 songs | `Workers/Music/Sources/MusicWorker/Registry.swift` | `music` |
| G4 image/video restoration | `Workers/Restoration/Sources/RestorationWorker/Registry.swift` | `restoration` |
| G5 detection / segmentation / photo search | `Workers/Perception/Sources/PerceptionWorker/Registry.swift` | `perception` |
| G6 depth / 3D | `Workers/Geometry/Sources/GeometryWorker/Registry.swift` | `geometry` |
| Image generation and editing (Qwen-Image) | `Workers/QwenImage/Sources/QwenImageWorker/Registry.swift` | `qwenimage` |

Depth may use the Perception kind while its implementation belongs to Geometry. Multimodal photo embeddings may use the Embeddings kind while their implementation belongs to Perception. The manifest chooses kind and worker independently. Curated Chat-kind models enter the catalog when they declare image/audio/video input or a translate/read_document/extract/ground_screen operation; general text chat remains user-added. Existing Chat remains available for the current Qwen adapter; new language adapters use `language`, with the same Chat kind and HTTP/MCP profiles. Registry ownership, not a universal mega-registry, avoids shared edits.

Add model implementation files and tests only to your family folder, plus `models/<id>/` folders. Tests live in `Workers/<Family>/Tests/` and are declared in your family's section of `Workers/Package.swift`, the one native build graph; your family manifest declares libraries only (a test or executable target there would compile a second MLX). Append `NativeRegistration(["checkpoint_model_type"]) { Adapter() }` to your registry. Shared pins, dispatcher, HTTP routes and MCP names are already owned by the foundation; report a missing contract rather than editing another family's files.

## Adapter contract

Implement `NativeModel` from `Workers/IntegrationKit`: `architecture`, `operations`, `parameters`, `peakBytes`, `load`, `run` and `unload`. Keep every stored learned parameter in the `parameters` inventory, including packed uint32 matrices; exclude computed integer index buffers explicitly. `load` receives verified model/weight folders, selected precision, settings, asset directories and a lab-only random-weight flag. No worker fetches missing assets.

`NativeFamilyWorker` supplies framed hello/request/reply, serialized engine execution, responsive cancellation, process memory, load timing, inventory and unloading. Check `NativeCancellation` between bounded tokens, steps, batches, frames and file writes. `emit` receives `WireFrame.json` events without an ID or `WireFrame.binary` PCM chunks; the host adds the request ID atomically. One request gets one terminal reply. An unloaded worker exits. Report unsupported architecture/operation as an error, never an empty successful result.

Use `NativeMedia.audio` for bounded decode/resample/interleaved host PCM, `NativeMedia.image` for bounded image decode and `NativeVideoReader`/`NativeVideoWriter` for frame-at-a-time AVFoundation processing. Video timestamps and transforms are explicit. The basic writer encodes video only; an adapter promising preserved audio must mux the original audio track and test that path. Write artifacts only into the supplied `out_dir`, then use `NativeMedia.file` and return all outputs as `files[]`, each with a nonempty unique id, path and MIME. Image metadata includes actual width/height; the service adds/verifies byte counts and hashes. A 3D result can include GLB and preview image files together. Default HTTP/MCP limits are still 16 MiB upload and 64 MiB combined output; oversized long video/3D results fail explicitly. Do not claim unlimited long-video serving.

`NativePrecisionTier` in `Workers/IntegrationKit` installs the selected snapshot produced by Get: half tensors or stored group-64 affine int8/int4 packs. Stored packs are installed directly, with matching selected bits, mapped dimensions, embedding dimensions, bias shapes and complete companion groups. A half source may also derive its selected tier at Load; it is never required when Get has already stored the pack. FP16 remains the default. Supply an explicit checkpoint mapping and matrix eligibility predicate; convolution reshaping/fused projections remain model-owned. Pass `retainedHalf` only for int8 components with recorded failed quality checks. Spatial kernels/norms remain half. BF16 requires a documented FP16 failure. Immediate owning arrays are replaced completely, preserving unchanged activations and blocks; sparse flattened module replacement is invalid on the pinned MLX. Never quantize an existing quant, hold a second dense source model, or claim quantized quality from a random smoke. Test nontrivial bias order, layout mapping, packed companion names and complete coverage.

## Folder and operation declarations

Add `model.toml`, README, gate/bench recipes, provenance, source pins/file sizes/hashes, and `gate/no-weights.json`. Set `schema = 3`, `[runner].worker` to your family, native FP16 (BF16 only with evidence), and no derived int8/int4 recipe unless the owner asks for a quantized tier (below). Register `NativeConversion(recipe:half:eligible:)` next to the adapter for flat safetensors snapshots. It uses the existing streamed converter, preserves config/tokenizer files and returns hashes; the host advertises convert only when a converter is registered. The same initializer now preserves nested paths. For transforms or multiple components use the additive callback below before advertising recipes.

### Get-time storage precision

Quantized tiers exist to save memory; every model performs best at its 16-bit base. Quantized tiers are optional, never required: a model ships at its 16-bit base alone unless the owner asks for int8 or int4. Never add one to a model under 1B parameters (under 2 GB at FP16): 8-bit would save under 1 GB, and one-pass models (detectors, depth, interpolation, embedders, rerankers) are compute-bound, so packed weights add dequantization work without speed. A tier whose quality check fails is removed rather than shipped with a warning.

For the IntegrationKit families (`language`, `embed`, `audio`, `music`, `restoration`, `perception`, `geometry`, `qwenimage`), the serving base is FP16 by default regardless of the pinned source's `weights.dtype`. Keep weights.dtype truthful about the download: a BF16 source remains declared BF16 even though Get installs FP16. An explicit policy also enables this path for another native worker. Legacy workers without the policy keep their existing recipes.

A BF16 exception requires a folder declaration and a local evidence file, for example EmbeddingGemma 2's documented activation overflow:

```toml
[precision]
dtype = "bf16"
reason = "The maker documents FP16 activation overflow."
evidence = "log/001-source.md"
```

Without that declaration the new native families serve FP16. Merely putting BF16 in source metadata or a limits sentence does not declare an exception. Owners of existing BF16-only scaffolds must add the declaration. The evidence file must be present in the verified folder; the declaration does not substitute for the later trained-weight trial.

Get invokes the reserved shared recipe `cast:native-half-v1` one source file at a time, verifies each output receipt, removes its original before fetching the next file, and commits the serving half snapshot with raw source recovery pins and normalization provenance. Correctly typed half archives are cloned without re-encoding; BF16/FP32 archives are streamed into the selected half. Metadata/tokenizers and nested relative paths survive. No model-specific cast registration is needed. Get retires a cold old raw-half snapshot after verifying its replacement; a resident lease or another model's live blob reference is preserved. External source caches are untouched.

Models needing architecture/layout transforms keep `[derive] fp16 = "cast:my-layout-v1"` (or bf16 matching the declared exception). Get first normalizes the pinned source file by file, then calls that model-owned converter with the complete half source and verified assets. Thus bounded groups can span source files without retaining the entire original checkpoint. The callback still assembles all learned asset components into its standalone selected snapshot using context.half; raw assets are inputs, not a second runtime path. Adapter Load must use the selected precision/`context.config["serving_dtype"]`, rather than config.json's original torch_dtype. Quantized Get starts from this normalized base, and both Get and Load validate provenance against its normalization policy. Load is not the storage conversion stage. A transform that needs full-range values before the half cast (Big-LaMa folds BatchNorm running variances past FP16's 65,504 into its convolutions) declares `[precision] source = "raw"` with `reason` and `evidence` on an FP32 source: Get then skips the normalization and hands the verified pinned files to the model's own `cast:` recipe, which stores the selected half; provenance records `raw-cast-v1:<half>`.

### Model-owned conversion and multiple repositories

Declare other repositories with the existing `[[asset]]` fields: `id`, `repo`, `repo_type = "model"`, pinned `revision`, exact `files`, per-file `sha256`/`bytes`, and `licence`. Optional top-level `assets = ["vae", ...]` selects required assets; absent means all. Get verifies these snapshots before conversion and passes their directories as `assets` on the convert request. Existing converter conformances keep working through a default protocol bridge. New conformances implement the asset-aware `PrecisionConverter.convert(...assets:...)` overload; WorkerPrecisionConverter already does this. New derived provenance binds all component pins, and changing them invalidates the snapshot. No worker downloads assets.

```swift
let conversion = NativeConversion(recipe: "cast:my-model-v1") { context, cancellation in
    // context.source: primary source; context.assets["vae"]: verified second repository.
    // context.destination: private empty staging; context.precision, half, bits: selected tier.
    try context.convertTree(from: context.source, cancellation: cancellation, eligible: eligible)
    guard let vae = context.assets["vae"] else { throw PrecisionError("Missing VAE") }
    try context.convertTree(from: vae, into: "vae", cancellation: cancellation, eligible: vaeEligible)
    return try context.receipt(cancellation: cancellation)
}
```

For weight-normalization folding, fused/split projections, renamed keys or layout transforms, use `TensorArchive.entries` to index shards, read only the bounded tensor group needed for one transformation, and write/release it before advancing. Feed transformed keys/read closures to `StreamedPrecision.convert` so selected half casts and group-64 packing remain shared. `TensorWriter` permits custom incremental output and creates nested parent directories. Use autorelease pools per group, check cancellation between groups/writes, and never accumulate the complete dense checkpoint. The callback returns the ordinary `{files:[{path,bytes,sha256}]}` receipt, easily generated by `context.receipt`; paths are relative to destination. The host requires a complete unique file list and tensors; Get verifies every hash/byte count and rejects unlisted files. Nested relative paths and config/tokenizer files survive `convertTree`; symlinks, special files, source/destination overlap and nonempty staging are refused.

An auxiliary repository the adapter only needs as converted weights (SAM Audio's T5 text encoder) declares `into = "text_encoder"` on its `[[asset]]` and is registered with `NativeConversion(recipe: "cast:my-model-v1", eligible: eligible, components: ["t5-id": "text_encoder"])` (and the same `components` on the derived recipes). Get converts it once into `text_encoder/` of the base snapshot in the selected half, packs it with the tiers, and keeps no separate copy; Get refuses a conversion that leaves the declared directory empty. Load reads `context.weights/text_encoder` and converts nothing; the asset is absent from `context.assets`.

Produce a complete independently loadable snapshot: YuE2's LM and converted VAE, or ACE-Step's DiT, planner, text encoder, VAE, silence latent and both tokenizers. Every learned component participates in the selected tier. Load converted components from `context.weights` subdirectories, never the raw asset source. int8/int4 normally receive the already converted base snapshot as source; distinguish this from vendor layout and do not fold/rename twice or re-quantize packs. A common pattern is one callback for the base `cast:...` recipe and a second identity-layout callback for int8/int4. The base callback transforms raw assets; subsequent callbacks stream the complete base snapshot. Raw asset directories are source inputs, not independently selected runtime components. Sources without FP32 still need a base `cast:...` recipe when layout/component preparation is necessary. Recipes and folder pins remain model owned; the callback does not certify numerical conversion quality.

### Checkpoints without model_type

Set top-level `architecture = "rife"` in model.toml and register `NativeRegistration(["rife"])`. The validated folder selector travels in the served load config and the no-weight smoke. The native host accepts absent config.json or a config without model_type only with this selector, requires exactly one registry match. The verified folder architecture wins over model_type (including vendor aliases); without it, model_type remains the legacy fallback. DA3 folders use this same key when the original config has only model_name. A checkpoint whose configuration file has another name (HT-Demucs ft ships `htdemucs_ft_config.json`) declares no config path: the architecture key alone dispatches it, and the adapter reads its own file from `context.weights`. Existing folders continue dispatching by checkpoint model_type. The selector is included in the load identity and folder digest; do not infer an architecture from an unverified filename.

### Models with zero eligible matrices

Keep the ordinary half/int8/int4 recipes and declare:

```toml
[quantization]
eligible_matrices = 0
reason = "Spatial convolution kernels only; no Linear or Embedding matrices."
```

This declaration is only for genuinely zero eligible matrices, not failed quantization quality. int8/int4 requests then run the same half spatial kernels with no compression. The API's additive `precisionDtypes` maps the int8/int4 selection IDs to the actual half dtype. Precision-cell labels show that half dtype; `precisionNotes` and existing hovers explain no compression. The irrelevant int4 quantization-loss warning is absent (this does not mark model quality passed); do not claim 8/4-bit weights or memory/speed savings. Figures remain independently measured or absent. The smoke permits unpacked tiers only with this declaration, rejects contradictory packed or potentially eligible rank-2 group-64 tensors, and reports effective_dtype/eligible_matrices/compression for every tier. A model with noneligible rank-2 tensors needs a more specific reviewed eligibility receipt; this conservative zero-matrix exception cannot silently exclude matrices.

Kinds are `audio`, `embeddings`, `chat`, `sound`, `restoration`, `perception` and `geometry`. New capabilities are opt-in: explicitly list each supported `[[op]]`; give each a model-local narrowed schema or reference `kinds/<kind>/ops/<name>.input.json` / `.output.json`. Do not expose options the architecture cannot implement. The schema validator and service enforce the declarations across HTTP, MCP and tool windows. Optional `good_for` and `limits` lists supplement `purpose`; reuse `languages` and `[licence]`. Fill in the full model information in this phase, and treat the later weight trial as the measurement run; a model is not added until both are done ([skill, Introduce a new architecture, steps 3 and 4](../skill/ADDING.md)).

Request objects reach the worker as decoded JSON, whose key order is not preserved. When an object's key order matters to the model (NuExtract's extraction `template`, a decision model's `state` and `questions`), mark that top-level input property `"x-arsenal-ordered": true`. HTTP then forwards the request body's own bytes for it, and the worker receives the decoded value as usual plus `ordered_json.<name>`: the exact text, in the order sent. A property typed `"object"` may also be sent as the JSON text of its object, which is how an MCP client keeps the order; a property that may hold any JSON value keeps a string as the text it is, and such a client sends the value plus `ordered_json: {"<name>": "<its JSON text>"}`, which must decode to that same value. Parse that text with an order-preserving reader; the decoded value remains the fallback for a client that sent a plain object through a route that re-encodes it.

File controls use `x-arsenal-file.accept`; semantic controls use `x-arsenal-control` (`image-picker`, `audio-picker`, `video-picker`, `document-picker`, `image-points`, `image-boxes`, `lyrics`, `language`, `slider`, `json-schema`). Point/box controls identify `x-arsenal-image`; coordinates are normalized to the displayed image's original extent. Detection/depth outputs explicitly label coordinate/depth units. Screen grounding also returns pixel coordinates with image_size; adapters convert from their model space to the original screenshot dimensions. Depth accepts image or a multi-view images array, with indexed per-view file IDs/camera data. UI implementation is separate; unsupported semantic controls currently retain the generic JSON form.

A segmentation session is scoped to the loaded model process. `segment_session(action:create,image)` returns a session ID; `segment` reuses it; `action:close` releases it. `video_session(action:create,video)` creates reusable tracking state and `action:close` releases it. `track_video` accepts video or session_id, plus prompt_updates with frame_index, object_id, add/update/remove action and points/boxes/text. Adapters bound session count/memory and reject stale IDs after unload/reload.

Concept prompts by example (SAM 3.1): `segment`, `track_video` and each `track_video` prompt update accept an optional `visual_exemplars` array (1–32) of `{image:{path}, box:{xmin,ymin,xmax,ymax}, label:foreground|background}`. Each box is normalized to its own example image, not to the target image or frame; example images pass the same inbox/jobs confinement and MIME checks as every file input. The field is additive in the kind schemas; a model offers it only by copying it into its narrowed model-local schema, and its adapter must then honour it (a SAM build without exemplar encoding keeps it undeclared).

Multi-vector image embeddings (TIPS raw patch and register tokens) are deferred: one 448 px B/14 image is 1,024 patch tokens × 768 values, about 3 MB as half-precision binary and roughly 10 MB as JSON numbers, so it needs a file output contract (a tensor file in `files[]`) rather than the JSON `embeddings` array. `embed` returns pooled vectors only.

### Installed-model image preparation

A folder can declare a single image pre-step on its operation:

```toml
[[op]]
name = "image_to_3d"
title = "Make 3D"
input = "kinds/geometry/ops/image_to_3d.input.json"
output_schema = "kinds/geometry/ops/image_to_3d.output.json"
output = "file"
stream = false
[op.prepare_image]
model = "birefnet"
operation = "remove_background"
file_id = "foreground"
```

The host validates the original input, runs the dependency with `image` and a private job directory, selects the named PNG file, validates a visible RGB foreground plus alpha transparency, and replaces `image.path` before loading/running the target. TRELLIS receives an ordinary RGBA file; its adapter needs no pipeline API. It must still validate direct worker inputs. Operation descriptors expose the additive `prepareImage` declaration. Other operation inputs remain unchanged.

The dependency must already be added and installed at its selected precision. Missing models/weights return `dependency_not_installed` naming the model and Get action; nothing is fetched automatically. Ordinary trust, memory/force-load and cancellation policies apply to both operations. The dependency has its own Activity entry; its intermediate directory is removed after target completion, failure or cancellation. The host does not unload an existing dependency resident behind another request. Installing both models does not guarantee they fit together: normal memory admission reports that shortage.

This is one image pre-step, not a recursive pipeline: a dependency cannot itself declare a pre-step and cannot be the target model. It must expose a schema-3 file/image operation accepting `image`; additional required inputs cause normal schema rejection. `file_id` is explicit so masks/previews cannot accidentally feed 3D. Invalid IDs, non-PNGs, RGB/opaque/empty-alpha images and output paths outside the private directory fail before the target runs. Structural alpha validation does not qualify segmentation quality. Test the host chain with fixtures as well as the adapter's direct RGBA smoke.

Terminal worker replies release request admission before becoming observable, on both success and failure; integrators can send the next sequential request immediately after a terminal reply. Concurrent requests still receive busy.

## No-download checks

Build and test through the repository scripts: `scripts/test.sh --changed` (app and native suites), or `scripts/test.sh --native --filter <YourTestTarget>` for your family's suite alone. Model inference and GPU parity tests are tagged `.heavy` and skipped by default; run them for your family with `scripts/test.sh --heavy --filter <YourTestTarget>`. `scripts/dev-build.sh` builds without running tests and lays out the dev helpers in `Workers/.build/helpers/`. Run model-folder validation with the built headless `Arsenal catalog validate`; inspect its help for the exact folder invocation. This uses the same manifest/schema checks as local folder refresh and never grants consent or signs the catalog.

For the random graph smoke, provide `gate/no-weights.json` with `config` (real or explicitly shrunk architecture settings), base64 input `files`, and an `operations` array of `{operation,input}` for every declared operation. `$INBOX` and `$OUTPUT` are replaced with private test directories. The adapter's `load` branches on `context.randomWeights`, constructs its real graph, initializes half parameters, and performs no checkpoint read. The ordinary `run` path supplies results. Fixture substitutions or constant output implementations do not establish model integration.

Random weights generate text that a structured operation's parser (JSON extraction, document layout, grounding coordinates) must reject. Such a case declares `"expect_error": "invalid_model_output"` and passes only when the worker replies with exactly that code; any other failure, or a successful result, fails the smoke, and no other expected code is accepted. `invalid_model_output` is reserved for that one rejection: the adapter's parser throws it on generated output it cannot accept, and nothing else uses it. Every operation checked this way also needs positive evidence in `gate/parser-fixtures.json`: `{"cases":[{"operation","input","raw","expected","source"}]}`, where `raw` is a maker-documented example output (model card, maker parser tests or reference repository; `source` names it with its pinned revision), `input` the request it answers (`$INBOX` files allowed) and `expected` the schema-valid result. A case may instead declare `expect_error` for a documented malformed output. The adapter conforms to `NativeOutputParsing`, whose `parseOutput` is the parser `run` calls after generation; the checker sends each case to the lab-only `parse_output` worker request at the base tier, requires the exact `expected` result and validates it through the production output validator.

```sh
# Run from the repository; the conformance venv supplies jsonschema.
tests/conformance/.venv/bin/python scripts/model-check.py smoke \
  --model models/birefnet --worker dist/Arsenal.app/Contents/Helpers/arsenal-worker-birefnet \
  --inventory lab/birefnet-parameters.json

tests/conformance/.venv/bin/python scripts/model-check.py header \
  https://huggingface.co/ZhengPeng7/BiRefNet/resolve/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4/model.safetensors \
  --parameters lab/birefnet-parameters.json --birefnet --output lab/birefnet-header.json
```

Random workers keep their family symlink name: pass `--worker Workers/.build/helpers/arsenal-worker-<family>` (a copy of the debug native worker, its family links and the shared metallib) or the bundle's `Helpers/arsenal-worker-<family>`. IntegrationKit workers use a lab-only MLX allocation ceiling of at most 8 GiB or half the currently reclaimable VM pages, while every smoke has a 590-second total deadline plus 90-second exchange deadlines. Every declared precision tier runs every declared operation, requires nonempty tensor/operation counts, validates outputs through the production OperationArtifacts path via the headless runner checker, and unloads with exit 0. These limits do not apply to ordinary serving.

The checker refuses HTTP 200 before reading its body, checks exact Content-Range and total size across requests, caps the header at 64 MiB, compares exact names/shapes/dtypes, and never reads tensor payloads. The BiRefNet map removes 67 integer buffers, normalizes `conv_out1.0`, and changes OIHW shapes to OHWI. Other adapters export parameter names in checkpoint layout or add and test an explicit mapping. Pass all shard URLs after `header` to merge their inventories; duplicate tensor names and combined headers above 64 MiB are refused. A model-owned `--mapping models/ID/gate/header-map.json` can declare exact `ignore` names and `tensors` entries with `name` and `permutation` fields. For example, `{"ignore":["counter"],"tensors":{"conv.weight":{"name":"encoder.conv.weight","permutation":[0,2,3,1]}}}` maps OIHW to OHWI. Keep these mappings in the model folder; integrators do not add model-specific branches to the checker. Dtypes stay strict except an explicit per-source-tensor FP32-to-FP16 Get plan in the mapping: `"casts":{"conv.weight":{"from":"F32","to":"F16","at":"Get"}}`. The checker reports `planned_casts`, still compares every mapped name and shape, and refuses missing cast names, a different source dtype or any other cast pair. This records a planned cast, not conversion-quality evidence. For native BF16 sources normalized by Get, the same explicit mapping accepts `from:"BF16"` to `to:"F16"` at Get. This reports the storage plan and still does not qualify activation quality. FP32 source folders declare a half base cast recipe; smoke runs the effective half base and every derived tier, never FP32.

Regenerate `docs/MODELS.md` with `python3 scripts/generate-model-docs.py`; `--check` fails on stale output. Update the model README's actual limits and pending qualification. Do not invent measured size/memory figures.

## Completion and handoff

Run all declared operation smokes, inventory/header checks, malformed-input checks, cancellation and unload checks, model-folder validation and affected tests. Commit on the assigned branch; return exact commit, commands/results, source/config choices, declared operations and unresolved trained-weight/reference/quality checks. No signing, install, push or trained-weight download occurs here. The coordinator reviews, merges and later conducts the user-approved weight trial and full precision qualification.

Provider rerank aliases (`/v1/rerank`, `/v2/rerank`, `/providers/cohere/v2/rerank`) forward `return_embeddings` into the selected model's rerank schema. Jina folders declare it as a boolean and may return per-result embedding vectors. An undeclared option is rejected by schema validation instead of silently dropped. Universal rerank operations use the same model schema.
