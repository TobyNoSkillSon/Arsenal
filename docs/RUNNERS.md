<!-- SPDX-License-Identifier: MIT -->
# Arsenal runners

A runner is an executable process that reads requests on stdin and writes framed replies on stdout. An engine can use MLX, Metal, Core ML, C/C++, Rust or another native runtime. Arsenal supplies verified model files, weights, assets and configuration; the runner owns inference. The same protocol serves every engine.

## Build and check

1. Implement the lifecycle and the operations declared by the model's `[[op]]` tables, using shared schemas in `kinds/<kind>.toml`. Use their input schemas, not an API provider's request body.
2. Add schema-valid smoke inputs in `gate/runner-check.json`. Include a cancellable workload and nondefault load settings. Run the checker in both modes and every offered dtype on the chips you claim.
3. Pass the model's numerical/quality gate separately. Preserve outputs and commands as evidence. Wire conformance alone does not establish model correctness.
4. Record each attempt in `log/`, including failures. Submit the runner, gate evidence and compatible performance results using the PR flow below.

The installed app exposes its checker as `Arsenal.app/Contents/MacOS/Arsenal runner-check`. Catalog tools use `Arsenal.app/Contents/MacOS/Arsenal catalog`. Both dispatch headlessly before AppKit and the single-instance claim, so they work while the menu-bar app is running. The checker still launches each runner in a separate child process with isolated support/cache directories; the service retains local-code consent. For development, build the app executable from the repository root:

```sh
scripts/mlock work runner-check-build -- env DEVELOPER_DIR=/Library/Developer/CommandLineTools \
  swift build --product Arsenal
.build/debug/Arsenal runner-check path/to/native-runner --model models/<id> \
  --weights /absolute/verified/snapshot --kind-dir kinds --timeout 30
```

All runner command arguments precede `--model`; checker options follow it. `--fast` selects Fast; the default is Exact. A structured fixture's `load.dtype` selects a dtype. `--weights` defaults to the model folder (useful only for the no-weights toy). `--kind-dir` names the top-level `Resources/kinds/` directory; otherwise the checker searches ancestors of the model folder. `--timeout` bounds each exchange to strictly above 0 and at most 180 seconds, default 30. `--results DIR` keeps stderr and outputs in that directory; otherwise the checker creates `/tmp/arsenal-runner-check/<uuid>/`. It sets isolated `ARSENAL_SUPPORT_DIR` and `ARSENAL_CACHE_DIR` for every child.

On this shared development Mac, CPU builds/tests run through `scripts/mlock work`. A checker that loads real GPU weights must run under `scripts/mlock gpulock <label> -- perl -e 'alarm 180; exec @ARGV' …`, with current authority and local copies; neither this document nor a fixture authorizes a model run. Ordinary CI CPU toy checks need no GPU lock.

## New family adapters

The `IntegrationKit` protocol and per-family registries are the shared native seam for language, embed, audio, music, restoration, perception and geometry. See [INTEGRATING.md](INTEGRATING.md) for adapter ownership, half/int8/int4 loading, media I/O and the lab-only random-weight/inventory path. Existing family workers remain supported. The foundation provides lifecycle machinery; each model still implements and tests its own operations and conversion recipes.

## Wiring a new native model

Introduce the model before running the tier/measurement intake in `ADDING-A-MODEL.md`. Follow `AGENTS.md` and `skill/ADDING.md` (Introduce a new architecture, step 0): choose and confirm the exact source first; keep its weights intact, without re-quantizing. Default to FP16, comparing a BF16 → FP16 cast against the original on fixed tensors; retain BF16 only if FP16 fails. Reuse an existing native family or Swift MLX port before writing a Swift port on the pinned MLX. Python runs only the independent reference, never serving inference. Check component keys/shapes/counts and numerical parity, then one end-to-end smoke with current model-run authority. Quantized tiers and performance qualification follow on request; static correctness is not a measured table figure.

### Inventory and pin before porting

Inventory the complete inference graph: backbone, text/prompt encoders, tokenizer/vocabulary, image/audio processor, decoder/codec, conditioning/voice assets, configs and licences. For each component record its role, repository, immutable 40-hex revision, exact filenames, byte sizes, SHA256, source dtype, and destination tensor names/shapes/layout. Check that the chosen implementation matches that exact variant, including preprocessing and postprocessing; a shared model name is insufficient.

Prefer the same variant's 16-bit MLX or original safetensors release. If the available source is a **16-bit GGUF**, inspect every tensor's stored type (the filename alone proves nothing), metadata, tokenizer and architecture. Locate an equivalent original release first and confirm any source substitution. Otherwise a model-specific converter must read the pinned GGUF, preserve tensor values while mapping names/layouts to the Swift loader, and emit independently loadable half weights plus configs/tokenizer. Record converter source revision, mapping and input/output hashes; assert complete tensor coverage and compare fixed tensors before the smoke. Never recover a supposed 16-bit original by expanding quantized GGUF tensors. Arsenal currently has no generic GGUF → MLX converter: listing `.gguf` in a manifest does not add one, and there is no universal conversion command to run.

For a **multi-repository model**, pin every repository independently. Put the primary checkpoint in `[weights]` of `models/<id>/model.toml`; list auxiliary repos as `[[asset]]` declarations selected by `assets`, each with its own revision, exact files, hashes, sizes and licence (`MODEL-FOLDER.md`, Weights, variants and assets). Load receives asset-ID → verified-directory mappings; do not flatten overlapping filenames or fetch missing components in the worker. Convert each weight component with its pinned converter/mapping and input/output hashes; preserve non-weight assets and verify the assembled snapshot. Get derives the primary checkpoint before downloading assets separately; it does not automatically cast or merge asset weights. If an asset needs conversion, implement and test that preparation path explicitly. A required Get-time cast/derive belongs in the native family's `convert` handler and `[derive]` recipe, through `Sources/ArsenalCore/Runtime/WorkerArtifact.swift` (`WorkerPrecisionConverter`) and `Sources/ArsenalCore/Fetch/ModelFetcher.swift`; check its recipe routing rather than assuming a new `cast:` string works. Keep temporary sources/conversions in an isolated lab profile and leave installed weights untouched.

### Native integration order (BiRefNet example)

If an existing family already loads the architecture, extend its loader and model declaration; a new family requires all four wiring seams below.

1. **Model package and declaration.** Implement the Swift MLX loader and ArsenalWire worker in `models/<id>/runners/swift/`, with dependency locks and upstream licence/provenance. BiRefNet uses `models/birefnet/runners/swift/Package.swift`, the `BiRefNetWorker` library and `Sources/BiRefNetWorker/Worker.swift` there; `Workers/BiRefNetEngine` links to that package. Its `models/birefnet/model.toml` selects `native`, `language = "swift"`, `worker = "birefnet"`, protocol 1 and the image-input `remove_background` schemas under `models/birefnet/ops/`. Declare all supported operations, gate/bench recipes and only real settings; omit an unused `settings.toml`.
2. **Package dependencies.** Add the family package and its worker product to `Workers/Package.swift`, the one native build graph (both `dependencies` and the `arsenal-worker-native` target), and declare the family's test target there. BiRefNet explicitly names the package `BiRefNet` at `BiRefNetEngine` and links `BiRefNetWorker`; explicit naming avoids SwiftPM identity collisions with a worktree called `birefnet`. Keep MLX dependencies on the pin in `scripts/pkg-lib.sh` and commit `Workers/Package.resolved`.
3. **Native dispatch.** Import the worker in `Workers/Native/Sources/arsenal-worker-native/Main.swift` and dispatch `arsenal-worker-birefnet` to `BiRefNetWorker.run()`. Dispatch uses the executable's family symlink name, not a model ID argument. `Sources/ArsenalCore/Runtime/WorkerArtifact.swift` resolves `worker` (or kind if omitted) to that same helper name.
4. **Packaging families.** Add `birefnet` to `NATIVE_FAMILIES` in `scripts/pkg-lib.sh` and map its nonstandard directory in `family_dirs()`. `scripts/assemble.sh` consumes that list to create relative helper symlinks and collect family licences/provenance; `scripts/smoke-helpers.sh` checks the links. `scripts/build.sh` builds the combined native binary and pinned no-fast-math MLX resource bundle. Resource bundles must accompany the helper. `scripts/dev-build.sh` builds the debug native graph and its helpers in `Workers/.build/helpers/`.
5. **Quality identities.** Add the actual family source closure to `FAMILIES` in `scripts/quality-identities.py`: BiRefNet includes both `Workers/BiRefNetEngine` and `models/birefnet/runners/swift`. Update `scripts/test-quality-identities.py` to prove family edits invalidate its identity without staling unrelated families. Assembly writes `quality-identities.json` in Resources; `BundledWorkerArtifact` looks it up by `worker`. Changed source/recipes/load configuration need current quality evidence, not an inherited pass.
6. **Check and hand off.** Run the affected tests and catalog validation; with explicit model-run authority, exercise ordinary Get/Load/operation and the runner checker above against the assembled family helper. Add a task-specific static gate when the adapters in `Sources/ArsenalCore/Measurement/ModelIntakeCommand.swift` do not support the operation: a generic image-generation intake is not a background-removal check. Keep raw captures/history in `lab/evidence/<id>/`; retain current compact bindings and table results in the model folder. Catalog signing, installation and publication belong to the separately authorized integration step.

After editing this document, update its `RUNNERS.md` entry in `Sources/ArsenalCore/Manifest/IntegrationReferences.swift`; `DropInTests.bundledIntegrationReferencesMatchDocs` checks the shipped mirror byte for byte after trimming outer whitespace.

## Process and environment

One process serves one resident runner. Send hello before loading weights. While idle, block on stdin: no timers, polling, network refresh or background conversion. Keep stdout exclusively for frames, including during imports and engine initialization; logs and tracebacks go to stderr. Flush each complete frame. Concurrent writers serialize the entire header and body, not separate writes.

The supervisor launches an argv array, not a shell string. Get has already verified the pinned weights and assets. Runners do not fetch weights, install packages or receive Hugging Face credentials. Native/custom engines ship as reviewed, signed artifacts. Python is used only for independent reference implementations and test tooling. Get never compiles native sources on the user's Mac. A process boundary isolates crashes, not malicious code: unsigned local code needs explicit consent before first run.

Use request paths, not the working directory, to locate files. `ARSENAL_SUPPORT_DIR` names the support root; `ARSENAL_CACHE_DIR` names regenerable caches. Outputs go under `<support>/jobs/`. The production supervisor also uses `ARSENAL_CATALOG` and `ARSENAL_API_PORT`; runners need neither to perform inference. Ignore unrelated inherited variables. Respect input EOF by cancelling work, releasing resources and exiting. Do not daemonize or leave helper children alive.

## Framing

Every frame is `u32 big-endian body length | u8 kind | payload`. Length includes the kind byte, excludes its four-byte header, and must be 1 through 16,777,216 inclusive. Exact-read loops are necessary: a pipe read can return part of a header/body or several frames. Clean EOF is allowed only before a header; EOF inside a frame is truncation. Reject zero/oversized lengths before allocating the body, unknown kinds, invalid UTF-8/JSON and truncated binary headers; terminate the connection rather than trying to resynchronize.

Kind 1 contains a UTF-8 JSON **object**, with finite numbers. This complete ping frame is 25 bytes:

```text
00 00 00 15 01 7b 22 69 64 22 3a 31 2c 22 6f 70 22 3a 22 70 69 6e 67 22 7d
             ^ JSON payload: {"id":1,"op":"ping"}
```

Kind 2 contains `u32 big-endian stream ID | bytes`. For stream 1 and two PCM samples (`0`, `1`, signed 16-bit little-endian):

```text
00 00 00 09 02 00 00 00 01 00 00 01 00
```

These are binary frames, not JSON lines, HTTP, SSE or base64 audio. Newlines have no framing role. Use `Packages/ArsenalWire` in Swift when convenient. A standalone implementation only needs the protocol in this document.

## Hello, requests and replies

The first outbound frame is exactly one hello:

```json
{"event":"hello","proto":1,"worker":"voices","runner":"native","pid":123,"build":"<source-or-build-id>","kinds":["voices"],"ops":["load","ping","status","cancel","unload","speak","voices"]}
```

`pid` is the actual process PID, `proto` is integer 1, names/build are nonempty, and `kinds`/`ops` describe what this process supports. Advertise every common lifecycle operation and every declared model operation; advertise `convert` only if implemented. The supervisor validates kind, protocol and selected runner.

Requests flatten parameters into an object: `{"id":7,"op":"speak","text":"Hello","stream":true}`. IDs are positive integers, unique among outstanding requests. `id`/`op` are reserved envelope keys and are excluded when validating params against an op schema. A malformed envelope is a protocol error; a well-framed invalid operation input is a recoverable error. Check types and bounds before starting engine work.

Each accepted request ends with **one** terminal reply, either `{"id":7,"ok":true,...resultFields}` or:

```json
{"id":7,"ok":false,"error":{"code":"invalid_request","message":"Text must be a nonempty string."}}
```

Errors use a stable nonempty code and one clean sentence; details can be additional fields. Required meanings: `unsupported_op` for an unknown op; `invalid_request` for invalid params; `not_loaded` when inference requires load; `busy` if work cannot be queued; `cancelled` for interrupted work; `unsupported_recipe` for an unknown conversion recipe. Engine failures are framed errors, with diagnostic detail on stderr. Failure must not masquerade as an empty successful result.

Streaming precedes the terminal reply: `{"id":7,"event":"token","text":"Hello"}` or another kind-defined event. Events carry the request ID. Pushed status has no ID: `{"event":"status","memory":{...}}`. The terminal reply is not another event. Multiple outstanding requests may interleave, but a given request's frames preserve order. Audio stream IDs must distinguish concurrent streams.

## Lifecycle and configuration

### Load

```json
{"id":1,"op":"load","model_dir":"/verified/models/id","weights_dir":"/verified/snapshot","dtype":"bf16","fast":true,"assets":{"voices":"/verified/asset-snapshot"},"config":{"settings":{"context_limit":262144},"context_limit":262144,"kernels":[]}}
```

- `model_dir`: immutable validated folder, including ops, kernels and gate metadata.
- `weights_dir`: verified standalone snapshot for the requested dtype; never assume vendor filenames outside this directory. Dtypes are `bf16`, `fp16`, `int8`, `int4` when the folder offers them. A fp32 source must first derive a supported 16-bit snapshot; fp32 is not a selectable dtype.
- `assets`: asset-ID → verified directory, separately pinned and shared across models. Read auxiliary files from these directories; do not duplicate downloads. Existing service launches also carry this mapping in `config.assets`; normalize top-level `assets` first, with `config.assets` as the compatibility fallback. The checker supplies both.
- `fast`: false = Exact, true = Fast; independent of dtype. Arsenal supplies the effective automatic mode from runner/kernel support, not a user switch. The example context is Qwen's current maximum; use each model's configuration-derived maximum.
- `config.settings`: load-time setting-ID → scalar value. The service supplies persisted values or declared defaults. Optional `maps_to = "config.<key>"` also copies the value into that legacy config key; new engines should read `settings`. Request-time settings are merged into op params, with explicit params winning, and do not arrive as pending load changes.
- `config.kernels`: resolved declarations for this chip family and mode, or an empty array. Each item has `id`, `chip`, `mode`, `files`, `entry`, `replaces`; the resolved `files` are absolute verified paths originating under the model's `kernels/` directory (the manifest itself uses relative paths). Use the declared implementation or reject an unsupported entry clearly. Do not silently call a rejected optimization Exact.

Successful load returns `load_s` (finite, non-negative seconds), `engine` (nonempty implementation name), optional `engine_reason`, and memory in the reply or a pushed status before it. Return only when inference can run. Failed load must leave a well-defined unloaded state and free partial allocations. Load-setting changes require supervisor reload; request settings apply to the next request. Arsenal owns persistence, red-row state and Reload, not the engine.

### Ping and status

`ping` returns an `ok:true` reply. `status` returns/pushes:

```json
{"memory":{"footprint_mb":120.5,"mlx_active_mb":80.0,"mlx_cache_mb":8.0}}
```

Numbers are current decimal MB (`bytes / 1,000,000`), finite and non-negative. Report real process memory, not a made-up zero or peak-as-current value. Non-MLX engines report both MLX fields as zero. The checker validates the envelope and bounds, not the accuracy of OS accounting. Push status after load, successful run and unload, and on explicit status requests; never sample continuously at idle. A model-quality/performance result may mark an unavailable metric `null`; a wire memory report is numeric.

Two optional fields let Activity say what the memory is for: `mlx_weights_mb`, MLX active memory measured once the load
has finished (the weights as MLX holds them), and `mlx_peak_mb`, MLX's peak since the current or last load or job began
(reset `Memory.peakMemory` when each begins; never send a lifetime peak). While a load or job runs a runner may push
`{"event":"status","live":true,"memory":{…}}` about once a second; live readings update Activity without making the menu
re-read its rows. The supervisor samples every runner's footprint twice a second during each job and reports the job's
highest as the model's peak, so a runner without these fields still shows its peak.

### Cancellation

`{"id":8,"op":"cancel","target":7}` acknowledges with its own terminal `{"id":8,"ok":true}`. Keep the reader responsive while inference runs. Stop the target at an engine-safe boundary, emit no further target events after its terminal reply, and finish it with `ok:false/error.code="cancelled"`. Check a cancellation flag between batches/tokens/steps; output writes must remain frame-atomic. Cancelling an already finished or unknown target is an idempotent successful acknowledgement. If inference finished before cancellation arrived, its original successful result is valid. A deterministic checker cancel fixture must be long enough to demonstrate interruption rather than that race.

### Unload and exit

`unload` cancels/drains active work, releases engine/Metal/MLX allocations, emits memory status and an `ok:true` reply, then closes stdout and exits **0**. It is not a reusable unloaded child. The supervisor waits a grace period, then sends SIGTERM and finally SIGKILL to the process group if necessary. Children and temp handles must not survive their runner. EOF follows the same cleanup, without inventing an unload request ID.

### Conversion (capability-dependent)

`convert {src_dir,dst_dir,dtype:"int8"|"int4",group_size:64,recipe}` runs in a fresh process without load and returns `files:[{path,bytes,sha256}]`. Source is a verified 16-bit checkpoint, never an int4→int8 requantization. Recipes are opaque folder strings interpreted by the engine. Preserve tokenizer/config/voice files so the output loads independently. Get owns snapshot/provenance manifests and source retention. A non-quantizing engine need not advertise convert. The wire checker does **not** certify numerical conversion or every recipe; test each declared derive through the model's Get/convert/load gate.

## Operations by kind

The folder's `[[op]]` capability list and its referenced JSON Schemas are authoritative for names, required fields, bounds and model-specific extensions. Kind schemas live in `kinds/<kind>/ops/`; model-local `ops/` overrides the shared declarations. The folder op list is authoritative: a shared kind operation is not automatically supported by every model. With no folder op list, only kind ops marked `default = true` are included. `input` is the schema path. `output` is `json|audio|image|text-stream|file`, not a schema path; optional `output_schema` validates the structured result without `id`, `ok`, `memory`. Use self-contained local `#/$defs/...` references, never remote references or code.

| Kind | Operations and wire result |
|---|---|
| voices | `speak`: text, voice, speed/language/instructions/reference where supported → PCM stream or audio file + duration/generation/first-audio/RTF figures. `voices` → `{voices:[{id,name,language,...}]}`. `design_voice {description,preview_text?,language?}` → voice_id + preview file; `clone_voice {audio:{path},transcript,name}` → voice_id where declared. Saved voices persist per model and appear in voices. Inline speak.reference is one-shot cloning; use model schemas for actual support. |
| sound | `generate`: prompt and generation controls → sound/music audio; use declared duration, seed and other fields. |
| enhancement | `enhance`: input audio path and declared controls → cleaned audio file/stream. |
| decisions | `decide`: state/questions → System One `{answers,model,usage}`. `judge`: items/questions → the declared batch result. Preserve typed yes/no, choice and score answers and Verdict/TypeSafe parity; arbitrary prose is not a valid replacement. |
| images | `generate`: prompt/width/height and declared seed/steps/guidance/n → progress events and `{images:[{path,width,height,bytes,sha256}],seed,seconds}`. |
| chat | `chat`: messages and declared sampling/tools/format settings → token events, then text/finish_reason/usage/TTFT/tokens-per-second. `count_tokens`: messages or text → count. |

Providers' HTTP dialects are mapped outside the runner. A new kind needs a kind definition, schemas and runner ops; the universal HTTP endpoint uses those declarations without a new hard-coded runner transport. A dialect module is only needed for an established provider shape.

### Audio and files

Before binary audio, emit `{"id":7,"event":"audio_begin","stream":1,"sample_rate":24000,"channels":1,"encoding":"pcm_s16le"}`. Send kind-2 chunks for that stream, aligned to complete samples across all channels. PCM is signed 16-bit little-endian, not WAV headers or MP3. Zero-length chunks do not establish useful audio output. Text streams use `token` events. Image progress uses `progress` with step/total.

Write large results below the request's absolute `out_dir`, a supervisor-created job directory inside `<support>/jobs/`. Do not follow symlinks, traverse paths or overwrite files outside that job. Finish/close files before replying; report absolute `path`, exact `bytes`, lowercase SHA-256. Example: `{ "path":"/support/jobs/job/audio.wav", "bytes":48044, "sha256":"<64 hex>" }`. Nested image/audio lists use the same receipt fields. The supervisor validates files before serving them. The checker re-hashes regular files and rejects escapes, symlinks and byte mismatches. Job cleanup/retention is supervisor policy, not permission for runner deletion of unrelated files.

## Checker fixtures and coverage

Recommended `gate/runner-check.json`:

```json
{
  "load": {"dtype":"fp16","assets":{},"config":{"settings":{"scale":2},"kernels":[]}},
  "ops": {"evaluate":{"values":[1,2,3],"iterations":1}},
  "cancel": {"op":"evaluate","params":{"values":[1,2,3],"iterations":1000000000}}
}
```

The flat form `{"evaluate":{"values":[1,2,3]}}` remains valid. Samples contain params only, no `id`/`op`. `load` supplies checker overrides for dtype/assets/config (not arbitrary model/weight paths); without an explicit `config.kernels` override, the checker resolves the current chip/mode from the folder manifest; `cancel` names an existing op and schema-valid params. Use absolute existing paths for real assets/references. Fixtures are inputs, not evidence and not execution authority.

The checker reads declarations directly; it does not validate the whole production model folder or its weight hashes. `Arsenal catalog validate` does that. For omitted samples it synthesizes only simple schema-valid inputs from examples/defaults/constants/enums, required properties and bounded arrays/scalars. Patterns or complex alternatives without explicit samples cause a failure, not a pretend pass. Supported validation vocabulary is the shared folder validator's checked draft-2020-12 subset; unsupported keywords and remote refs fail closed. `format` is an annotation. Arbitrary draft features are not silently supported.

Each run prints one `PASS` or `FAIL` line per check and a summary; nonzero exit means failure. Coverage:

- first-frame hello/version/PID/capabilities;
- load envelope, ping, numeric memory after status and successful ops;
- unknown-op error and a schema-invalid input for every declared op (a closed empty-object schema is tested with an unexpected property);
- every declared op with validated inputs, basic output transport and optional output schema; audio `stream` paths also get a separate binary-stream check when declared;
- cancel acknowledgement plus target terminal reply; explicit cancel fixtures must finish as cancelled;
- unload reply, EOF and process exit; fresh-process malformed probes for zero/oversized/truncated lengths, unknown frame kind, truncated binary header, invalid UTF-8/non-object JSON and invalid envelope;
- bounded pipe reads/writes and process-group cleanup on failure. Smoke-response retention is limited to 64 MiB or 100,000 events; this is a checker safety limit, not an inference-output limit. Child stderr is kept, never echoed as protocol output.

A streaming/file check is not a quality gate. The checker does not measure idle CPU, verify kernel arithmetic, certify performance, test every schema boundary or all conversion recipes, or establish official SDK/API compatibility. Those are separate tests. Run it on the actual model, mode, dtype and engine you are claiming, not merely on a toy with the same hello.

## Exact, Fast and acceptance

**Exact** means bit-for-bit output equality to the pinned reference runner on the same chip, dtype, seed, inputs and settings. Faster scheduling, allocation or kernels only qualify when those numbers remain unchanged. Preserve intermediate comparisons if they explain a failure. **Fast** can change numbers, but must pass the model's stated quality thresholds with the claimed speed benefit. Changing dtype is a separate experiment; do not call a quantization Exact relative to a different dtype.

Model gates freeze the reference version and inputs, compare required outputs, validate shapes/types/finite values, cancellation/files, and state modality-specific tolerances. Exact comparisons need actual numeric/byte evidence; Fast uses recorded quality limits (e.g. typed accuracy, intelligibility, reference deviation) and human review where automatic checks cannot judge quality. Missing reference/ASR/listening evidence stays missing. Kernels are selected by chip family × mode; both wildcard and specific entries need tested coverage where claimed.

Write results to `results/<chip>.json` using `Resources/schemas/results.schema.json`: model, kind, runner/worker hash, chip/OS/build, actual dtype/mode, suite ID/version/hash, profile, gate status, conditions, metrics and per-case measurements. Use the existing serialized keys required by the result schema; legacy key names are not UI/CLI terminology. Keep compatible baselines and attempts in separate linked evidence files rather than overwriting history. A quick gpulock proof records `measurement_lock=false` and contention, is never ranked and does not earn Measured. CPU toy/stub results are never model results. This v2 development build authorizes no controlled benchmark session unless separately approved.

Every attempt gets `log/NNN-slug.md` with portable TOML front matter:

```toml
+++
hypothesis = "Reuse scratch buffers without changing arithmetic."
change = "Moved allocation outside the generation loop."
measurement = "Reference gate passed; controlled suite unchanged."
verdict = "kept"
numbers = { before_s = 1.2, after_s = 0.8 }
links = ["results/m5.json", "gate/evidence/exact.json"]
+++
```

Then explain the command, source revisions, environment, evidence, failures and next useful step. Verdict is `kept|rejected|inconclusive`; numbers are actual, not illustrative promises. Preserve rejected attempts and their evidence. Existing log/result formats are specified in `CONTRIBUTING.md` and `BENCHMARKS.md`.

A runner PR includes source/licences, pinned dependency locks or signed artifact integration, checker output, the numerical gate, compatible performance evidence and the log. Review rejects claims without that evidence. The fastest **accepted** compatible runner that passes the gate becomes `[selection].default` or its chip-family entry in the next app release. Selection is deterministic: no on-device racing or auto-benchmarking. Faster but failing stays a rejected log entry; unmeasured work can remain Unverified without taking a measured default. See `CONTRIBUTING.md` for the closing checklist.

## Independent toy implementation (agent acceptance task)

Build a native runner from this document alone. It supports kind `toy`, ops `load,ping,status,cancel,unload,evaluate`, and no convert. `evaluate` returns `result = sum(values) * scale`. `scale` is `load.config.settings.scale` (default 1); `iterations` repeats arithmetic only to make cancellation observable. Both Fast and Exact use the same arithmetic. Its schema is:

```json
{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","required":["values"],"additionalProperties":false,"properties":{"values":{"type":"array","minItems":1,"maxItems":64,"items":{"type":"number","minimum":-1000000,"maximum":1000000}},"iterations":{"type":"integer","minimum":1,"maximum":1000000000,"default":1}}}
```

For this protocol-only exercise create your own `<scratch>/toy/ops/evaluate.input.json`, `gate/runner-check.json` using the structured fixture above, and this abbreviated **checker-only** descriptor:

```toml
kind = "toy"
[weights]
dtype = "fp16"
[[op]]
name = "evaluate"
input = "ops/evaluate.input.json"
output = "json"
stream = false
```

This abbreviated folder is not catalog-valid and is not a product model. Use finite numbers, reject booleans as numeric input, and reject unknown fields. Keep stdin responsive during the billion-iteration cancel sample; a separate thread plus a cancellation flag is sufficient. Obtain current process memory (Mach task_info or an on-request `/bin/ps` RSS reading), with MLX fields zero. Load needs `load_s`, `engine` and memory; valid evaluate returns `result` and a memory update. With scale 2 and values `[1,2,3]`, the result must be 12. On invalid framing log to stderr and exit, never print diagnostics to stdout. Run the checker in both modes and retain the implementation and output. Do not read the bundled templates for this exercise.

## Shipping a native engine

Schema-3 local folders support agent-built Swift executables without an app release. Choose the runtime by the **Introduce a new architecture** procedure in `arsenal://integration/ADDING.md`: bundled native worker first, then an existing Swift MLX implementation, then our own Swift MLX port, then a native Mac engine wrapped as-is. Python never serves a model; it is the reference an MLX port is checked against. Custom kernels and tuning require the owner's per-model go after the precision first pass. Arsenal never compiles source or resolves Swift packages. Use the build procedure below once that implementation is authorized.

For a local Swift runner:

1. Put reviewable source in `runners/swift/Package.swift` and `Sources/`. Pin dependencies by immutable revision. When using MLX, match the native worker: mlx-swift `901941965d82e4a216d4d117231d847d194c563d`, mlx-swift-lm `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` (and swift-transformers `150169bfba0889c229a2ce7494cf8949f18e6906` if used). Reuse the existing SwiftPM caches. A Foundation-only runner needs no MLX dependency.
2. Build arm64/macOS 26 with `/Library/Developer/CommandLineTools/usr/bin/swift`, `DEVELOPER_DIR=/Library/Developer/CommandLineTools`, `--build-system native`, and an isolated scratch path outside the model folder. Follow the Swift 6.4 weak `swift_initBorrow` link fix in `scripts/build.sh`: compile the weak-import C reference with the same toolchain and link its object with `-weak_reference_mismatches weak`. Preserve the native worker's Metal resource build when MLX needs it. `Tests/Fixtures/DropIn/fixture-layers/runners/swift/build.sh` is the deterministic fixture's concrete build recipe.
3. For MLX, copy the matching `mlx-swift_Cmlx.bundle` beside the executable in `runners/swift/bin/`, including the `default.metallib` built with the pinned MLX source and the native worker’s no-fast-math recipe. Copy any other `Bundle.module` resources beside it too; Arsenal does not inject its own resource directory. Test from another working directory to prove resource discovery.
4. Copy the final executable to `runners/swift/bin/<name>`, ad-hoc sign (`codesign --force --sign - --timestamp=none`), and verify its arm64 architecture/signature. Agent-built local files normally carry no quarantine. Inspect attributes and provenance before approving; rebuild a quarantined executable locally rather than weakening app library validation or Gatekeeper. Test launch from the hardened app in an isolated profile and record the result.
5. Retain `runners/swift/build-receipt.json`: `command` and `toolchain` strings plus `sha256`, an object mapping every file under `runners/swift/` (except this receipt and `build.sh`) to its SHA-256. Include Package.swift, Package.resolved when used, Sources, executable, and every runtime resource. Create it after signing the final executable; stale or incomplete receipts fail admission. This binds approval to the reviewed inventory; the integrating agent must still verify the recorded build actually produced it.
6. Declare `language = "swift"`, `entry = "runners/swift/bin/<name>"`, `protocol = 1` and normal runner selection/chip/mode fields. Implement this same ArsenalWire protocol. Run the checker on the built entry, then the real Get/Load/operation/cancel path. Review source and exact binary digest, gate against the independent reference, and approve the final folder content. Any subsequent source or binary change requires Refresh and new approval.

The executable is approved code, not a sandboxed plugin. It must be self-contained or ship reviewed runtime resources inside the folder; the parent app's hardened runtime does not grant a child permission to load unreviewed libraries into the app. Schema-1/2 and bundled `worker` lookups keep their existing release contract.

To ship an engine, submit a PR with its source and pinned dependencies, a reproducible build step in `scripts/build.sh`, and packaging in `scripts/assemble.sh`. The manifest's `worker = "<name>"` resolves exactly to `Contents/Helpers/arsenal-worker-<name>`; language does not change that lookup. For a family added to the existing combined native worker, add its dispatcher and the assembly family symlink. For a standalone Rust/C/etc. executable, copy that binary and required runtime resources into Helpers in assemble, add them to the strict bundle/inventory checks, and include them in the inside-out signing pass of build.sh. The final app is signed only after its helpers and libraries. Preserve third-party licences and provenance. Submit Get → Load → declared operations → cancel → Unload evidence from the packaged app, not just the standalone checker. Shared runtimes have one copy, and an idle engine blocks on stdin with no helpers left after unload.

App downgrades with a newer authenticated cached catalog are not a supported catalog rollback: the new catalog may require instruction files absent from the older bundle. Keep the newest compatible app, or explicitly choose a compatible developer source.

Voice design previews are transient, not anonymous saved voices. `design_voice` without `name` returns a preview token; supplying `name` explicitly saves a durable design. Every preview returns `design: {seed, model_revision, dtype, fast, preview_text, language}` alongside its audio. If no seed was requested, the design runner chooses one before generating audio. Save retains this recipe, the description, and a durable copy of the chosen preview clip. ElevenLabs `create-previews` → `text-to-voice` saves that selected preview in its original design model (no cloning model is required), not a fresh draw from its description. The service checks the token is less than one hour old and its audio still exists; a legacy token without reproducible metadata must be redesigned.

Speaking a saved design always restores its description and seed, even if a caller also supplies instructions or a seed: these identify the saved voice. Language defaults to the preview language. On the same model revision, dtype, Fast/Exact mode, text and language, the seeded design reproduces its preview; different text is a new utterance with that fixed design recipe, not a guarantee of identical timbre for arbitrary text. Revision changes fail plainly rather than silently changing the voice. dtype/mode are recorded provenance, not forced reload settings; changing them can change the audio. Preview audio is retained evidence, not an unsupported cloning reference. `delete_voice` is a service operation synthesized only for folders with design/cloning capability; it deletes saved data without loading or running an engine.

Schema admission, the runner checker and live validation share the same bounded subset and work budget. Local refs are memoized for admission; recursive refs and unsupported assertions fail closed. Request-setting merges are limited to declared properties and skip a caller-owned top-level group.


Progress frames keep the existing phases and may add `label` and `unit` (each at most 128 characters), alongside step/total. Unknown totals are omitted. HTTP `Accept: text/event-stream` on a streaming operation yields progress then one result/error event; files are retrieved through result URLs. Synchronous disconnect cancellation remains distinct from durable MCP jobs, which require explicit cancellation.

For file operations, `Accept: application/json` returns the full validated payload with files[].url/mime/bytes/sha256. The default single-file bytes/multipart behavior remains. URLs are retained for one hour, with opportunistic expiry at startup/new service jobs; expiry uses no background timer. MCP returns the same fields, resource links, bounded thumbnails and a short status line. Unknown MIME types download as files.

Refresh stages metadata while resident requests retain their old schemas/settings. Approval of a changed loaded implementation requires current protocol/parity gates and a passed quick speed check. It automatically swaps at the next idle window with no running/queued request, queues arrivals while swapping, and loads the new worker in place. Failure restores the previous approved worker and reports the error. HTTP `POST /arsenal/v1/models/<id>/rollback` with `{}` or MCP settings action rollback provides one-step rollback. Row/activity `revision` exposes active/pending/previous digests and swapping/error. Evidence-only edits need approval but do not reload inference code.

The checker accepts embedded kinds, output schemas and structured fixture `expected` entries such as `"separate":{"files":4,"ids":["layer-0","layer-1","layer-2","layer-3"]}`. It checks exact count/order before accepting output; its protocol report never asserts independent numerical parity. Use `--results` inside retained evidence and run the approved native executable for qualification. Short GPU smokes follow current task authority and run directly; long GPU work/downloads require the agreed resource window.
