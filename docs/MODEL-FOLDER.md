<!-- SPDX-License-Identifier: MIT -->
# Model folders

A folder is the complete, inspectable recipe for one model: pinned vendor weights, accepted runner defaults, settings, operations, kernels, quality gate, and the history of every optimisation attempt. See `RUNNERS.md` for the process protocol and `CONTRIBUTING.md` for log headers. Schemas 2 and 3 are additive; schema 1 folders remain readable. Schema 3 covers local drop-ins and embedded kinds below. Files are instructions, never bundled weights.

`good_for = ["..."]` and `limits = ["..."]` are optional lists of up to 32 nonempty entries (1,000 characters each). They supplement `purpose`, `languages` and `[licence]` in generated `MODELS.md`; they do not change operation validation. Regenerate it after folder/evidence changes. New native integrations use [INTEGRATING.md](INTEGRATING.md).

## Layout

```
models/<id>/model.toml
            settings.toml             # optional settings panel (see below)
            README.md
            runners/<name>/…
            ops/<op>.input.json       # model-local JSON Schema
            ops/<op>.output.json      # optional structured result schema
            kernels/manifest.toml     # optional; files relative to kernels/
            kernels/*.metal
            gate/gate.toml
            gate/runner-check.json    # optional {op: schema-valid params}
            bench/bench.toml
            log/001-port.md           # at least one checked research log
            results/<chip>.json       # optional measured evidence
```

IDs use `[a-z0-9][a-z0-9.-]*`; the folder name equals the ID. File references are exact relative paths: no globs, absolute paths, traversal, symlinks or special files. Revisions are 40-hex commits; SHA256s are 64-hex, not Git blob SHA1. TOML supports scalar strings/numbers/bools, arrays, tables, inline tables, arrays of tables and comments, not date values or multiline strings.

## Shipped inventory

`CatalogShipping.files` selects the signed model-file list used by both Get and `scripts/bundle-inventory.py`. It retains the manifest, README, optional settings panel, declared gate/bench recipes (required by manifest admission), runner-check fixtures, runner source closures (their Tests and fixture trees only when a folder-local Swift entry needs the whole closure), operation schemas, declared kernel manifest/sources/headers, research logs used by catalog history, and top-level JSON/Markdown table results. Embedded kinds and explicit runner/schema file references are retained too. Shared kinds have their own signed inventory. The bundle still contains exactly the signed files; packaging has no separate exclusion filter.

Intake captures under `gate/evidence/` and `results/intake/`, runner test and fixture trees, quality bindings, gate probes/parity files, benchmark scripts, undeclared kernel probes and local measurement overlays remain local in `lab/`. `catalog validate` checks tracked bindings and table results, and checks deeper evidence when present locally. Historical runs and raw captures live in ignored `lab/evidence/<model>/`; current bindings and top-level table results stay tracked. The runtime does not read quality bindings or raw capture results: top-level table results already carry the signed quality verdict and identity; `PrecisionEvidence` checks that identity against the current recipe, runner implementation, kernels and load settings for both catalog-only and installed rows. No reduced capture format or replacement quality identity is needed.

## Worked example: Qwen3.8 27B

The committed folder `models/qwen3.8-27b` is the complete example, including all immutable pins. These are its v2 declarations (existing licence, weights, runners and gate/bench remain):

```toml
schema = 2
suggested = true

[facets]
modalities_in = ["text"]
modalities_out = ["text"]

[selection]
default = "native"
chips = { m5 = "native" }

[[setting]]
id = "context_limit"
title = "Context"
type = "choice"
default = 262144
choices = [262144]
applies = "load"
maps_to = "config.context_limit"
help = "Model maximum; memory grows with use."

[[setting]]
id = "thinking"
title = "Thinking"
type = "toggle"
default = false
applies = "request"
maps_to = "request.thinking"

[[op]]
name = "chat"
title = "Chat"
input = "ops/chat.input.json"
output = "text-stream"
stream = true
```

`context` / `native_context` declare model context; chat resolves the maximum from installed `max_position_embeddings` (nested text config first), with the manifest maximum before Get. The table, context setting and Load use that same maximum, not a fixed cap. Supported context is not measured performance at that limit. Precision is a separate user control. Fast/Exact is an automatic runner policy: exact kernels reproduce reference numbers bit-for-bit; fast kernels pass the model's quality gate. Do not claim a kernel passed without evidence.

## Settings and operations

`[[setting]]` has `id`, `title`, `type`, scalar `default`, `applies` and optional `help`. Only a request-time `number` setting may leave out `default`: nothing is then sent until a value is pinned, so the runner picks (a seed: a new random one each request), and clearing the value returns there. A load setting always has a default, because it always reaches the runner's config. Types: `toggle` (bool), `choice` (unique scalar `choices`, including the default), `slider` (number with `min/max/step`), `number` (optional bounds/step), `text`, `voice` (strings; live choices come from the `voices` operation). IDs use lowercase letters, digits and underscores, starting with a letter. Defaults must satisfy the declaration.

Load settings arrive in `load.config.settings` keyed by setting ID. Optional `maps_to = "config.<key>"` copies the value into a legacy load key too. They become pending until reload when a model is loaded. Request settings require `maps_to = "request.<param>"`, apply to the next request and never require reload. Explicit request parameters win over setting defaults. Optional `excludes = ["<param>", …]` leaves a request setting out whenever the caller sends one of those parameters (FLUX's width and height step aside for a caller's `resolution`).

A retired setting keeps working through `[[setting_alias]]`: `id` names the old key and `values` maps each of its values to the current settings it stands for, e.g. `values = { "512x512" = { width = 512, height = 512 } }`. A settings write of the alias (HTTP, MCP) sets those settings, and a value saved under it moves to them when they hold none. Every target must be a declared setting that accepts the value. The service owns persistence and pending state; folders only declare the contract.

`[[op]]` has `name`, `title`, `input`, `output` and `stream`. `input` names a draft 2020-12 JSON Schema object; `output` is the transport (`json|audio|image|text-stream|file`), not a schema filename. Optional `output_schema` names the structured result schema. Use self-contained local schemas or an explicit `kinds/<kind>/ops/<file>` reference; references inside schemas must resolve within that JSON document (`#/$defs/…`), never remote. A declared model `[[op]]` list is authoritative (including an explicit empty `op = []`); it does not inherit additional kind ops. Without model ops, only kind ops marked `default = true` are exposed. A voice-design-only model must never accidentally advertise cloning or preset speech. Kind ops provide shared schema/title definitions, not permission to claim capability. Unsupported validation keywords and recursive schema references fail rather than being silently ignored; schemas/references have a 12-level structure limit and evaluation a 24-level limit; `format` is an annotation, not an assertion. Use `additionalProperties = false` in JSON to reject unknown input fields where appropriate.

Optional `gate/runner-check.json` supplies `{ "op-name": { "input": "…" } }` samples. Alternatively use `{ "load": { "config": {}, "assets": {}, "dtype": "bf16" }, "ops": { "op-name": {} }, "cancel": { "op": "op-name", "params": {} } }`. Every flat/ops key and cancel op must resolve to a resolved declaration, and their params must satisfy its schema. `load` is a checker-only override, not an operation input. These are checker inputs, not permission to run/download weights.

## Settings panel

The Models menu shows only a model's load settings: its gear opens a small settings card beside its table row, holding Arsenal's own rows (Keep loaded; Show in menu for a model with tools; Runner only when two or more runners can run it) and then the model's `applies = "load"` settings, the ones that change how Arsenal hosts it. `settings.toml` beside `model.toml` lays out those load settings: their order, labels, units and controls (a choice is always drawn as segments there); without it the card lists them in declaration order. Request settings (`applies = "request"`: seed, steps, shape, style, thinking, system prompt, voice and the like) belong to the tools and the API: the app or tool making the request sets them (Make an Image, Chat), and the API and MCP accept them with every request. Their rows in `settings.toml` stay valid and complete the layout of everything the runner accepts, but the Models menu does not show them. The file arranges and presets settings; it declares none. `[[setting]]` in `model.toml` stays the contract the runner accepts.

Start from your kind's reference: copy `kinds/<kind>/settings.toml` into the folder, copy the `[[setting]]` blocks your runner really accepts from `kinds/<kind>/reference.toml` into `model.toml` (with this model's defaults, bounds and help), then tailor the panel: remove rows for settings you did not declare, set presets to the values this model is tuned for. References exist for images (free width and height; `kinds/images/resolution/` for one size choice), chat, decisions, voices (preset voices; `kinds/voices/design/` for a model that designs or clones voices and takes a language), sound, enhancement and video. Presets follow the model's own tuning: Final is the step count it is tuned for and Draft fewer, and a distilled model gets no Quality row. Duplicating a reference into each folder is intended.

```toml
schema = 1
hidden = []                       # declared settings deliberately left out of the panel

[[section]]                       # sections are separated by a hairline; no headings

[[section.row]]
label = "Shape"                   # a preset row: segments that set several values at once
presets = [
  { title = "Square", icon = "square", values = { width = 1024, height = 1024 } },
  { title = "Portrait", icon = "rectangle.portrait", values = { width = 832, height = 1216 } },
]

[[section.row]]
setting = "seed"                  # a setting row; a seed declares no default, so it reads Random until pinned
randomize = true                  # a dice button pins a random value within min..max

[[section]]
title = "More"                    # only a collapsed section has a title: its disclosure row
collapsed = true

[[section.row]]
setting = "width"
unit = "px"

[[section.row]]
setting = "height"
unit = "px"
```

A setting row takes `setting` and optionally `label` (overrides the title), `control`, `labels` (value → name), `icons` (value → SF Symbol, drawn on segments), `unit` (number and slider settings), `randomize` and `show_when`. Controls by type: toggle `switch|segments`, choice `segments|menu`, slider `slider|field`, number `field` (or `slider` with min and max), text `field|editor|file` (`editor` is a multi-line editor where a tool draws one; `file` adds a glyph that picks a file's path), voice `voice`; omitted, the type's default applies. Number fields step with the arrow keys; a choice with one value shows as plain text. Value keys are written as the value: `true`, `1024`, `0.5`, `english`. A preset row takes `label`, `presets` (2–6 entries of `title`, optional `icon`, `values`) and optional `show_when`; every preset sets the same settings, every icon or none. `show_when = { setting = value }` or `{ setting = [values] }` shows a section or row only while those values hold.

`settings.toml` is part of the folder's content digest (consent, revisions) but not of the implementation fingerprint, so a layout-only change never marks the runner's evidence stale. Validation runs with the rest of the folder and names each problem by place (`settings.toml: section 2, row 1: guidance is not a setting declared in model.toml`). Every declared setting has exactly one row or is listed in `hidden`; presets and conditions use only values the declarations accept; a setting a preset sets also has its own row (or is hidden), so a value that matches no preset is visible: its collapsed section opens by itself. Icon names that SF Symbols lacks fall back to the title.

## Runner and kernel selection

Schema 2 requires `[selection] default = "<runner>"` with optional `chips = { m5 = "<runner>" }`. Chip keys are family names (`m1`, `m2`, …), not marketing suffixes. The exact chip-family entry wins, otherwise default. No runner races, local ranking or choice cache. The accepted default must be installed and compatible; incompatibility is an error, not a silent alternate.

A Swift runner without an entry references a bundled family `worker`; schema 3 also supports the folder-local Swift entry described below. Python model runners are unsupported; Python serves only as an independent reference or test tool. Other native engines use a packaged worker under the same protocol; language is descriptive, never a command to execute. Add the engine through the artifact resolver before shipping it. Get never compiles source.

`kernels/manifest.toml`:

```toml
[[kernel]]
id = "attention"
chip = "m5"                  # or "*"
mode = "exact"               # exact | fast
files = ["attention.metal"]  # relative to kernels/
entry = "attention"
replaces = "attention"
api = "mlx"                  # default; or metal (full MSL library)
inputs = ["x", "w"]          # required for mlx
outputs = ["out"]            # required for mlx
# header = "header.metal"     # optional, must be in files
```

Each `(id, chip, mode)` and `(replaces, chip, mode)` is unique. Resolve one kernel per replacement site: Exact picks exact@chip then exact@*; Fast picks fast@chip, fast@*, exact@chip, exact@*. Return chosen declarations in manifest order. Metal files must exist inside the folder. Runner load config receives the resolved declarations; kernel compilation/execution belongs to the engine. Gate and log evidence accompany every accepted optimisation.

## Weights, variants and assets

`[weights]` supplies `repo`, `revision`, `dtype`, exact `files`, `[weights.sha256]` and `[weights.bytes]`. Source dtype is bf16/fp16; fp32 requires `[derive] fp16 = "cast:<family>"` and is not offered. Derives (`int8`, `int4`) start from the verified 16-bit checkpoint; recipe strings are opaque to catalog/Get. `[vendor.int4]` instead declares an independently pinned vendor quantization (optional `format/group`), never both vendor and derive for the same dtype. Conversion output must load independently with all tokenizer/config/voice files; provenance records source pins.

`assets = ["<id>"]` selects `[[asset]]` declarations with repo, `repo_type = "model"|"dataset"`, revision, exact files, licence (SPDX string or the licence table), hashes and sizes. Assets reuse content across models; load supplies their directories. An asset with `into = "<dir>"` is instead converted by the model's own `cast:` base recipe into that directory of the base snapshot (serving half format, packed with the derived tiers); it is staged only during that conversion, keeps no snapshot of its own, and Load reads `weights/<dir>`. HF tokens never reach runners. Source-retention policy is CONTRACTS §14; do not remove sources without its verified derive and receipt.

## Browser metadata and kinds

`suggested = true` is our curated flag, not a vendor claim. Optional `[popularity] hf_downloads, hf_likes, as_of` is a dated HF snapshot written by pin; the runtime never refreshes it. `pin <id> --metadata-only` updates just this snapshot at the already declared revision, leaving existing weight pins unchanged. `[facets] modalities_in/modalities_out` are nonempty unique arrays drawn from text/audio/image/video. Kind, tasks, languages, licence, publisher, parameter label and offered dtypes already come from the manifest: do not duplicate them in facets. `purpose` is what the model does in a few plain words ("Image generation", "Background removal", "Speech, voice cloning"; one line, at most 40 characters), shown under its name in the Models table and Get more models; without it Arsenal derives one from the kind and the declared operations.

`kinds/<kind>.toml` declares existing `id/title/order/suite/columns/api`, plus optional `icon` (SF Symbol), `title_singular`, `[facets]` and standard `[[op]]` declarations. Kind-local `input = "ops/<file>.json"` resolves inside `kinds/<kind>/`. Model-local ops resolve inside the model. Adding a new kind needs worker operations and any desired dialect module; universal ops work from schemas without a dialect.

## Catalog and local folders

Non-chat folders under `models/` ship, regardless of suggestion. `catalog/lineup.toml` is historical and not consulted by v2 build. After committing folder changes, `Arsenal catalog build` validates the complete repository folders, then checks every repository byte against HEAD and creates a monotonic sequence, source commit and runtime folder inventory (`CatalogShipping.files`). Intake captures, quality bindings and other gate/benchmark evidence stay tracked in the repository; Get and bundles contain the same signed runtime files. `sign --key <path>` (or `ARSENAL_CATALOG_SIGNING_KEY`) signs with the release Ed25519 key kept outside Git; the file must exist. Without a key option, `sign` uses the profile's development key. A key that does not match `Resources/catalog-public-key` is refused unless `--rotate` replaces the committed public key. `verify` checks the exact bytes. Never commit or print the private key. `scripts/catalog-refresh.sh` validates all folders and rebuilds/signs; `--pin` additionally refreshes pins (commit those before building).

Bundles put all listed model files in `Contents/Resources/models/`, shared kind files in `Contents/Resources/kinds/`, with catalog and public key in Resources. The signed `kind_files` inventory lists every file under kinds/ by SHA256 and size. The loader/installer receives that verified kinds root explicitly; `kinds/<kind>/ops/...` resolves there even while a model is in a staging directory. Dev uses repo kinds/ only after verifying it against the same catalog. There are no duplicated shared schema files in model folders. The bundle source validates paths and the installer checks file inventories, byte sizes, SHA256s and manifest identity before activation. It needs no source repository or network. LocalGit remains a development source. No Get may fetch model instructions from HF; only pinned weights/assets.

Out-of-band folders live at `<support>/local-models/<id>/`, validate with the same rules, and carry a Local badge rather than signed trust. Local IDs may not shadow signed model IDs: a local folder named like a signed model is ignored and left on disk, and the signed model takes its place (as when an add-on joins the catalog). Code outside bundled workers (folder-local Swift entry or custom engine) needs an explicit once-per-folder-content consent before first execution. Validation is not consent. The service owns this consent seam.

## Scaffold from Hugging Face

`Arsenal catalog new --from-hf hexgrad/Kokoro-82M --kind voices` reads only HF metadata and writes a new folder (never runs repo code, never downloads weights). It records the immutable revision, file sizes and LFS SHA256s, popularity and reported licence. Git-file SHA256s are unavailable in metadata; they remain unpinned until `pin`. The tool lists the remaining human work: reviewed runner, dtype/file choice, licence/commercial check, settings/ops/gate/bench and log evidence. A scaffold is intentionally not a release-ready model. Its `scaffold = true` flag prevents catalog validation or execution; remove it only after completing the listed checks. It never overwrites an existing folder.

## Format variants (v2.2)

An arbitrary safe label can identify a dtype/quantization variant. Keep its pins in `[vendor.<id>]` or its opaque recipe in `[derive]`. Declare interpretation separately:

Breeze's existing folder already pins the vendor MXFP4 checkpoint under `[vendor.int4]` (`mlx-community/Breeze-TTS-2-mlx-4bit`, revision `3a06d26b172ea4ae1da2f42d708383e9c79d5526`, group 32). To declare that representation explicitly, add:

```toml
[[format]]
id = "int4"                         # existing variant key; need not equal its storage name
storage = "mxfp4"
bits_per_weight = 4
source = "vendor"                    # uses the existing [vendor.int4] pins, not [derive]
runners = []                         # visible/filterable; no compatible Arsenal runner yet
```

Its effective checkpoint size is the sum of the existing vendor file pins. A new variant can instead use an arbitrary safe key (`mxfp4`, `oQ4e`, `exl3-3.25`…), with corresponding `[vendor.<key>]` pins. Optional `size_bytes` overrides the effective size; `ram_mb` supplies a RAM estimate. `runners` names compatible declarations in `[[runner]]`; an empty list means “No runner in this Arsenal yet.” Such formats remain browsable but Get/Load refuse them. Unsupported-format API requests return non-retryable HTTP 422 with `code: "unsupported_format"`. Declaring a format does not add engine support; the example does not modify the shipped Breeze folder.

`source` is `vendor` (download its pinned checkpoint), `derived` (run the named `[derive]` recipe from base weights) or `runner-native` (download unchanged base weights; the declared runner interprets this variant at Load). `storage` names the representation, while optional `method` (e.g. `dwq`, `awq`, `oq`) names how it was made. Optional `profile` is an inline table, e.g. `{ group_size = 64, default_bits = 4, module_bits = { lm_head = 8 } }`. oQ profiles are mixed-bit affine, not a new storage format. `bits_per_weight` summarizes the profile; it may be fractional.

Sizes in the browser are **effective checkpoint** sizes for each dtype: exact vendor pins or `size_bytes`, otherwise a labelled bit-ratio estimate for derived formats. RAM uses dtype-matched evidence, `ram_mb`, or checkpoint bytes plus runner overhead. Actual Get network/local-copy bytes are separate: deriving 4-bit may first need its 16-bit source. All selected dtype, size and RAM predicates must match the **same** variant; the catalog response returns `matchingDtype` and projects that variant's numbers onto its row. Existing v1/v2 folders need no changes.

## Loaded menu sections

```toml
[[menu.section]]
title = "Reader"
items = [{ id = "speak-clipboard", title = "Speak Clipboard", op = "speak", input = "clipboard", params = { voice = "af_heart" } }]
```

Items are declared operations, never shell commands. `input` is `clipboard` or `none`. The section appears only while this model is loaded and disappears on Unload; legacy `[[menu]]` actions still work. Loaded menu/operation declarations retain their verified snapshot through Refresh until explicit reload.

## Refresh and dependency plans

After editing `config.json` or dropping a valid folder into `<support>/local-models/`, click Refresh (the arrow at the top right of the Models table), `arsenal_settings(action:"refresh")` through MCP, or `POST /arsenal/v1/refresh`. It re-reads files on demand without restarting or interrupting resident requests. While the app runs it also watches `local-models/` (one FSEvents stream, no timer, nothing at rest), so a folder dropped in or removed shows in an open table within a few seconds without a click. Signed catalogs retain signature/sequence checks; local folders remain unsigned and require existing exact-content code consent.

`GET /arsenal/v1/models/<id>/get-plan?dtype=<label>` returns the weights, local imports, assets and native engine needed by Get. Plans inspect metadata and presence without downloading or hashing large weights. Native engines use a bundled worker or an approved folder-local Swift executable; Get never fetches or compiles native code.

## Schema 3 local kinds

Personal integrations use `<support>/local-models/<id>` and `schema = 3`. `kind_file = "kinds/local.<name>.toml"` embeds a `local.*` descriptor and its `kinds/local.<name>/ops/*.json` schemas in the immutable approved folder. Embedded kinds cannot override signed IDs, except that a folder may carry an identical copy of a signed kind (same descriptor apart from its bench suite, byte-identical operation schemas), as one made before its kind was signed does; folders sharing a local kind must have identical kind-tree hashes. With no kind_file, resolution uses the verified signed kind inventory. All resolved operations need both input and output schemas. The kind's `suite` may be absent; `primary_operation` chooses the generic tool's initial operation. Optional `comparison_metric` and `comparison_direction = "higher" | "lower"` describe qualified comparison semantics. Without evidence, the accepted manifest default remains selected.

`tools = ["operation"]` selects the generic native tool; `tools = ["background"]` places the model in Remove Background (one window for every such model; it offers each operation that takes one image and returns files, `remove_background` first, and shows every other input its schema declares). Schema 3 infers it when tools is absent; `tools = []` explicitly hides tools. With `operation`, the capability windows place the model by its declared operations (transcribe, separate, search, translate, document, extract, select, find, depth, upscale, smooth, make3d); a folder may also name one of those windows in `tools`, and must then declare at least one operation that window offers. Existing schemas keep their behavior. `[[op]]` supports `timeout_s` and `first_response_timeout_s` (1–3600 seconds), and an optional `[op.annotations]` object. Input/output use the bounded draft-2020-12 subset with local-only references; schemas are at most 1 MiB.

String schemas can add `"x-arsenal-maxUTF8Bytes":1024` to bound encoded UTF-8 bytes independently of `maxLength` (Unicode scalar count). Admission validates this nonnegative integer and rejects oversized values before worker dispatch. Generic forms hide properties with `const` and send their fixed values automatically.

Annotate a local file object with `"x-arsenal-file":{"accept":["image/png","image/jpeg"]}`. The object carries `path`; nested object/array positions are checked by the service against inbox/jobs containment and actual image MIME. Use MCP upload to obtain the path. Remote URLs and symlinks are rejected. Legacy unannotated input contracts retain their behavior.

Schema-3 file output is `{files:[{id,path,mime,label,order,width,height,...}],metadata:{...}}`; the kind schemas allow an optional display `label` and `order` on every file item, and a model-local schema keeps them when its adapter sets them. Output schemas describe the full payload with only wire `id` and `ok` removed. Every file needs a unique ID and actual MIME; images need matching width/height. The service verifies regular files inside this request's job, limits aggregate output to 64 MiB, and adds calculated bytes and sha256. Model-specific fields survive transport. The runner and its gate assert cross-request counts (e.g. requested layers equals returned file count).

Results can contain `checks`: protocol/parity/task/speed statuses (`pass`, `fail`, `not-run`), reference identity, implementation_fingerprint, folder_digest, fixture_identity, case counts and retained evidence paths with evidence_hashes. Reference identity records independent=true only after the agent has reviewed the independently invoked path. Inference fingerprints cover code/helper/kernel bytes, locks, schemas and inference-affecting manifest fields, excluding results/logs; fixtures have a separate fingerprint. Evidence becomes Stale when those identities or retained files differ. Legacy evidence acquires no new parity claim. Quick timings remain uncontrolled and never establish a dtype ranking.

## Folder-local Swift runners

Keep SwiftPM build/test scratch directories outside the model folder. Dependency checkouts can contain symlinks that the model-folder safety check rejects.

For schema 3, a local folder may select `language = "swift"`, `entry = "runners/swift/bin/<name>"`, with its reviewable package at `runners/swift/Package.swift` and `runners/swift/Sources/`. `worker` is optional for this entry path. The agent builds the arm64 executable locally before validation/Refresh; Arsenal only verifies the immutable approved folder and launches that exact binary. Source, dependency locks, resources and executable bytes all participate in the content digest and inference fingerprint. Build trees belong outside the model folder. Missing executables, source, unsafe paths and symlinks fail validation.

Keep the binary and every runtime resource it uses inside the folder. Use explicit request paths or the executable's immutable directory, rather than a mutable checkout. With no entry, Swift keeps the existing `worker` lookup in Contents/Helpers. The local build/signing procedure and app-compatible MLX pins are in RUNNERS.

## Per-precision stats (schema 2)

New table stats are saved as `results/<chip>-<dtype>-<runner>-<mode>.md` with `+++` TOML front matter. The header is `PrecisionStatsResult` (`schema = 2`, `model`, `mode`, `[stats]`), validated by `Resources/schemas/precision-stats.schema.json`. `[stats]` declares kind, actual dtype, runner, availability and every metric for its kind. Metric cells under `[stats.metrics.<StatMetric>]` have an explicit state: measured, quickCheck, declared, notMeasured, or unavailable with a reason. Numeric cells carry value and provenance (chip, ISO date, method, baseline and source). Units are fixed by the metric ID, including the battery denominator. Prose below the header preserves the method's details. `PrecisionStatsResult.markdown()` supplies the canonical writer.

Old schema-1 JSON remains readable. A quick check remains a quick check; task accuracy/WER does not stand for agreement with own 16-bit. Optional runner `features = ["batching", "sequentialOnly", "streaming"]` declares capabilities for Get More Models; omit unknown capabilities rather than invent them.

Local introduction measurements use `results/local/*.md` (schema 2, TOML front matter). This is the sole mutable result-data exception in a signed installed model folder: each extra file must parse, name that model/kind, and remain at most 1 MiB. Other extra files, links and code are refused. Current stats require provenance naming the current pinned weights; signed catalog results and code remain hash-verified. Native capabilities can be declared in runner features; bundled native family defaults expose batching or sequential-only/streaming as supported by the existing operation path.

A local folder whose own kind declares no table figures (a kit kind such as `local.document` or `local.forecast`) shows peak memory and one warm request's time as quick checks, without any file added to the folder (every byte of an installed folder is bound to its code approval). Arsenal ships them in `Resources/figures/folder-quick-checks.json` (schema `folder-figures-1`, written by `scripts/import-quick-checks.py`; the record stays in the repo's kit folder under `results/quick-checks/`), each entry bound to the model id, kind, weights revision, precision and runner; the folder's own `results/quick-checks/*.json` records (as `kit/new-model.py try` writes them) stand in, and per precision the newest wins. A folder kind titled like a group Arsenal's kinds already show joins that group (the kit `Documents` kind shows with TeleOCR); there its speed follows the group's unit, so a reader in Documents shows a triangle for speed rather than a request time.
