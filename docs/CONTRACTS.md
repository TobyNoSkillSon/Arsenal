# Arsenal contracts

Current folder and native runner contracts are in [Model folders](MODEL-FOLDER.md) and [Runners](RUNNERS.md). Product and verification rules are in [AGENTS.md](../AGENTS.md). The module contracts below describe the current service boundaries. Historical versioned briefs live locally in ignored `lab/archive/`.

## 15. Folder schema v2 (owner: schema)
Additive to §3/§13 (`schema = 2`; v1 folders stay valid):
```toml
suggested = true                       # curated by us; browser shows these first
[popularity]                           # snapshot written by `Arsenal catalog pin`, never fetched at runtime
hf_downloads = 123456
hf_likes = 789
as_of = "2026-10-03"
[facets]                               # browser filter data (anything derivable is derived, not typed)
modalities_in = ["text"]               # text | audio | image | video
modalities_out = ["audio"]
[[setting]]                            # per-model settings, rendered generically
id = "context_limit"
title = "Context"
type = "choice"                        # toggle | choice | slider | number | text | voice
default = 262144
choices = [262144]        # choice; slider/number use min/max/step; voice lists come from the model op `voices`
applies = "load"                       # load (row turns red + Reload until applied) | request (applies to the next request)
maps_to = "config.context_limit"       # load config key, or "request.<param>" default for requests
help = "Longest conversation the model keeps."   # tooltip, one sentence
[[op]]                                 # operations the model serves (kinds/<kind>.toml declares the standard ones; a model adds/overrides)
name = "speak"
title = "Speak"
input = "ops/speak.input.json"         # JSON Schema (draft 2020-12) files in the folder or in kinds/<kind>/ops/
output = "audio"                       # json | audio | image | text-stream | file
stream = true
[selection]                            # deterministic default runner (no on-device racing)
default = "native"                     # runner used when no chip entry matches
chips = { m5 = "native", m3 = "native" }   # chip family → runner; accepted from PR results
```
Kernels: `kernels/manifest.toml` lists `[[kernel]] id, chip = "m5"|"m4"|…|"*", mode = "exact"|"fast", files = [".metal" sources], entry, replaces = "<op or module id>"`. The runner receives the resolved list for (this chip, mode) in `load.config.kernels`. **Exact** kernels must reproduce the reference runner's outputs bit-for-bit on that chip (gate check); **Fast** kernels must pass the model's quality gate. History lives in `log/` (every attempt, kept or rejected, with numbers) and `results/<chip>.json`.
Local folders: `<support>/local-models/<id>/` — validated by the same schema, badge **Local**, unsigned; first run of non-bundled executable code (folder-local Swift executable or engine binary) asks once with a plain sentence.

## 16. Catalog v2 (owner: schema)
Every non-chat folder under `models/` is in the catalog (lineup.toml is replaced by `suggested` flags). `CatalogShipping.files` selects its runtime files for `Contents/Resources/models/` and the signed catalog: manifest, README, settings, admission recipes, runner-check fixtures, runner closures, schemas, declared kernels, catalog history and table stats. The signed catalog records `source_date` (git committer time of `source_commit`); declared figures such as context maximums are dated by it, and local folders by their last commit (model.toml's modification time without git). Intake captures, bindings and other development evidence remain tracked in the repository (see MODEL-FOLDER.md); the bundle source is the default (`catalog.source = bundle`; LocalGit stays for development). Get downloads weights only (Hugging Face; local import first). `Arsenal catalog new --from-hf <repo> [--kind K]` scaffolds a folder from a Hugging Face repo (metadata, pinned revision, files + LFS hashes from metadata, licence, a log entry stub, kind defaults) for the human/agent to finish.

## 17. Service v2 (owner: service)
`ModelsService` additions: `catalog(query: CatalogQuery) -> [CatalogEntry]` (all models with facets, sizes per dtype, suggested, popularity, kind, licence/commercial, languages, params, min RAM estimate, added/installed/loaded flags; query = filters + sort + text); `add(id)`, `remove(id, deleteWeights: Bool)`, table order persisted (`config.table`); `settings(id) -> [SettingState]` (declaration + current value + `pending` for load settings not yet applied), `setSetting(id, key, value)`, `applySettings(id)` (= reload); `ModelRow.needsReload: Bool`; `run(id, op, params, sink)` validates params against the op's input schema. Runner choice = §15 `[selection]` for this chip (no local benchmarking, no ranking cache; the existing selector reduces to this rule). Load passes `config.kernels` resolved for (chip, mode).

## 18. API v2 (owner: api)
- **Universal endpoint:** `POST /arsenal/v1/models/{id}/ops/{op}` — body validated against the op's input schema; response per `output`/`stream` (JSON, SSE `text-stream` events, chunked binary audio, image/file with content type). `GET /arsenal/v1/openapi.json` generated from all folders' ops (+ the dialect routes). `GET /arsenal/v1/catalog?kind=&q=&sort=&...` (browser data), `POST /arsenal/v1/models/{id}/add|remove`, `GET|POST /arsenal/v1/models/{id}/settings`.
- **Dialects are modules** behind one `Dialect` protocol (routes + mapping onto kind ops), registered in one list: OpenAI, Anthropic, ElevenLabs, Replicate, TypeSafe/Verdict. Adding a kind with an established standard = one module; without one, the universal endpoint covers it.
- New standard surfaces: ElevenLabs `/v1/sound-generation`, `/v1/music` (+ stream where Vireo had it), `/v1/audio-isolation` (enhancement), `/v1/text-to-voice` (voice design, Qwen3 VoiceDesign), voice cloning reference where Vireo had it; Replicate predictions for slow sound jobs. Verdict parity: System One, `/v1/judge`, `auto` model routing, response bytes as Verdict. Compare against Vireo's and Verdict's own API docs/routes and record the parity table in `docs/API.md`.

## 19. Kinds v2
`voices` (TTS, voice design, cloning), `sound` (sound effects and music: Stable Audio 3), `enhancement` (speech cleanup: RE-USE), `decisions`, `images`, `chat`. Each `kinds/<kind>.toml` adds `icon` (SF Symbol name), `title_singular`, standard `[[op]]`s with schemas in `kinds/<kind>/ops/`, browser facets.

## 20. Runner contract v2 (owner: runners-docs)
`docs/RUNNERS.md` is the single, complete contract for anyone building a runner or engine: process model, framing, handshake, ops by kind (from the op schemas), load config (assets, kernels, settings), errors, cancellation, memory reporting, unload/exit, what the gate checks, Exact vs Fast, how results/logs are written, how a PR is accepted. `Arsenal runner-check <runner command> --model <folder>` drives any runner through the protocol (handshake, ops with schema-valid inputs, cancel, malformed input, unload/exit, memory reports) and prints one line per check. Templates in `Tests/Fixtures/Templates/runners/{swift,c}/` (the C one is the non-MLX example: plain Metal or CPU) pass the checker in CI-style tests.

## 21. Native worker packaging
`Workers/Package.swift` is the one native build graph: it links every native family into one binary (one MLX copy) and owns every family's test target. Each family owner may add only its own dependency line, test section and dispatch case to `Workers/Package.swift` / `Workers/Native/Sources/arsenal-worker-native/Main.swift`; rebase before editing. Bundle size is reported by the build, not gated.

## 22. Licence MIT
`LICENSE` = MIT; Arsenal-authored and V-family-ported files use `// SPDX-License-Identifier: MIT`; upstream third-party headers unchanged; NOTICE keeps third-party notices. New files are MIT.

### §15 addenda (coordinator, 3 Oct)
- **Where settings reach the runner:** every load-time setting is passed as `load.config.settings = {<setting id>: <value>}` (canonical). `maps_to = "config.<key>"` is optional and additionally copies the value to that legacy key (e.g. `config.context_limit`, which the chat runner already reads). Request-time settings (`applies = "request"`, `maps_to = "request.<param>"`) are merged into the op's params unless the request sets that param.
- **Runner check inputs:** a folder may carry `gate/runner-check.json` (`{"<op>": {params…}}`) — valid, model-specific example inputs used by `Arsenal runner-check` (and usable as API examples in OpenAPI). The validator accepts and schema-checks it against each op's input schema.

### §17/§18 addenda (coordinator, 3 Oct)
- `ModelsService.operations(id:) -> [ModelOperation]` (`name, title, inputSchema: JSONObject, output, stream`), resolved from the **verified** installed/bundled folder plus its kind's standard ops; the API and OpenAPI use only this (never read folders themselves). Owner: service (from schema's declarations).
- Voice operations (voices kind): `speak {text, voice?, reference?:{path,text}, speed?, language?, instructions?, stream}` (inline reference = one-shot cloning), `voices {}` (presets + saved voices), `design_voice {description, preview_text?, language?} -> {voice_id, preview:{path}}` (Qwen3-TTS VoiceDesign, Breeze), `clone_voice {audio:{path}, transcript, name} -> {voice_id}` (Qwen3-TTS Base, Breeze, Pocket where supported). Saved voices persist per model under the support dir and appear in `voices`. ElevenLabs mapping: text-to-voice → design_voice; voices/add → clone_voice.
- Settings POST: `{"values": {...}, "apply": false}`; `apply: true` reloads. Add `{}`; remove `{"delete_weights": false}`.
- `gate/runner-check.json` accepts **two shapes**: flat `{"<op>": {params}}`, or structured `{"load": {config, assets, dtype, …}, "ops": {"<op>": {params}}, "cancel": {"op": "<op>", "params": {…}}}`. The validator schema-checks `ops` entries and `cancel.params` against the folder's merged op declarations; `load` is a checker override for exercising non-default settings and has no authority over real loads.
- **Operation capability (decided after mb-v2-voices' findings):** a folder's `[[op]]` list is **authoritative** for what that model supports; the kind's op declarations only supply shared schemas/titles for those names. A folder with no `[[op]]` gets only the kind ops marked `default = true` (e.g. `speak`, `voices`; never `design_voice`/`clone_voice`). Owners: schema (`resolvedOperations`), service (`operations(id:)`).
- **Request-time settings merge:** a request-time setting is merged into an op's params **only if that op's input schema declares the target property**, and only when the request did not set it. Unset/empty defaults (empty string, empty object, null) are never merged. Caller-supplied unknown fields still fail validation. Owner: service.
