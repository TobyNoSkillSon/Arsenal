# Arsenal HTTP API

One IPv4-loopback listener. Preferred port **8798**; `ARSENAL_API_PORT=0` chooses a free port, and an occupied preferred port falls back to a free port. The actual port and PID are in `$ARSENAL_SUPPORT_DIR/status.json`. The default support directory is `~/Library/Application Support/Arsenal`.

| Client | Base URL | Implemented profile |
|---|---|---|
| OpenAI Python 2.6.1, sync/async | `http://127.0.0.1:8798/v1` | Models list/retrieve; text Chat Completions, SSE, usage, caller-executed tools where the model supports them; speech WAV/PCM; image generation PNG base64/URL |
| Anthropic Python 0.72.0, sync/async | `http://127.0.0.1:8798/providers/anthropic` | Models; text Messages, named SSE, function tools, token count |
| ElevenLabs Python 2.21.0, sync/async | `http://127.0.0.1:8798/providers/elevenlabs` | TTS convert/HTTP stream; sound/music/isolation; voice design + reference cloning; model/voice discovery |
| Replicate Python 1.0.7, sync/async | `http://127.0.0.1:8798/providers/replicate` | Image and sound predictions create/wait/get/cancel; `run`; local model/version schema; output files |
| TypeSafe / Verdict raw HTTP | `http://127.0.0.1:8798/providers/typesafe` | System One and batch judge; models |
| fal | — | Not supported; its Python SDK does not provide a plain local-HTTP base override |

These are tested **wire profiles over deterministic CPU fixtures**, not claims that every installed model implements every capability. v2 catalog kinds are chat, images, voices, sound, enhancement and decisions. No JavaScript SDK certification is claimed.

Before Load or the first operation on an unloaded model, Arsenal compares the chosen precision’s table peak memory (if absent, known Load peaks and the declared memory forecast provide a conservative fallback) with the room under the Mac-wide line now: the RAM the Mac can use without swap, less the RAM already above the line of 100/128 of its RAM (107.4 decimal GB on a 128 GB Mac, which the app shows as 100 GB of 128 GB). A Load that would take the Mac past that line is refused, which keeps the Mac out of swap; the Mac-wide guard below would otherwise stop it. A shortage returns `insufficient_memory` with HTTP status 409, `details.needed_gb`, `details.free_gb` (decimal GB, the room under the line), `details.limit_gb` and `details.total_gb` (decimal GB), model and precision, and a force hint. Wire figures are decimal GB (10^9 bytes); the app shows Activity Monitor's GB (2^30 bytes), so 107.4 decimal GB reads as 100 GB. Cancel or unload other models; retry the same HTTP request or MCP tool with top-level `force:true` only after the user accepts macOS swap and its effect on the whole Mac. Force is request-scoped, never a saved preference, and does not bypass trust, missing assets or invalid inputs. HTTP clients can instead send `X-Arsenal-Force-Load: true`; this also works for multipart requests. For direct operation endpoints, `force` is transport metadata and is removed before model schema validation. An already loaded model needs no new load admission. Without a peak or declared forecast, loading proceeds with `loadMemoryNote` on the model row and a system-log diagnostic saying that fit could not be checked. Reload counts the outgoing worker’s released footprint and reserves the replacement before unloading; a genuine refusal keeps the resident. For replacement refusals, `free_gb` includes that release and `released_gb` identifies its contribution; the warning says that capacity is available after unloading. Recovery uses its captured revision’s evidence and at least its observed resident footprint. The native Load action shows a quiet line when the requirement is unknown. Legacy queue/swap preferences never suppress this warning or authorize swap for a load. A forced resident is exempt from automatic pressure eviction until it unloads; configured idle unloading still applies, and later Loads need their own force. Automatic recovery from a failed replacement preserves the previous resident’s consent, without authorizing the candidate or any later Load. Serving workers never stop for system swap growth alone. While any worker request runs (Load, operations, Get-time int8/int4 conversion, measurement and intake), Arsenal samples every half second the worker’s memory footprint and the whole Mac’s memory in use. When the Mac has more than 100/128 of its RAM in use, it stops one running worker: among those that grew by more than 512 MB during their current requests, the one furthest past its expected peak plus 1 GB (past where its requests started, when no peak is known). A worker that did not grow is never stopped for memory other apps took, and stops happen one at a time. Separately, a model with a measured request peak is stopped as a runaway at four times that peak and at least 16 GB above it, whatever the Mac’s state; a Load peak or forecast sets no per-worker bound. A forced resident takes no part in the Mac-wide stop; it is stopped only at twice its expected peak and at least 4 GB above it. The stopped request fails with `memory` (HTTP 507) once the worker has exited, with `details.reason` `mac_memory_limit` (`details.mac_in_use_gb`, `mac_limit_gb`, `start_gb`, `footprint_gb`, and `expected_gb` when known) or `runaway` (`details.footprint_gb`, `ceiling_gb`, `expected_gb`); the model unloads, and the stop does not count toward runner quarantine. Other strict memory and swap stops belong to lab/intake runs. Cache budgets and request validation remain active. Allocation errors and worker exits are reported as request failures without taking down Arsenal.

## Requests

Provider/control routes need no auth locally; MCP requires its bearer token (see [MCP](MCP.md)). SDKs may use the dummy key `local`. Keys are never logged. Any browser `Origin` is refused. Host must be `127.0.0.1:<actual-port>` or `localhost:<actual-port>`. No wildcard CORS or LAN listener is provided.

`/v1/models` lists only models added to the Arsenal table, including models awaiting Get, in the OpenAI schema with an additional `kind` field. Model IDs are exactly the table/MCP IDs; no cloud model names silently select a different model. Inference loads an installed model on demand and **never downloads**. An absent model returns 404 saying it is not installed and naming `arsenal_download(model: ID)` and its pending download size where known.

```python
from openai import OpenAI
client = OpenAI(api_key="local", base_url="http://127.0.0.1:8798/v1")
audio = client.audio.speech.create(
    model="kokoro-82m", input="Hello.", voice="af_heart", response_format="wav")
audio.write_to_file("hello.wav")
```

### Speech and voices

OpenAI speech accepts `response_format="wav"|"pcm"` explicitly. Both are mono signed-16-bit at 24 kHz; WAV has a real RIFF header. **The OpenAI default MP3 is rejected**, not relabelled WAV. MP3, AAC, Opus and FLAC are not offered in this profile. Local voice IDs are discovered through ElevenLabs, not an invented OpenAI voice endpoint. Unknown voices are refused by the selected worker.

ElevenLabs `output_format` is a query parameter: `pcm_<rate>` or `wav_<rate>` at 8000, 16000, 22050, 24000, 32000, 44100 or 48000 Hz. Explicit `pcm_24000` is compatible with the pinned SDK. Its default MP3 is rejected. The `/stream` endpoint sends binary HTTP chunks, never SSE; this initial adapter buffers the rendered clip before delivery. Alignment, WebSocket input and cloud voice-setting controls are unsupported. Sound/music, voice design and uploaded-reference cloning use their folder-declared operations (below).

### Chat

OpenAI `POST /v1/chat/completions` preserves messages and supported sampling/token parameters for the runner. Streaming sends stable completion IDs, deltas, a finish chunk, optional usage-only chunk, then `[DONE]`. Heartbeats are SSE comments, never model chunks. Tools pass through only when the worker supports them; tool arguments remain JSON strings. Image/audio content and named JSON schemas follow the capability profile below; hosted tools, logprobs and Responses are unsupported.

Anthropic `POST /providers/anthropic/v1/messages` translates its system/text/tool blocks into the chat runner's messages and its results back into Messages blocks. Streaming uses `message_start`, indexed block events, `message_delta`, `message_stop`; tool inputs are fragmented JSON and reconstructed as objects by the official client. Local thinking is exposed as separate `thinking` blocks / `thinking_delta` events when enabled (OpenAI: `reasoning_content`). The answer never includes that reasoning. Thinking signatures are empty local compatibility fields, not cloud-authenticated signatures; Anthropic `budget_tokens` is validated but ignored with `X-Arsenal-Warnings` because the local control is on/off. Caching, citations, hosted tools, Files beta and batches are not supplied. `messages/count_tokens` uses the runner's token-count operation.

### Images and predictions

OpenAI `POST /v1/images/generations` returns PNG `b64_json` or local `url` output; it stays synchronous. Supported `size` is `WxH`, multiples of 16, 64–2048 per side; `n` is 1–4. Image edits, masks, preview SSE and cloud quality/style controls are not implemented.

Replicate predictions use model-schema-specific inputs: `prompt`, `width`, `height`, `num_outputs`, `steps`, `seed`, `guidance`. Select a model by catalog ID in `version`, or its discovery version hash; SDK `run("local/<model-id>:<version>")` and official-model `run("local/<model-id>")` both work. Async creation returns 201. `Prefer: wait[=1..60]` waits boundedly, otherwise poll the returned local URL. Cancellation wins terminal races; polling/disconnect does not cancel a durable prediction. Webhooks, SSE output and `Cancel-After` are not offered.

**Pinned Replicate SDK limitation:** 1.0.7 wraps HTTPS/data URLs as `FileOutput`, but leaves local HTTP URLs as strings. Default `output_type="url"` returns downloadable local HTTP URLs; fetch those with your HTTP client. The model input extension `output_type="data_uri"` returns PNG or WAV data URIs, which the unchanged SDK wraps and reads as `FileOutput`. Neither route contacts a cloud origin.

Prediction records, selected runner/dtype/mode/catalog commit and output paths are stored under `support/jobs`. On restart, interrupted work becomes failed and is never regenerated. Completed output remains available at the new local origin. Output files are retained until the user removes the job directory; there is no automatic one-hour cloud expiry. Request JSON cannot name arbitrary local files, and output reads are confined to the jobs directory with resolved-symlink checks.

### Decisions

`POST /v1/systemone` and `/v1/judge` also exist at `/providers/typesafe/v1/...`. Noul is a probability, Choice preserves its distribution, and Score preserves its weighted value/legend. The API returns the decision worker's answers without numerical conversion. Universal decide/judge validate their input, then use the same raw-byte request/reply path as System One; question, option and structured-state order are preserved, and no `_raw_body` wrapper is exposed. `x-typesafe-request-id` and `x-verdict-bits` are supplied. Requests do not reload to a different `bits`; mismatches return 409. Root `/v1/models` remains OpenAI; existing Verdict clients must use the TypeSafe prefix for discovery. `auto` is accepted for existing decision callers but is not a table model and is not listed; its worker-directed routing preserves original request bytes and batch order. Actual fp16 checks on all five models confirm byte-identical System One and universal decide responses for the same ordered body. All five Verdict models are in the v2 catalog; no decision model installed still returns an actionable 404.

## Control API

Under `/arsenal/v1`: `GET status`, `GET models[?kind=]`, `POST get`, `GET get/{id}`, `POST cancel-get`, `POST load`, `POST unload`, `POST keep-hot`, `POST delete`, `GET catalog/browse, GET catalog/status`, `POST bench`, `GET bench/{job}`. JSON commands use `id`; Get adds `dtype`; Load optionally `dtype` and `fast` (legacy `precision` is accepted); Keep Hot requires `on`. Get starts asynchronously; load returns when loaded. Model rows include pending `downloadBytes` per dtype and the service's `verification` wording. Benchmark requests decode the injected `BenchmarkRequest` (including `quick_check`); a host without a configured benchmark controller returns 503 rather than pretend to run one.

A Get of a gated Hugging Face model ends with `get_error_code: "waiting_on_person"` (HTTP 403 where an error is returned), a message that says what the user must accept and where, and the row's state `failed` with phase `gated` and a `gate` object (`repo`, `page`, `hasToken`, `tokenRejected` when Hugging Face refused the token found, `mode`, `tries`, `limit`, `retryAt`; no `retryAt` once Arsenal stopped retrying). The app opens the model's page (and the token page when it has no token) once, retries every 30 s up to 10 times, and a new `POST get` retries at once. The token comes from Arsenal's Keychain item (Settings › Hugging Face Token…), else `HF_TOKEN`, else the Hugging Face CLI's token file; no route sets it.

## Universal operations and model controls

`POST /arsenal/v1/refresh` re-reads config, catalog and local model folders on demand, without restarting or interrupting running operations. It returns updated counts/catalog status.

A pinned weights file that Hugging Face does not have at the pinned revision ends a Get for good: `pinned_file_missing` (HTTP 422), with the row's state `failed` and phase `pinned_file_missing`; the message names the file, repository and revision, and retrying cannot help. A local folder whose base pins are missing (checked against the repository at discovery, approval and Get) becomes an error row and refuses approval and Get with `local_folder`.

`GET /arsenal/v1/models/{id}/get-plan?dtype=D` previews weights, local imports and dependencies before Get: `{items:[{name,kind,downloadBytes,localCopyBytes}],totalDownloadBytes,requiresDownload}`. Unknown download sizes are `null`, not zero; bundled/installed dependencies need zero download bytes. Clients should show the plan and obtain consent before `POST /get`. Source retention (16-bit copies above 8 GB, removed once a derived int8/int4 tier is selected and loaded) is visible: that row's 16-bit state is `notInstalled` with `phase:"sourceRemoved"`, `retentionNote` names the restore download, and a plan that must restore the copy says so in its weights item name. A plan reviewed before the removal no longer covers the Get and returns `get_plan_changed` with the revised plan.

`POST /arsenal/v1/models/{id}/ops/{op}` accepts the operation's input object directly. `ModelsService.operations(id:)` supplies the hash-verified, resolved model/kind declaration; the API never reads model folders. The service merges request-setting defaults and validates against that JSON Schema before invoking the runner. Unknown ops return 404; invalid inputs return 422 before streaming begins. Inference never downloads.

Outputs follow the declaration: `json` returns JSON; `text-stream` sends named SSE events and a final `result` event (request `stream:false` for a JSON result); streamed `audio` returns chunked signed-16-bit PCM with `X-Arsenal-Sample-Rate`, `X-Arsenal-Channels`, `X-Arsenal-Encoding`; otherwise `audio`, `image` and `file` return the generated file bytes with their content type (multiple outputs use `multipart/mixed`, not a silently discarded extra file). Inline file output (the binary response body) is limited to 64 MiB; a larger file result is requested with JSON, which returns the job's files as `files[]` URLs under `/arsenal/files/`. Those files are written to the job directory and streamed from disk, so they do not count toward the 64 MiB inline cap; each job's files together are limited to 4 GiB, and a job above that fails with an error naming the 4 GiB per-job artifact limit. Files are confined to Arsenal jobs. HTTP clients cannot set `out_dir`; Arsenal creates and cleans its own transient directory. Stream disconnect cancels the model call; no binary stream contains SSE heartbeats.

BiRefNet declares `remove_background` under Images: send `{"image":{"path":"<uploaded inbox path>"}}` to `/arsenal/v1/models/birefnet/ops/remove_background` with `Accept: application/json`. It returns `files` in foreground/mask order, with PNG URLs, dimensions, bytes and SHA256. The foreground is RGBA and the mask is 8-bit alpha. PNG/JPEG only, up to 16 megapixels; preprocessing is 1024×1024. Its declaration replaces the Images kind's generation operation, so `/v1/images/generations` does not apply. See the model README for precision evidence and limits.

`GET /arsenal/v1/openapi.json` produces OpenAPI **3.1**, from all catalog models' resolved operation input schemas and the registered dialect routes. Input schemas appear under `components.schemas`; local references are relocated there. Provider schemas document the supported wire profile, not every provider feature. The six `Dialect` modules are registered in `DialectRegistry.standard`; new kinds need no router branch to use the universal route.

```sh
curl -s http://127.0.0.1:8798/arsenal/v1/models/stable-audio-3-small-sfx/ops/generate_sound \
  -H 'Content-Type: application/json' -d '{"prompt":"Bird calls","duration":2}' -o bird.wav
```

`GET /arsenal/v1/catalog` returns `[CatalogEntry]` (each entry contains its `row` plus facets, suggestion/popularity, sizes and added/installed/loaded flags), not catalog trust metadata; trust metadata moved to `/catalog/status`. Query keys: `q`, comma-separated `kind/licence/language/task/dtype/modality_in/modality_out`, booleans `commercial/suggested/added/installed/loaded/fits_ram`, `min_size_bytes/max_size_bytes`, `min_ram_mb/max_ram_mb`, `sort` and `order=asc|desc`. Sort columns are `name/kind/size/ram/popularity/likes/params/publisher/licence/released/added/installed/loaded`; suggested models remain first. Dtype, size and RAM constraints must match the same offered format. `matchingDtype` names it; the entry projects its row size/RAM onto that dtype, with `ramMBByDtype` and `formats` describing the alternatives.

- `POST /models/{id}/add` with `{}` adds to your table without downloading.
- `POST /models/{id}/approve` with `{"content_digest":"<row.contentDigest>"}` grants consent to that exact local folder. A stale digest returns 409; row `codeConsentRequired/contentDigest/needsReload` are always present in table JSON.
- `POST /models/{id}/remove` with `{"delete_weights":false}` removes/unloads; `true` also deletes weights.
- `POST /models/{id}/delete-withdrawn` with `{}` deletes the files of precisions still on this Mac that the model no longer offers (the row's `withdrawnTiers`, precision to bytes), as the table's Remove does; offered precisions stay. It returns `{"ok":true,"removed":["int4","int8"]}` (empty when there was nothing to remove), and 409 while a Get or Load runs or while a withdrawn precision is loaded.
- `GET /models/{id}/settings` returns declaration/value/pending states.
- `POST /models/{id}/settings` with `{"values":{"context_limit":8192},"apply":false}` sets values. Load settings stay pending until Reload; request settings apply to the next call. `apply:true` explicitly reloads. Values are applied in sorted key order; a failed key returns an error, not a success. `dtype`, `keep_hot`, `show_in_menu` and `runner` are accepted beside the model's own settings, with MCP `arsenal_settings`'s checks, all checked before any is applied.

These paths have the `/arsenal/v1` prefix. Rows still serialize the existing `selectedPrecision/precisions` keys for compatibility; their values are dtypes, not bit counts.

## Vireo and Verdict parity

ElevenLabs paths below have `/providers/elevenlabs` prepended. `X-Arsenal-Model` names the local model; `X-Arsenal-Warnings` names ignored controls. The new audio routes default to WAV; explicitly request PCM for official SDK output. MP3/Opus/G.711 encoders are **not ported**: requests for those formats fail honestly. This is a supported local wire profile, not full Vireo/provider parity.

| Route | ElevenLabs behaviour | Arsenal |
|---|---|---|
| `POST /v1/text-to-speech/{voice}/stream` | Incremental speech, disconnect cancellation; many codecs | PCM/WAV profile retained; currently buffered. Inline `reference:{path,text}` passes to `speak` for one-shot cloning. |
| `POST /v1/sound-generation` | SA3, prompt, duration, model aliases; loop/influence warned | `generate_sound` (needs a user-added sound model; no catalog model has it); local IDs or `eleven_text_to_sound_v2`; defaults 5 s; loop/influence warned; stereo WAV retained at native rate. |
| `POST /v1/music` and `/stream` | Instrumental SA3; composition plans flattened; stream after render | `generate_music`; music aliases; duration ms → seconds; flattened styles/negative prompt; direct lyrics/language/instrumental fields follow model support; composition-plan lyric lines still warn; binary chunks after render. |
| `POST /v1/audio-isolation` and `/stream` | Uploaded audio → RE-USE; SAM extension unavailable | Multipart upload → `enhance`; accepts PCM16/16 kHz form mode; no arbitrary client file paths; SAM returns 409. |
| `POST /v1/text-to-voice/design`, `/create-previews` | Three description-based previews | Three `design_voice` calls; SDK preview fields; bounded preview history (32), preview controls not supported are warned. |
| `GET /v1/text-to-voice/{id}/stream` | Preview audio until restart | PCM/WAV preview bytes; preview IDs are session-scoped. |
| `POST /v1/text-to-voice` | Saves a selected preview | Uses `clone_voice` with preview audio/transcript/name; worker owns persistent saved voices. Requires an installed cloning-capable model. |
| `POST /v1/voices/add` | Uploaded reference; transcript uses Base, otherwise Pocket; optional Vella transcription/noise cleanup | `clone_voice` using uploaded first reference, optional transcript/labels; no automatic Vella transcription; current clone schemas require transcript text; no automatic Vella transcription or noise-cleanup. |
| `GET /v1/voices`, `/v2/voices`, `/{id}` | Curated/preset/saved voices; facets/paging | Lists worker `voices` including saved voices; one-page profile, no Vireo paging/filter parity. |
| `POST /v1/systemone` (also TypeSafe prefix) | Ordered System One input and response; probabilities unchanged | Raw request/response preserved through decision worker; 39 golden response-byte cases pass. |
| `POST /v1/judge` (also TypeSafe prefix) | Batch results, per-item errors, order retained | Worker batch reply preserved; worker-directed split routing reassembles input order. |
| Decision `model:auto` | ≥99.5% ASCII letters → English, otherwise Multilingual | The decision worker owns that rule; router forwards unchanged bytes and follows `wrong_model` routes. Discovery offers auto only with both installed. |
| Replicate `/v1/predictions` | Standard durable slow jobs (Vireo has no Replicate adapter) | Images and sound (`generate_sound`/`generate_music`); wait/get/cancel/restart persistence, WAV URLs/data URIs. |

Replicate sound inputs: `prompt`, `duration`, `seed`, `steps`, `negative_prompt`, `op="generate_sound"|"generate_music"`, `output_type="url"|"data_uri"`. A model must declare the requested operation. Folder settings and Exact/Fast/dtype are shared by every client; provider requests never switch dtype.

## Transport and tests

HTTP/1.1 has one request per connection (`Connection: close`); request chunking and response chunking are supported. Requests are bounded to 64 KiB headers, 1 MiB JSON and 16 MiB multipart, with 32 concurrent connections, 10-second head timeout and 30-second body-idle timeout. Duplicate framing/security headers, ambiguous Content-Length/Transfer-Encoding and malformed multipart are refused before routing. Multipart parsers preserve repeated fields; uploads do not imply an implemented image-edit route. Synchronous disconnect cancels its task through ModelsService; durable predictions have independent tasks.

```sh
tests/conformance/run.sh
```

The command creates the private hash-pinned SDK venv, builds the real listener, runs sync/async SDK tests plus new sound/music/isolation/design/clone profiles, generated OpenAPI validation and adverse transport/disconnect/restart tests, and prints a compact pass/fail table. It checks the measurement window and runs through `scripts/mlock work`. Evidence goes to `.build/conformance/cache/results.json` by default; support/cache never use the real app directory. Dependency pins and hashes are in `tests/conformance/requirements.lock`.

Responses, browser, LAN and full provider parity are not advertised. Capability profiles below define the supported embedding, transcription and Realtime subset. An unsupported endpoint returns a dialect-native error rather than a successful no-op.

## Embedding in the app

```swift
let server = ArsenalAPIServer(service: realModelsService, supportDirectory: paths.support,
                             benchmarkController: controller)
let actualPort = try await server.start()
// Keep server alive with the app; server.stop() cancels connections and clears its status port.
```

`ARSENAL_TESTSERVER_CATALOG=stub` selects plain `StubModelsService()` with every model initially not installed (for an isolated agent Get flow); the default fixture catalog starts installed.

The library owns no AppKit or inference engine. `arsenal-testserver` hosts the same library over deterministic fixtures backed by `StubModelsService`.

## Agent connection and playback

`GET /arsenal/v1/mcp/connect` returns `{url,claude,codex,token_path}` with token-free command templates for the native Connect menu. The app reads the private bearer token in-process to assemble copied commands. MCP uses `/mcp` on this listener.

`GET /arsenal/v1/playback` returns `{muted,current,queued,playing}`. Current carries `id,path,requester,model,voice,gainDB`. `POST` accepts `{muted:true}` to pause/resume or `{stop:true}` to clear the queue. The voice menu and MCP share this queue.

Model discovery: `arsenal_find(on_mac:"table")` and provider model lists show the table only. `arsenal_find` and `GET /arsenal/v1/catalog/browse` browse Get more models. `GET /arsenal/v1/catalog` remains for in-tree browser callers. Listing a row does not imply its weights are installed; inference never downloads. Research and deliberate serving limits: `research/model-contracts/`.

## Capability profiles

The API foundation adds these local wire profiles. A route existing does not mean a compatible model is installed. The selected model's operation/input schema is authoritative; unsupported options fail before model execution. No request downloads weights. SDK certification uses deterministic CPU fixtures, separately from model-quality qualification.

| Profile | Routes | Operation / supported wire |
|---|---|---|
| OpenAI transcription | `POST /v1/audio/transcriptions`, `/v1/audio/translations` | Multipart `file,model,language,prompt,response_format,timestamp_granularities[],stream`; json/text/verbose_json/diarized_json/srt/vtt. Translation sets `task:translate` and means English output. Known-speaker references/logprobs/temperature are refused. |
| OpenAI transcription SSE | same, `stream:true` | `transcript.text.delta`, optional `transcript.text.segment`, `transcript.text.done`. Model must declare streaming. |
| OpenAI embeddings | `POST /v1/embeddings` | `input` text or array, dimensions, float/base64 (little-endian Float32), task query/document, instruction, normalize/truncate. Token-ID arrays are outside this profile. Image/audio content parts are local extensions. |
| Cohere | `POST /v2/embed`, `/providers/cohere/v2/embed` | Texts/images or grouped mixed-content inputs, search_query/search_document, float embedding output, output_dimension, NONE/END truncate. Grouped inputs preserve all text/image/audio parts as one embedding item. Quantized embedding encodings are refused. |
| Jina/Cohere rerank | `POST /v1/rerank`, `/v2/rerank`, `/providers/cohere/v2/rerank` | query, text/object documents, top_n, max_tokens_per_doc, instruction, model-declared return_embeddings; original index, relevance_score, optional document text and per-result embedding vectors. Unsupported options return 422. |
| DeepL | `POST /providers/deepl/v2/translate` | JSON or form: text array, target_lang, source_lang, context; translations with detected source language. Glossaries/tag handling/formality/model selection are refused. |
| DeepL languages | `GET /providers/deepl/v2/languages`, `/v2/languages` | Union of explicitly enumerated source/target language codes in installed translation schemas; open language schemas do not fabricate a language list. |
| Mistral OCR | `POST /providers/mistral/v1/ocr` | document image_url/document_url, pages, table_format, include_image_base64, document_annotation_format; pages carry markdown/tables/formulas and optional images/annotation. |
| OpenAI chat | `POST /v1/chat/completions` | Text, image_url and input_audio content; named response_format json_schema passes to the model and the final JSON is schema-validated. Model-local schemas must narrow unsupported content/options. |
| Anthropic image input | `POST /providers/anthropic/v1/messages` | User image blocks with base64 PNG/JPEG source are mapped to image_url parts. Text/tool blocks retain the existing profile. |

DeepL SDK `server_url` must end with `/providers/deepl/` (trailing slash); OpenAI Realtime clients use `websocket_base_url="ws://127.0.0.1:PORT/v1"` explicitly for this loopback listener.

Media URLs are inline base64 data URLs; remote origins are never fetched. Mistral also accepts the local `document.path` extension for an already uploaded inbox file. Universal/MCP file objects retain the existing jobs/inbox confinement. OpenAI chat data parts remain inline for the model's own processor. A schema-valid response does not imply grammar-constrained decoding; the adapter must implement that when its model promises strict generation. An invalid final JSON answer fails with 502. Already emitted streaming deltas cannot be recalled.

## Capability aliases

Each alias accepts the operation's exact input plus `model`. Omitting model works only when exactly one installed model declares the operation. Remove the model selector and the request is identical to `POST /arsenal/v1/models/{model}/ops/{operation}`. Input/output schemas and validation are shared with MCP.

| Alias after `/arsenal/v1/` | Operation |
|---|---|
| `audio/transcribe` | `transcribe` |
| `audio/diarize` | `diarize` |
| `audio/end-of-turn` | `end_of_turn` |
| `audio/separate` | `separate_audio` (stems or text prompt, extract/remove) |
| `audio/music` | `generate_music` (optional lyrics/language/instrumental) |
| `audio/enhance` | `enhance` |
| `embeddings/embed` | `embed` |
| `text/rerank` | `rerank` |
| `text/translate` | `translate` |
| `text/extract` | `extract` (JSON schema or labels) |
| `documents/read` | `read_document` |
| `images/detect` | `detect` (labels, referring phrase or instruction) |
| `images/ground` | `ground_screen` |
| `images/segment` | `segment` (points, boxes, text or reusable session) |
| `images/segment-session` | `segment_session` (create/close, scoped to loaded model) |
| `video/session` | `video_session` (create/close reusable tracking state) |
| `video/segment` | `track_video` (video/session_id and frame-indexed prompt updates) |
| `images/depth` | `estimate_depth` (relative/metric, points/intrinsics) |
| `images/upscale` | `upscale_image` |
| `video/upscale` | `upscale_video` |
| `video/interpolate` | `interpolate_video` (video or two frames) |
| `images/colorize` | `colorize` (image, optional input_size) |
| `images/inpaint` | `inpaint` (image and a same-size mask) |
| `images/to-3d` | `image_to_3d` (GLB output) |

Spatial prompts are normalized to the original source image/frame. Screen-grounding results also support coordinate_system:pixels with image_size, preserving screenshot pixel points. Depth accepts image or multi-view images and can return per-view indexed camera/file metadata. Depth explicitly returns `depth_unit:relative|meters`. File operations return `files[]` with JSON Accept or binary/multipart via the universal transport. GLB uses `model/gltf-binary`, MP4 `video/mp4`. The 64 MiB inline bound applies to binary video/3D responses; the files of a job are limited to 4 GiB in total, and larger jobs fail explicitly. No HF, face, lip-sync, forecasting or tabular profile is claimed by this foundation because those are outside the 46-model integration set.

## Generic asynchronous operations

Send `Prefer: respond-async` on any universal operation or alias. Valid requests return 202, `Preference-Applied: respond-async`, Location and `{id,model,operation,status,output,error,failure,progress,urls:{get,cancel}}`. A failed job's `failure` is the object a direct request's error body carries — `{error,code,status,details}` — so an `insufficient_memory` refusal keeps `details.needed_gb`/`free_gb` and its retry-with-`force` message; it is null otherwise, and `error` keeps the message alone. Invalid requests fail synchronously with 422. Poll `GET /arsenal/v1/jobs/{id}`; cancel with `POST /arsenal/v1/jobs/{id}/cancel` (empty body allowed). Status words are `starting`, `processing`, `succeeded`, `failed`, `canceled`. Cancellation is idempotent and a terminal state wins races. Disconnecting the creation/poll request does not cancel the durable job. At most 32 jobs execute; only 128 terminal records stay cached in RAM. Records/output persist under jobs/operations; on restart an interrupted record becomes failed when read, never regenerated. Polling completed jobs rebuilds file URLs for the current listener origin after an app restart.

## Live transcription sessions

`GET /v1/realtime?intent=transcription&model=ID` upgrades a version-13 WebSocket. Browser Origins remain refused. Input is mono signed-16 PCM at 24 kHz. Masked text frames (including continuation, ping/pong and close) are bounded to 1 MiB; an utterance transcribed at commit is bounded to 16 MiB of audio.

Send `session.update` with `session.type:transcription`, `audio.input.format:{type:audio/pcm,rate:24000}`, and `audio.input.transcription:{model,language?}`. The older `input_audio_format:pcm16` / `input_audio_transcription` keys are accepted. Append base64 audio with `input_audio_buffer.append`, then commit or clear. Server events include `session.created/updated`, speech_started/stopped, committed/cleared and `conversation.item.input_audio_transcription.delta/completed`; errors are session error events. A model that declares a native session operation streams: Arsenal opens it (`transcribe_session`, or `diarize_session` for a diarizer chosen by `model`) at the first append, resamples the socket's audio to 16 kHz continuously with one resampler per session, pushes about every 200 ms of appended audio and forwards each stable `delta` as `conversation.item.input_audio_transcription.delta` while the speaker talks; commit pushes the rest, finishes (its final text is the `completed` transcript) and closes the session, and clear or disconnect closes it. A failed push reports an error event and closes that session; the next append opens a fresh one, and text already forwarded for the item stays in its transcript. A diarizer returns its whole speaker timeline on every push, and earlier segments can still grow, merge or be renumbered, so each changed timeline arrives as `arsenal.diarization.snapshot` `{item_id,final,segments,speakers}` that replaces the previous snapshot in full; at commit the finished timeline arrives as a final snapshot and as `conversation.item.input_audio_transcription.segment` events `{item_id,id,start,end,speaker?,text?}`, which are never revised. A streaming utterance is bounded by its model's session limit (ten minutes for the bundled ones) rather than the 16 MiB buffer. Other models transcribe at commit; their single delta is the committed utterance. 

For model-driven turn completion use `audio.input.turn_detection:{type:semantic_vad,model:LOCAL_END_OF_TURN_ID}`. The model field is a local extension selecting a separately installed `end_of_turn` model. It evaluates the latest eight seconds after each appended chunk of at least 100 ms and commits a predicted completed turn. No energy-based/cloud server_vad is fabricated. Session updates expose the actual local selector as session.arsenal_model; the echoed transcription.model is omitted because the pinned SDK restricts that field to cloud model literals. All server events include event_id, speech events share the utterance item_id, and completed transcriptions include measured audio-duration usage. Omitting turn_detection.model resolves the unique installed end-of-turn model; ambiguous selection requires an explicit model. Standard semantic_vad options support auto/medium eagerness and create_response:false/interrupt_response:false; other values are explicitly refused in this transcription-only profile. Sessions and image/video sessions end at disconnect/unload; no audio is retained outside the temporary isolated job after session close.
