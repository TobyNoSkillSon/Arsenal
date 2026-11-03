---
name: arsenal
description: Use Arsenal's on-device models on this Mac through its MCP tools (HTTP for scripts) to transcribe, translate, read documents, extract data, decide or filter text, speak or make voices, make or edit images, make music, separate or clean audio, detect, segment, estimate depth, embed, search by meaning or rerank; also to find which local model fits a job, read its README and choose between models, and to add a Hugging Face model.
---

# Arsenal

Arsenal is the Mac app that runs AI models locally. You drive it through its MCP tools; the loopback HTTP API is for scripts. Answers are short text written for you to relay; errors start with a `[category]` tag. The work goes **find → read → choose → run → wait**, then you relay the result.

## Connect

If the `arsenal_*` tools are missing, start Arsenal, then register its stdio bridge once; it finds the app's port and token on every launch:

```sh
claude mcp add arsenal -- arsenal-mcp
codex mcp add arsenal -- arsenal-mcp
```

Without `arsenal-mcp` on your PATH, use `/Applications/Arsenal.app/Contents/Helpers/arsenal-mcp` or the `~/Applications` copy. HTTP clients register the endpoint instead; the port is only preferred and can change after a restart, so read `status.json` again when the connection fails:

```sh
support="${ARSENAL_SUPPORT_DIR:-$HOME/Library/Application Support/Arsenal}"
port=$(plutil -extract port raw -o - "$support/status.json")
export ARSENAL_MCP_TOKEN=$(cat "$support/mcp-token")
claude mcp add --transport http arsenal "http://127.0.0.1:$port/mcp" --header "Authorization: Bearer $ARSENAL_MCP_TOKEN"
codex mcp add arsenal --url "http://127.0.0.1:$port/mcp" --bearer-token-env-var ARSENAL_MCP_TOKEN
export ARSENAL_URL="http://127.0.0.1:$port"
```

Keep the token out of transcripts. Codex needs the exported variable when it starts.

## The tools

| Tool | Use |
|---|---|
| `arsenal_find(task?, query?, on_mac?, language?, commercial?, runs_here?, state?, limit?)` | Models that fit a job, with size, RAM need, state, good for and limits. |
| `arsenal_model(model, docs?, operation?)` | One model: state per precision, RAM against free memory, each operation's fields and a `Run:` line. `docs:true` adds its README. |
| `arsenal_run(model, operation?, input, play?, allow_download?, force?)` | Do the work. |
| `arsenal_wait(job)` / `arsenal_cancel(job or model)` | Receive a job's result / stop a job or a download. |
| `arsenal_status(history?)` | Memory against the limit, loaded models, running jobs and downloads. |
| `arsenal_download`, `arsenal_load`, `arsenal_unload` | Download without running; load at a `dtype`; unload (`model:"all"` too). |
| `arsenal_add`, `arsenal_remove`, `arsenal_delete` | The user's table in the Arsenal menu; use only when the user asks. The table holds exactly the models with files on this Mac or a download under way, so a consented download puts a model there and add of one not downloaded waits for its download. Remove only clears a model without files; delete erases files, which takes the model off the table: without `confirm:true` it only reports what it would erase, so relay that and confirm only after the user's yes. |
| `arsenal_settings(model?, set?, apply?, action?, content_digest?)` | A model's or the app's settings. `approve_code` needs the user's yes. |
| `arsenal_upload(filename, data)` | Base64 up to 700 KiB, only for a file that is not on this Mac. |
| `arsenal_playback(action)` | `stop`, `mute`, `unmute`, `status` for spoken output. |

`format:"json"` returns structured data when you compute over a result. Full schemas: `arsenal://models/ID/operations/OP/schema`; README: `arsenal://models/ID/README`.

## Find

Call `arsenal_find` with the job, not a model name you remember:

- `task:` an operation or kind: `transcribe`, `diarize`, `translate`, `read_document`, `extract`, `decide`, `speak`, `clone_voice`, `design_voice`, `generate` (images), `generate_music`, `remove_background`, `inpaint`, `colorize`, `upscale_image`, `upscale_video`, `interpolate_video`, `separate_audio`, `enhance`, `detect`, `segment`, `estimate_depth`, `ground_screen`, `embed`, `rerank`, `end_of_turn`.
- `query:` plain words when you don't know the operation: `query:"remove the background from a photo"`, `query:"cut out the subject"`. `task` takes one of the words above only; a phrase in `task` matches nothing.
- `on_mac:"installed"` lists what is already downloaded; `language:"pl"` keeps models that handle Polish; `commercial:true` keeps open-for-business licences; `state:"failed"` finds broken downloads.

The catalog and the user's own added models both appear. An empty answer means no model matched those words: try once with other words or a `task:`, then tell the user no local model does it, and offer nothing invented.

## Read

Before the first run of a model in a session, call `arsenal_model(model:ID, docs:true)`. The README says what the model is for, the exact input it expects (sample rates, image sizes, series format, question format), its limits, licence and upstream links. Field names, bounds and defaults in the operation list are authoritative; copy them, never guess.

## Choose

Work down this list and stop at the first rule that separates the candidates:

1. **The user's terms.** A model, language or licence the user named wins.
2. **It does the job.** The operation exists and the README's purpose and limits fit the input (language, length, file type, data shape).
3. **It runs here.** RAM need fits the free memory `arsenal_model` reports. A downloaded model beats one that needs a download.
4. **Licence for the use.** For anything shared or sold, prefer `commercial` licences and read conditions such as CC-BY attribution or revenue caps.
5. **Quality against size and speed.** In one family the larger model is more accurate and slower (Qwen3 ASR 1.7B over 0.6B, Index Translate 9B over 2B). Pick the larger when accuracy matters and it fits; the smaller for quick drafts, long batches or low memory. Figures `arsenal_model` prints are labelled measured or quick check, on the chip named; treat a quick check as a rough guide.
6. **Still tied:** the one already on the user's table or loaded.

Say which model you chose and why in one line.

## Run

```text
arsenal_run(model:"ID", operation:"OP", input:{...})
```

- Name `operation` whenever the model has more than one; the default is the model's main one, which may not be the one you want.
- File inputs are absolute paths on this Mac: `{"audio":{"path":"/absolute/talk.m4a"}}`. Arsenal clones the file into its inbox and leaves the original untouched. Only regular files are accepted; pass only files the user means to use.
- Speech plays aloud by default: pass `play:false` to only save the file. That is enough for silence; mute and unmute are the user's controls, shared with every client.
- Per-request settings (seed, steps, voice) go in `input`; `arsenal_settings(model:ID)` lists the model's own settings, and changing one affects every client.
- Results are text, JSON or files. Give the user the absolute file paths Arsenal prints. Long text also lands in `output.txt`, named on the first line.

HTTP for scripts: `POST $ARSENAL_URL/arsenal/v1/models/ID/ops/OP` with the same input object (no key needed locally; JSON bodies up to 1 MiB, so chunk large piles by bytes). `GET $ARSENAL_URL/arsenal/v1/openapi.json` has every schema; send `Prefer: respond-async` and poll `GET /arsenal/v1/jobs/ID` for long work.

## Downloads need the user's yes

A model that is not downloaded answers with facts, not an error: `Not run: … isn't downloaded yet — 2.5 GB, about 4 min …`. Tell the user the size and time and ask. Only after their yes, repeat the same call with `allow_download:true`. The flag is your statement that they agreed, so never pass it on your own; the same holds for `arsenal_download`, `arsenal_load` and updates. `force:true` (load despite low memory, letting macOS swap) also needs their yes.

## Long jobs

- Quick work returns in the run call (`wait_seconds`, default 25, at most 45). Slow work, downloads and loads return a job id.
- Tell the user what is happening, then call `arsenal_wait(job:"…")`. It returns after about 50 s with progress and `still running`; relay the progress and call again until it finishes or fails. Pass a longer `timeout_s` only when your client moves long tool calls to the background.
- Stopping a wait leaves the job running; `arsenal_cancel(job:"…")` stops it. A stopped download resumes next time. After an Arsenal restart `arsenal_wait` reports the job interrupted; repeating the original call resumes it.

## Special inputs

The README has the exact format; these are the ones that trip agents:

- **Audio** (transcribe, diarize, separate, enhance): `.wav`, `.flac`, `.mp3`, `.m4a`, `.aiff`, `.caf`; Arsenal resamples and mixes down itself. Transcription takes up to ten minutes per file; split longer recordings.
- **Decisions** (`decide`, `judge`): typed questions over a `state` object, answers are probabilities. Write them with [DECISIONS.md](DECISIONS.md) (MCP: `arsenal://integration/DECISIONS.md`).
- **Extraction** (`extract`): a template whose values name types, not a JSON Schema; see the model's README.
- **Search by meaning**: `embed` the documents with `task:"document"` and the query with `task:"query"` (where the schema has `task`), read the vectors with a script from the `output.json` the text answer names (`format:"json"` would paste every vector into your context), rank by cosine similarity, then `rerank` the top candidates with the query. Keep one embedding model per index.
- **Voices**: list real voice ids with `operation:"voices"` before `speak`; cloning needs a short reference clip (and for Qwen its exact transcript); voice design needs a description.
- **Images and video**: images are PNG or JPEG, videos what the schema accepts; a mask for inpainting must match the image size. Generation sizes are multiples of 16.
- **Chat**: no catalog model chats; a user-added one offers `chat` with `messages`, as its README says.
- **Time series** (forecasting): Chronos-2, Chronos-Bolt, TimesFM 2.5, Granite TTM and T0 forecast (`task:"forecast"`). Each README gives the series layout (one numeric column, its order and frequency) and horizon limits. Read the CSV yourself and pass the values in the shape the schema asks for.

## Refusals

- `[memory]` names what to unload; unload it, or retry with `force:true` after the user accepts swap. `Won't run on this Mac` cannot be fixed.
- `[invalid_input]` names the field: re-read `arsenal_model` and fix the call. `[unknown_model]` suggests the closest id. `[busy]` names the job to wait for or cancel. `[approval]` needs the user's yes for local code. `[disk]` needs free space. `[download]` usually gives the resume call (the user already agreed). Two exceptions: when the download plan changed size, it says to ask the user again before retrying with `allow_download:true`; when a pinned file is not in its repository, it says not to call again: tell the user the model's pins must change. `[unsupported]`, `[not_found]`, `[limit]`, `[cancelled]` and `[failed]` say what happened: relay them. `[waiting_on_person]` means a gated Hugging Face model needs the user to accept its terms (and, without a token, to add one in Arsenal's Settings): tell them; Arsenal opened the pages and retries by itself.

## More

- Write decision questions and know their measured weak spots: [DECISIONS.md](DECISIONS.md) (`arsenal://integration/DECISIONS.md`).
- Add a Hugging Face model, port a new architecture or optimise one: [ADDING.md](ADDING.md) (`arsenal://integration/ADDING.md`).
- Every catalog model with uses and limits: `arsenal://integration/MODELS.md` (in the repository, `docs/MODELS.md`); wire details: `docs/MCP.md` and `docs/API.md` in the Arsenal repository.
