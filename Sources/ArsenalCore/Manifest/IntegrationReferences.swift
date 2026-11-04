// SPDX-License-Identifier: MIT
import Foundation

// Embedded read-only integration documents. Keep byte-for-byte parity with their source documents.
// swiftlint:disable:next type_body_length
public enum IntegrationReferences {
    public static let documents: [String: String] = guides.merging(["MODELS.md": models]) { a, _ in a }
    static let guides: [String: String] = [
        "SKILL.md": ##"""
        ---
        name: arsenal
        description: Use Arsenal's on-device models on this Mac through its MCP tools (HTTP for scripts) to transcribe, translate, read documents, extract data, decide or filter text, speak or make voices, make or edit images, make music, separate or clean audio, detect, segment, estimate depth, embed, search by meaning or rerank; also to find which local model fits a job, read its README and choose between models, and to add a Hugging Face model.
        ---

        # Arsenal

        Arsenal is the Mac app that runs AI models locally. You drive it through its MCP tools; the loopback HTTP API is for scripts. Answers are short text written for you to relay; errors start with a `[category]` tag. The work goes **find → read → choose → run → wait**, then you relay the result.

        For showcases, tests, demos and evaluations, run models through the installed Arsenal, not an isolated copy; isolate only to test an unreleased Arsenal build.

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
        """##,
        "DECISIONS.md": ##"""
        # Arsenal decisions: writing questions

        Typed decisions (`decide` for one state, `judge` for many items) over the decision models. Each model's README (`arsenal_model(model:ID, docs:true)`) has its exact question format, context limit and languages; this page is how to ask well.

        Use for a fairly long instruction plus a few defined answers, repeated over thousands of text items (or images with d1-3B): shortlist, route, classify or score. Give each item the evidence needed for one focused judgment. Use code for exact matching, numeric requirements, dates and counts; read sources or use a reasoning model for cross-item arguments, missing context and consequential verification. Typed answers are an interface guarantee, not a truth guarantee.

        ### Pick a decision model

        | Model ID | Role and limits |
        |---|---|
        | `d1-3b` | Text/JSON or images; 32,768 tokens, 16 advertised languages including Polish. Vendor bf16, 3.12B, on MLX. Pick over Laya for images or context above 8,192 tokens. Fresh Polish job Noul: 24/32 versus Laya 17/32; no general quality advantage (see below). |
        | `laya-multilingual` | Non-English or mixed-language text, including Polish written without diacritics; 1,024-token context. First candidate for semantic filtering of Polish ads. |
        | `laya-english` | General English decisions; 512 tokens. |
        | `laya-typed-decisions` | English workflow decisions (tickets, invoices, incidents); 1,024 tokens. Evaluate on your data before replacing another model. |
        | `von-1.2` | English alternative, scores options independently of their order; 8,192 tokens. |
        | `von-1.1` | Earlier English alternative; 2,048 tokens and option-order sensitivity. Prefer 1.2 for new workflows. |

        Context includes state and questions. Oversize items receive errors rather than truncation. HTTP/Python `model:"auto"` requires both Laya English and Multilingual installed; the worker chooses English when at least 99.5% of the state's letters are ASCII, otherwise Multilingual. This is a character heuristic, not language detection: select Multilingual explicitly for ASCII Polish. Jev is TypeSafe's hosted model, not an Arsenal model ID. No local English-model superiority is established by the Polish validation below.

        ### Write the questions

        These rules adapt TypeSafe AI's MIT agent skill and its primitives/confidence/cookbooks guidance:

        - Put source, query, constraints and current facts in named JSON state fields. Put the judgment in `instructions` and answer meanings in `criteria`. Question IDs only identify answers for code; include the full meaning in the question.
        - Ask one coherent judgment per question. A long instruction may define context, distinctions and exclusions; independent dimensions belong in separate questions. Combine their answers in code.
        - `Choice` selects among competing options. Describe the distinctions and include a no-match option in every Choice (`other`, `none` or `unclear`). Keep the option set small enough to test; neither option count nor schema acceptance proves accuracy.
        - `Noul` returns P(yes) from 0 to 1, optionally with `criteria:{"true":...,"false":...}`. It measures probability of a proposition, not intensity. It is not inherently more reliable than Choice.
        - `Score` returns the weighted level index, 0 through `len(criteria)-1`, plus probabilities and a legend. Describe each ordered level as a concrete situation that stands on its own. Normalize different-length rubrics before combining them in code.
        - Ask independent questions about the same state together. They cannot read each other's answers. A second request is needed when an earlier result supplies new evidence or options.
        - Preserve probabilities. Choice/Score `confidence` summarizes distribution concentration, not correctness or permission. Arsenal's legacy Noul replies may also include `confidence`; use `noul` itself for P(yes), not that field as a separate quality estimate.
        - Code owns top-k selection, weights and thresholds. For semantic shortlisting, try ranking by the relevant option's probability and read the shortlist. Validate cutoff/ranking on labelled examples from the user's data, including negatives and boundary cases; a universal 0.5, 0.85 or 0.9 cutoff has no evidence here. `calibrate()` fits the provided examples: assess a chosen cutoff on separate held-out cases before relying on it.

        ### Calls and answers

        Single-item MCP example; `criteria` gives the meanings, not just labels:

        ```text
        arsenal_run(model:"laya-multilingual", input:{
          "state":{"stanowisko":"Serwisant","obowiazki":"Diagnostyka i naprawa maszyn u klientów"},
          "questions":{"kind":{"type":"choice","instructions":"Jaki to rodzaj pracy?",
            "criteria":{"serwis":"serwis, naprawa, konserwacja lub instalacja urządzeń",
                        "produkcja":"produkcja nowych wyrobów lub obsługa maszyn produkcyjnych",
                        "other":"inna praca albo brak informacji pozwalającej dopasować kategorię"}}}})
        ```

        For many items, use `arsenal_run(model:"laya-multilingual", operation:"judge", input:{"items":[STATE1,STATE2],"questions":QUESTIONS})`; the answer has one line per item. Reuse the same questions for the whole pile. Scripts use the raw batch endpoint:

        ```sh
        curl --fail-with-body -sS "$ARSENAL_URL/v1/judge" -H 'Content-Type: application/json' \
          -d '{"model":"laya-multilingual","items":[{"opis":"Eksport nie działa"},{"opis":"Dziękuję, problem rozwiązany"}],"questions":{"bug":{"type":"noul","instructions":"Czy opis zgłasza nierozwiązany problem?"}}}'
        ```

        Single state: `POST /v1/systemone` with `{"model":ID,"state":STATE,"questions":QUESTIONS}`. Both also exist under `/providers/typesafe/v1/...`; TypeSafe SDK clients use base `$ARSENAL_URL/providers/typesafe` with dummy key `local`. Universal equivalents: `POST /arsenal/v1/models/ID/ops/decide` or `/ops/judge`, omitting `model` inside the input so the endpoint's model answers.

        Single responses contain `answers`, `model`, `usage`. A Noul answer has `noul`; a Choice has `choice`, `probabilities`, `confidence`; a Score has `score`, `probabilities`, `confidence`, `legend`. Batch responses have `results` in input order, each with `answers`, `model`, `ms`, or `error`. Keep errors separate from low scores; verify one result per item before applying decisions. A request with `bits` requires that loaded precision and returns 409 on mismatch; it never changes the shared dtype.

        d1-3B image input is `images:[{"path":"/absolute/picture.png"}]` beside `state` and `questions`; use `state:null` for an image alone. Batch `images` is a list of image lists matching the order and count of `items`. No image URLs or audio/video. For reproducible cross-interface probabilities, use consistently ordered inputs: MCP sorts object keys, while raw HTTP preserves received order; the vendor renders that order into its prompt. Arrays retain order. This model uses LFM Open License v1.0: commercial use is subject to the US$10M annual-revenue threshold; inspect the pinned terms. It is a decision model and does not generate prose.

        ### Measured weak spots

        d1-3B runs a Swift port of LiquidAI's System One prompt and readout over an MLX bf16 forward, pictures included, in the native worker. Its historical 8 Oct gate compared it with pinned vendor code run in float32 on the CPU: maximum absolute probability difference 0.0039 and 24/24 answer argmax agreement across Noul, Choice, Score, batch and image inputs (M5 Max, 8 Oct 2026). That retained result proves fidelity on those cases, not job-ad accuracy. New work follows the quality gate in the repository's `AGENTS.md` (Precision and the quality gate): independent GPU half-precision references or vendor-published outputs; the historical CPU/FP32 route is not a procedure to repeat. The earlier torch MPS runner and its reported 0.0 drift (MPS compared with itself) are superseded: torch 2.12's MPS causal attention is wrong in bf16 beyond ~100 tokens, which moved some answers by up to 0.73 in probability.

        Fresh comparison, same chip/date: d1 bf16 on MLX versus Laya Multilingual fp16 Fast. Inputs and manual labels were saved before inference: 32 pracuj.pl ads (16 service/installation), plus 24 sections mentioning SEP (11 mandatory). Both used the exact questions retained in the probe. Polish job Noul got 24/32 versus Laya 17/32 (the historical vendor FP32 reference got 25/32; an earlier 27/32 came from the faulty torch MPS kernel); six-way Choice winning labels got 24/32 versus 23/32, finding 15/16 versus 12/16 service jobs. Ranked top 16 by Choice found 12/16 versus 11/16. Packaged HTTP timing on M5 Max, 8 Oct 2026, Mac in use, not a quiet benchmark: one decision (one ad, three questions) about 48 ms (the earlier torch runner: 123 ms); 32 ads × 3 questions 40 ms per ad (103); 32 ads × 1 question 24 ms per ad (80); a one-line state with one question about 21 ms; Laya about 12.6 ms per ad. d1's worker holds about 6.5 GB; a first call that has to load it takes about 6.4 s.

        For full requirements, SEP presence got 24/24 versus Laya 20/24, but every section mentioned SEP, so false positives were not tested. Mandatory-SEP Noul got 20/24 versus 11/24, detecting 8/11 mandatory cases versus 9/11. SEP Choice got 13/24 versus 19/24 and found 0/11 mandatory cases versus 9/11. Excerpts did not rescue d1's Choice (0/11); its mandatory Noul got 12/24 and found 4/11. Use d1 for images/longer context and evaluate focused quality alternatives; keep exact text checks and source reading for requirements. This diagnostic sample does not establish a general quality winner.

        Validation (pre-release build): `laya-multilingual` fp16 Fast, M5 Max, 7 Oct 2026, real Polish pracuj.pl ads. A manually labelled 32-ad subset had 16 service/installation jobs and 16 other jobs. English Noul at 0.5 got 16/32 right; Polish Noul got 17/32: both answered yes for 30–31 of 32 ads, so a yes/no question here sorted nothing. Polish six-way Choice with a no-match outcome got 26/32 right by winning label. Sorting its service probability put 14/16 true service jobs in the top 16; a fixed P(service)>0.5 retained 10/16 and classified 25/32 correctly. This supports Choice plus source review for this shortlist, not universal calibration or perfect exclusion.

        On 24 full requirement sections, a SEP-name presence Noul got 20/24 right. Distinguishing mandatory SEP from preferred/obtainable/absent SEP was much worse: Noul 2/24 (yes for all 24; answering "no" every time would have scored 22/24), Choice 8/24; on shorter excerpts, 3/24 and 7/24. There were only 2 mandatory positives. Language change and shorter text are candidate tests, not guaranteed fixes. Use exact text checks for presence and inspect the wording for mandatory versus optional requirements. The Typed model was not installed at validation, so no Multilingual-versus-Typed quality comparison was run.

        Other traps: negation, sarcasm, niche terminology, manipulated source text and knowledge outside the item can produce confident errors. Inspect representative kept/dropped items and boundary cases before closing a triage task. Keep enough candidates for review; do not silently discard an item that failed inference.

        Sources: the 7 Oct validation inputs, labels and question definitions were a private sample of public job listings and are not distributed; those historical counts are separate from the fresh comparison above. A 200-listing triage run on the same app, model, chip and date took 0.6–1.3 seconds; that is task evidence, not a benchmark or latency promise for other inputs. New correctness calls used a GPU lock, not a quiet performance measurement. TypeSafe guidance: https://docs.typesafe.ai/primitives · https://docs.typesafe.ai/confidence · https://docs.typesafe.ai/cookbooks/parallel_questions · https://docs.typesafe.ai/cookbooks/rerank_typesafe. Arsenal-specific quality results do not transfer to hosted Jev.

        ## TypeSafe attribution

        Decision-question guidance adapted from TypeSafe AI's official `typesafe-ai` agent skill (MIT), Copyright (c) 2026 TypeSafe AI. https://docs.typesafe.ai/agent-skill

        Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

        The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

        THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
        """##,
        "ADDING.md": ##"""
        # Arsenal: adding and maintaining models

        Read this when the user asks to add a Hugging Face model, port a new architecture, optimise a model, or drive Arsenal's folder tools. Running catalog models is in [SKILL.md](SKILL.md).

        ## Model settings

        Inspect `arsenal_settings(model:ID)`. Set with `arsenal_settings(model:ID,set:{SETTING:VALUE},apply:false)`; `dtype`, `keep_hot`, `show_in_menu` and `runner` are set the same way, and the answer says which keys applied; request settings affect the next call, load settings remain pending until Reload (`apply:true` reloads explicitly). HTTP: `GET /arsenal/v1/models/ID/settings`, `POST` with `{"values":{SETTING:VALUE},"apply":false}` takes the same keys, `dtype`, `keep_hot`, `show_in_menu` and `runner` included, with the same checks. Changing a setting affects other clients too.

        ## Scripts and folder tools

        For scripts, `GET $ARSENAL_URL/arsenal/v1/catalog/browse` lists the catalog, `GET /arsenal/v1/models` the user's table, and `GET /v1/models` the same table in OpenAI form. Every operation accepts its input object at `POST /arsenal/v1/models/{id}/ops/{operation}` with no auth (MCP needs the bearer token; browser Origins are refused); `GET /arsenal/v1/openapi.json` carries live schemas; HTTP JSON is limited to 1 MiB, so chunk a large pile by bytes as well as item count. Control calls: `POST /arsenal/v1/load` with `{"id":ID,"dtype":DTYPE}`, `/unload` with `{"id":ID}`, `/keep-hot` with `{"id":ID,"on":true}`. Dtype and Keep Hot are shared settings: restore changes after temporary work, and prefer on-demand inference to loading by hand.

        The installed folder tools are `Arsenal.app/Contents/MacOS/Arsenal catalog` and `Arsenal.app/Contents/MacOS/Arsenal runner-check`. Find the app with `APP=/Applications/Arsenal.app; [ -d "$APP" ] || APP="$HOME/Applications/Arsenal.app"`, then `CLI="$APP/Contents/MacOS/Arsenal"`. The support folder is `SUPPORT="${ARSENAL_SUPPORT_DIR:-$HOME/Library/Application Support/Arsenal}"`; pass `--support "$SUPPORT"` rather than writing the default path. They run headlessly while the menu-bar app is open. Catalog arguments and runner-check options follow the subcommand; see [Model folders](../docs/MODEL-FOLDER.md) and [Runners](../docs/RUNNERS.md). The checker launches runner code in a separate child process; validation does not grant local-code consent.

        The installed precision drainer is `Arsenal.app/Contents/MacOS/Arsenal stats`: `stats list --support DIR` reads the held queue; `stats seed --support DIR --models MODEL_ROOT` introduces entries without inference. Only the owner's current run approval allows `stats run --support DIR --models MODEL_ROOT --source-support APPROVED_WEIGHT_STORE --go`. Helpers and signed Resources resolve from the app bundle. Use a writable model evidence tree matching accepted revisions. The drain clones source weights into per-model isolated profiles, uses ordinary Get/Load/operations and deletes derived snapshots afterwards; it never downloads. Ordinary memory admission runs before Get/Load, refuses swap, and leaves insufficient-memory entries pending with a skipped reason while continuing. See [Benchmarks](../docs/BENCHMARKS.md) for refusals and memory boundaries.

        ## Add a model from Hugging Face

        The user names a Hugging Face model (a link or `owner/repo`) and wants it in Arsenal. Follow `arsenal://integration/ADD-A-MODEL.md` (in the Arsenal repository, `docs/ADD-A-MODEL.md`) from the top: it is the complete procedure, with the kit commands, the MLX porting rules and the checks. The kit (`kit/new-model.py`) ships inside the app at `Arsenal.app/Contents/Resources/kit`; run its commands from `Arsenal.app/Contents/Resources` (a repository checkout has the same `kit/` at its root). Every added model runs on MLX in Swift; Python never serves one. The user's yes comes before the download, before Arsenal runs the new code, and before adding a model whose licence is not open, on its own terms and for this Mac only.

        ## Introduce a new architecture

        For a model whose architecture no bundled runner serves, the work is a new runner: a Swift runner on the pinned mlx-swift, checked against the provider's own code. The reference documents are `arsenal://integration/ADD-A-MODEL.md`, `arsenal://integration/ADDING-A-MODEL.md` and `arsenal://integration/INTEGRATING.md`. Catalog models are qualified with the intake procedure in `arsenal://integration/ADDING-A-MODEL.md`, which also covers rechecking a model after a change. Downloads, long GPU runs and installs need the user's yes from step 0. The procedure below describes the complete introduction, not permission to skip that confirmation.

        For an implementation/run request, make a correct, gated and usable standard model within twenty minutes from the original request. Treat this as the introduction target; optimize later. An existing kind alone does not prove architecture compatibility. Record actual elapsed phases and disclose a concrete mismatch/transfer constraint when it threatens the target.

        0. **Choose the source and confirm it with the user before any download.** A link names the model the user wants, not the files to fetch.
           - Inventory every component the model needs (prompt/text encoder, generator, denoiser or language model, VAE/decoder, vocoder, tokenizer/processor) and where each one lives. A repository can hold only part of a model, e.g. a fine-tuned text encoder for an image model; name the missing parts and their source.
           - Prefer a release built for MLX: the vendor's own MLX repository, mlx-community, or a community MLX port (names like `*-mlx-bf16`, `*-mlx-8bit`). These are made to run on Apple silicon. Next best: the vendor's original safetensors. A GGUF repository points at the model; look for the MLX release of the same model and use that.
           - **Keep the weights intact, in FP16 by default.** Use the chosen release as shipped, without re-quantizing (an mlx-community int8 stays int8). A bf16 release is cast to fp16, because M1 and M2 Macs have no bf16 arithmetic; check the fp16 parts against the original on fixed tensors and keep bf16 only where fp16 fails. The work is making the model run.
           - When the MLX release differs from what the user linked (a different fine-tune or edit, another quantization, a 32-bit part), say what differs and recommend one. The user picks.
           - Then send the user a short plan: each component with repository, revision, format, precision and size; total download; licence; the engine (bundled worker, wrapped established engine, or new runner) and the work that needs; the precision each part runs at (as shipped). Wait for their yes. Only an explicit "just do it, no questions" skips the wait; then state the plan and proceed.
        1. Research pinned vendor code/model card, licence, architecture, exact files and an independent reference route using a different implementation on the GPU at native half precision (bf16/fp16): the vendor's own code or an established engine, or the vendor's published outputs. Search local stores first. Name any approval-bound download or expensive reference cost before running it.
        2. Inspect live operations and a working folder. Read `arsenal://integration/MODEL-FOLDER.md` for declarations and `arsenal://integration/RUNNERS.md` when implementing a runner. Reuse a kind only when its semantics fit; otherwise embed a `local.*` kind with complete input/output schemas. Pick the first runtime that fits:

           1. The bundled native worker, when it already runs the architecture: configure the folder without new code.
           2. An existing MLX implementation in Swift (mlx-swift / mlx-swift-lm, an mlx-audio or mlx-community Swift port), built into a Swift runner using RUNNERS.
           3. Our own Swift runner on the app's pinned mlx-swift, ported from an MLX Python implementation (mlx-lm, mlx-audio, mflux, an mlx-community port) or the vendor's code as the reference.
           4. A mature native engine for the Mac (llama.cpp and the like), wrapped as-is behind ArsenalWire, only when no MLX implementation of the model exists.
           Python never serves a model. It runs the provider's own code, exactly as they document it, as the reference that an MLX port is checked against. For a different version than the provider's (a fine-tune, an edit, a conversion), look for the instructions its makers or users publish for running it and use those.

           Record the selected engine and why earlier choices do not fit. Custom kernels and tuning need the owner's per-model go after the precision first pass in **Optimize later**.

        3. Write the pinned folder, reviewed executable/dependency delta, README limits and frozen fixtures, and its settings panel from `kinds/<kind>/settings.toml` (MODEL-FOLDER, **Settings panel**). Fill in the full model information now, before any approval: the full official name with family and version (Ming-3.0-tts-tiny is "Ming 3.0 TTS Tiny"), total parameters from the model card (and the active count for a mixture of experts, in `description`), `summary`, `description`, `purpose`, `good_for`, `limits`, the languages this runner actually speaks with whether it speaks Polish, `[licence]` and `[facets]` modalities. Approval binds every byte of the folder, so any later edit asks for a new yes. `good_for`, `limits`, `purpose`, `[facets]`, ops and runner fields are also in the implementation fingerprint: after a change there, approval is refused without independent-reference parity evidence. Validate with the existing folder validator. Review code, dependencies and reference independence before approving the current digest. Prepare the folder in an isolated profile for qualification.
        4. Get/Load and run the existing checker, exact output count/order/error/cancel checks, component parity against the reference on small fixed tensors and one end-to-end smoke. Precision tiers, the intake quality pipeline and speed work come later, on request (**Optimize later**); a user who says "install it and let's test it" wants it running, not tuned. Retain fixtures, full actual/expected outputs, thresholds, commands, environment/revisions and hashes under results. The smoke doubles as the measurement run: record speed (real-time factor for speech and audio, tokens per second for text, seconds per request otherwise), cold and warm load time, peak memory and swap before and after in `results/`: a `quick-check-1` record under `results/quick-checks/`, a schema-2 table file bound to it, and a short `<chip>.md` with every number and what was not checked (layout: `models/chandra-ocr-2/results/`). A schema-1 result for the revision gate uses profile=quick, measurement_lock=false. Missing reference means parity=not-run and incomplete qualification, not a green claim.
        5. Atomically place the final folder at `<support>/local-models/ID`, preserving rollback outside the scanned root. `arsenal_settings(action:"refresh")`, inspect the digest `arsenal_model(model:ID)` prints, then `arsenal_settings(model:ID,action:"approve_code",content_digest:DIGEST)`. Add, download and load with `arsenal_add`, `arsenal_download` and `arsenal_load`, using cached verified assets. The named implementation request authorizes this destination; exploratory work remains isolated. No app reinstall or catalog signing is needed.
        6. Rediscover operations; exercise HTTP JSON and MCP results/resource retrieval, table, activity and native tool through automated paths. Keep model-specific examples/limits in `arsenal://models/ID/README`. Return the usable operation, outputs, current checks and missing evidence; leave the requested model ready and clean only your disposable artifacts. A model is not added until its full name, parameters, description, good_for, limits, languages (Polish yes or no), licence, modalities and measured speed, load time, peak memory and swap are all filled in; report a missing one as missing.

        The installed integration references are read-only MCP resources. Schema-3 files use inbox path objects at x-arsenal-file annotations, and results retain model metadata plus URLs, hashes and byte counts. `arsenal_model` lists each operation's inputs; full input and output schemas are the `arsenal://models/ID/operations/OP/schema` and `/output-schema` resources. The app process stays running during folder updates.

        ## Optimize later

        Keep optimization separate from introduction. First report fp16/bf16 versus int8 versus int4 for that model: FLOPs and bytes per step against this Mac's compute and memory-bandwidth ceilings, a short check of speed, peak RAM and quality delta, and a one-line recommendation. Verify components on small fixed tensors against a half-precision reference captured once and reused; use at most one end-to-end smoke of the final default path. Wait for the owner's per-model go before custom kernels or tuning. Then optimize a revision of the same model ID in an isolated profile, with the old production revision continuing to serve. Gate the candidate against an independently invoked pinned reference; retain current protocol/parity evidence and speed=pass for a short speed check with the inference/fixture fingerprints and retained evidence hashes. Mark reference.independent=true only after inspecting the independently invoked reference path. Review the executable/dependency delta and approve the final digest through MCP after Refresh.

        Approval stages the optimized revision and Arsenal swaps it automatically at the next idle window with no request running or queued for that model. No further agent action or app restart is needed; arrivals during the worker swap wait briefly. If unloaded, the next Load uses the approved revision. Failed hello/load restores the prior worker and reports the error. Inspect row/activity revision.active/pending/previous and its error to confirm the handoff. Use `arsenal_settings(action:"rollback",model:ID)` for one-step rollback. Evidence-only edits change consent without restarting a busy worker.

        Multi-GB downloads, long GPU runs and controlled benchmarks retain their separate resource authorization. Approval of code or a Get plan does not create that authority.
        """##,
        "ADD-A-MODEL.md": ##"""
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
        """##,
        "MODEL-FOLDER.md": ##"""
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
        """##,
        "RUNNERS.md": ##"""
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
        """##,
        "ADDING-A-MODEL.md": ##"""
        # Model intake

        The short path to a working model (16-bit only, a folder dropped into `local-models/`, no reinstall) is [ADD-A-MODEL.md](ADD-A-MODEL.md). This page is the optional qualification that follows, on request.

        Intake starts from a fully described and measured folder ([skill, Introduce a new architecture, steps 3 and 4](../skill/ADDING.md); [ADD-A-MODEL.md](ADD-A-MODEL.md)). Intake's table figures replace the quick figures; they never stand in for the description.

        Intake starts after the source is chosen and the user has confirmed the plan: [skill, Introduce a new architecture, step 0](../skill/ADDING.md) (prefer a 16-bit MLX or original safetensors release over GGUF or pre-quantized files; inventory every component; ask before downloading). To recheck a model after a recipe, runner or kernel change, rerun this intake; a changed identity already marks old outcomes unchecked, and saved captures can be re-scored without the GPU (below).

        Agents run intake in a worktree against a copy of the model recipe under ignored `lab/models/<id>/`, using pinned weights already on this Mac and a fresh isolated lab support directory. Set `MODEL_FOLDER` to that lab copy; promote the final recipe, table results and quality binding after review. Never change the installed model store, download a second copy, or install from this procedure. Folder and runner contracts remain in `MODEL-FOLDER.md` and `RUNNERS.md`.

        1. Inventory every component, including prompt encoders and decoders. Pin repository, revision, original hashes and sizes. Use the native 16-bit release; accept FP32 files only where no half release exists, convert those files before executing the model, and preserve that limitation in the reference identity.
        2. Build `arsenal-stats` and the native worker through `nice -n 10 scripts/mlock work`, using the project's CLT toolchain and pinned no-fast-math MLX metallib. Put the worker, family symlinks and resource bundle in a lab helpers directory.
        3. Obtain current authority for short model runs, then run the adapter through `scripts/mlock gpulock`:

        ```sh
        scripts/mlock gpulock model-intake -- .build/debug/arsenal-stats intake \
          --model "$MODEL_FOLDER" --source "$EXISTING_PINNED_SNAPSHOT" \
          --helpers "$LAB_HELPERS" --support "$FRESH_LAB_SUPPORT" --go
        ```

        Get APFS-clones source files into the lab and verifies their original pins. It casts one verified file at a time, records output hashes separately from source hashes, and collects unreferenced originals. The tensor writer shifts its payload backwards in the same staging inode to insert the safetensors header; it never writes a second payload-sized temporary file. Capacity checks cover source plus output and a header margin. Kernel-held staging leases protect active Gets; the next Get removes abandoned leased staging. Unmarked staging from older versions is reclaimed only after 24 hours. Failure and cancellation remove partial multi-file casts without committing a snapshot. The source must hold regular files: a Hugging Face cache snapshot is symlinks into its blob store, which the intake rejects, so first APFS-clone it dereferenced (`cp -cL`) into a lab `tmp/` directory and delete that copy afterwards.

        For precision qualification, use `measurement_components` on the loaded production worker. Save integer token IDs and encoder masks, reference conditioning, exact reference latents at three separated timesteps, timestep/position inputs, and decoder input **once**. Save the original reference outputs and one A/A repeat. Candidates independently consume those same saved inputs; candidate encoder output never feeds the denoiser gate, and candidate denoising never feeds the decoder gate. Validate exact keys, shapes and element counts before values. Hash the fixture and check it after every candidate; make its archive read-only.

        Freeze separate bounds before evaluating candidates. Native 16-bit source comparisons retain their named A/A-repeat policy. **FP32-source qualification** uses the shared `IntakeComponentGate` policy `fp32-source-bf16-nearest-truncation-fp16-ulp-v1`: convert original source weights to BF16 round-to-nearest and, separately, BF16 truncation toward zero; execute both on identical saved inputs on the GPU. Their per-component relative L2 and cosine differences are the source-rounding floor. Save a separate exact BF16 A/A repeat as the reproducibility check; it never substitutes for that floor. Before executing any FP16 candidate, freeze relative L2 ≤ twice the perturbation floor plus one **FP16** ULP (spacing at each reference value's exponent, bounded below by the FP16 subnormal spacing, combined in L2); cosine ≥ `1 - 2*(1-floorCosine) - ulpRelativeL2²`. Require finite outputs/activations, no new clamps and candidate FP16 range |x| ≤65,504, rather than reference activation extrema. Report output counts that become FP16 zero or subnormal where BF16 is normal, as diagnostics; these counts do not prove every intermediate avoided underflow. If FP16 fails, select BF16 and judge both quantized tiers against BF16. Source FP32 is read only for conversion; no FP32 model executes. Freeze immutable bounds with policy and capture identities before the candidate; retain historical outcomes under their original named policies. FLUX records pre-clamp residuals so a later clamp cannot hide overflow.

        **Quantized tiers are optional, never required.** A model ships at its 16-bit format only; an int8 or int4 tier is derived and qualified only when the owner asks for it. Existing quantized tiers stay until he says otherwise.

        **Quantized-tier policy:** on each component's saved identical inputs, require relative L2 ≤0.05, cosine ≥0.999, all finite, no new clamp events, and all activation maxima within the selected 16-bit format's range (FP16 65,504; BF16 approximately 3.39×10³⁸). A 16-bit rounding floor would reject intentional quantization error by construction; these are the coordinator's “close enough” quality bounds, not a requirement for 16-bit parity. Both policies assert exact output/activation keys, shapes and counts first. Code owns the constants in `IntakeComponentGate`; each evidence run stores its bounds and hash. Independent vendor qualification remains separate.

        Saved captures can be re-scored without weights, workers or GPU execution: `.build/debug/arsenal-stats intake --rescore --model "$MODEL_FOLDER" --run "$SAVED_RUN_ID"`. This creates a new result and bounds directory linked to the original capture run; it never rewrites the captured run or re-times it.
        Capture one final end-to-end sanity artifact per precision. Save one image SSIM or audio log-mel diagnostic per tier with model-specific reference tooling in the lab. Inspect image structure or audio plausibility separately; any pixel, waveform or perceptual distance is diagnostic and cannot select FP16/BF16 or qualify a quant. Matching seeds alone do not ensure identical noise across activation dtypes. Do not turn that difference into a quality verdict.

        Time the production Get/Load/operation path, with conversion outside warm timings. Keep cold Load latency (including worker self-tests) and the first request separate. Unload capture workers and start fresh ordinary serving workers for every model kind. Those workers must never receive `measurement_components`; record their PIDs and capture-request counts and reject any contaminated peak. Warm each resident arm once, then run balanced `ABBA BAAB` windows in one session, with the original baseline and one candidate resident. Each reply follows evaluated output and usable-file delivery. Report all four samples per arm, median, min/max spread and early-to-late baseline drift. Default workloads are FLUX 1024×1024/four steps and Stable Audio five seconds/eight steps. Busy-machine figures remain `quickCheck`; overlapping spreads or baseline drift do not establish a speed gain. Physical-footprint peaks are per worker, covering Load and execution, not the sum of both resident arms and not packed file size.

        The intake guard caps registered workers at the smaller of 60 GB or 65% of initial available headroom, refuses a second worker without sufficient current headroom, and kills only its own workers on any increase above initial system swap or at the 590-second run ceiling. Existing swap is recorded separately; this procedure must add none. Every worker request also has a bounded deadline. Cloned weights, donor files and derived caches are removed on exit; saved inputs, outputs and logs remain as local evidence.

        Each run has a timestamp plus UUID. Policy, fixture, bounds, component captures and snapshot receipts live under `gate/evidence/intake/<run>/`; the result is `results/intake/<run>.json`. A failed run retains its own identity. Table stats must link their exact run and bounds hash; a later run never replaces another run's evidence. All per-run captures and rendered QA artifacts stay in ignored `lab/`. Commit current source, test fixtures, table statistics and the current compact quality binding. Never reuse a result after changing its recipe or fixture.

        Voice intake adds `--voice --worker-build <actual-build-configuration>` to the same command for Breeze, Qwen3 Base/VoiceDesign, Pocket and Kokoro. It uses `MeasurementWorkload.request(.voices)` unchanged (streamed English sentence, seed 0; Base's standard tone reference), the ordinary default Load configuration, fresh serving workers for physical peaks and balanced warm windows. The separate `format-trial.json` retains the rejected FP16 candidate; the completed result has only the chosen half, int8 and int4 rows. Incomplete/prototype receipts stay under `gate/evidence/`, outside validated table results.

        Int8 voice fallback uses `intake-v3:<model>:<matrix selectors>`: begin with every eligible matrix packed, identify failing component owners on saved inputs, then re-quantize groups and individual runtime projections with all other owners packed (fused Q/K/V projections stay atomic). An ablation worker holds a separate selected-half donor and swaps only the implicated matrices. Diagnostic trials replay every selection case of changed owners; unchanged owners retain their known passing evidence on immutable component inputs. Identical keeper sets reuse their saved trial. Scale fingerprints must be unique in that inventory and every target must map to a real loaded module; passing fused codec matrices need no ablation mapping. Repeat single-projection removals until none passes, retaining each trial and its unchanged bounds. This is a locally irreducible recipe for those fixtures, not a proof of the globally smallest subset. To qualify an existing conservative candidate without further removals, supply its model-local selection inventory with `--selection-seed-coverage` and `--fixed-int8-selection`; provenance is frozen before the run and no minimum is claimed. Passing owners stay packed; only owners failing the current all-packed selection gate may retain matrices from that inventory. Freeze both selection and distinct held-out inputs and bounds before search. Removal trials use selection data only. Optional tighter selection-only safety bounds are frozen separately; final selection/held-out qualification still uses the shared 0.05/0.999 policy. Derive the final stored recipe and load it through ordinary Load to qualify every held-out component (different texts/token IDs, longer prefixes/caches, conditioning and codec inputs). The completion binding retains both fixture identities, exact counts, bounds and verdicts; a searched recipe cannot bind without distinct held-out evidence. Every fixed recipe (Laya, d1, FLUX, Stable Audio, Von and RE-USE), including its baseline row, uses the same `IntakeQualification.freeze/complete` boundary with its own frozen format or quantized bounds, complete component evidence and ordinary Load, explicitly `searched: false`; it makes no held-out claim. Every completion writer requires that qualification before publishing. Current table results cannot remove qualification to downgrade validation: catalog validation recomputes each component verdict from its finite flag, activation verdict, relative L2 and cosine versus the frozen bounds, then checks aggregate, row and table verdicts. `intake --reconcile-qualification --model <folder> --binding <model-relative-binding>` can re-emit sufficient retained evidence without model execution, preserving original measurements, failures and capture/build/load provenance. Missing captured input/output identities, frozen bounds or ordinary Load requests require a new isolated run; historical receipts remain separate. A failed held-out attempt cannot drive removals: continue only on selection data and qualify against a new untouched held-out generation. Historical schema-2 voice completion bindings qualify only their selection set. Discard capture/ablation workers before fresh serving figures. Coverage records both extra fallback parameters and total 16-bit remainder, including structural skips. Previous whole-component recipes and A/A-floor format outcomes remain historical.

        Voice packs include all eligible matrices in text/conditioning models, generation backbones/heads and codecs. Adapters preserve Breeze's special text end-of-instruction vector and split its batched output heads; Pocket's standalone speaker projection becomes a stored Linear; Qwen Base's large ungrouped 1×1 speaker convolutions use stored per-position Linear packs. Spatial/grouped/causal convolutions without that adapter, small tensors, norms and incompatible group widths stay at the selected half with per-tensor reasons. Saved Qwen predictor KV states are copied before later cache updates; candidates replay those exact states and token embeddings. Stock/cached/fused Breeze depth traces share last-token layer shapes. Pocket names the transformer/upsampling boundary and separately checks SEANet waveform output on saved normalized inputs under `saved-unit-rms-floor-0p01-seanet-v1` (RMS floor .01), with the predeclared shared bounds. Kokoro also checks its pinned BART phonemizer: tensors are rounded before module construction and eligible projections packed one at a time at ordinary Load, without retaining source FP32 learned arrays; its merged audit and recipe identity cover that auxiliary path. End-to-end common-prefix spectra are diagnostic only and retain both sample counts; they never select precision.

        Laya (English, Multilingual, Typed) keeps its catalog recipe `mlx-affine-g64:verdict-at-load`: Get makes the derived view from the FP16 files and the decisions worker quantizes at load every Linear layer and the token embedding (read in row blocks straight from the file, so no dense table exists); norms, biases and small tensors stay FP16. Its `measurement_components` captures, for a yes/no row and a two-option row, the embedding, every encoder and head layer and the scorer, plus three served-path probability vectors (max |dp| ≤0.017, unchanged answers). A model is graded only on inputs in its supported languages: the intake sends English states to English-only models and a German second state to Multilingual, and records them in `policy.json`.

        Historical voice `intake-v2:<model>:<sorted-half-components>` recipes used whole-owner fallback and selection-only replay. Those outcomes remain historical; current intake uses the v3 selection/held-out procedure above. Int4 packs every eligible part and is declared only where its quality check passed or was never run; a failed int4 is removed.

        Pilot recipes: FLUX `mlx-affine-g64:flux2-klein-4b` packs transformer and Qwen matrices/embedding; the small convolutional VAE stays half. Stable Audio `intake-v1:stable-audio-3-medium` packs DiT, T5Gemma matrices/embedding and SAME-L matrices. Its old `mlx-affine-g64` recipe packed only DiT and cannot inherit the new results. Eligible weights are matrices with at least 65,536 parameters and input width divisible by 64; vectors, norms, convolutions and smaller matrices stay half. This cutoff is an engineering choice for these pilots. Loading installs stored Linear/Embedding packs directly, without evaluating a complete dense model first.

        FLUX FP16 must pass its format gate against original BF16 before replacing it; the pilot failed (reference pre-clamp maxima 98,816; candidate non-finite residuals), so FLUX stays BF16 and its quants derive from BF16. Whole-part int8 passes the quantized-tier policy; int4 fails.

        Stable Audio's FP32-only vendor SAME-L source is cast before execution. The coordinator accepts its FP16 decoder on existing finite/no-clamp evidence and round-1 parity against retained original PCM (spectral relative L2 0.0157; waveform relative L2 0.0212), without an FP32 reference run. That is the accepted cast basis, not independent vendor component parity. SAME-L now anchors its entire decoder stream to the stored half dtype; new audio evidence must cover that arithmetic change. Whole-part int8 is enabled only with the passing saved-input gate; int4 is declared only where its quality check passed or was never run.

        Upgrades never convert installed weights during catalog refresh, table selection or Load. A changed recipe makes the old snapshot stale; the normal **Get** action performs the documented in-place, file-by-file cast/derivation and records new pins. An old Stable Audio DiT-only int8 snapshot cannot load as the whole-part recipe. Get the new FP16 base before deriving from it (Get int8 does both). Existing source files are not silently reinterpreted.
        After qualification, enable only the appropriate recipes in metadata and write schema-2 `PrecisionStatsResult` JSON for the table. Include chip, GPU core count, date, revision, worker hash, workload, component bounds, recipe and immutable run identity. Mark every provenance object from the default workload with `defaultWorkload: true`; these figures take precedence over historical non-default workloads even on another chip, and missing default-workload figures stay missing. Record `stats.qualityCheckPassed` separately from runtime availability (absent means unchecked). int8 is offered unless its quality check failed (an unchecked int8 is offered without a mark; a failed int8 is greyed with its reason); int4 is declared only where its quality check passed or was never run; `ModelRow.precisionQualityWarnings["int4"]` says when it has not run. Retain the failed gate, bounds, numbers and per-precision speed/peak figures; removing a failed tier does not discard those metrics. The catalog generator includes these schema-2 result files and their hashes; integration regenerates and signs the catalog. Run the touched suites and commit; integration, installation and publication remain separate actions.

        FP32 kernel accumulators and numerical intermediates (such as RoPE angles and d1’s few option-logit dot products, which preserve near-tie probabilities) are allowed; learned parameters and other component matmuls use half precision or stored affine weights. Small SFX uses `cast:stable-audio-3-small-sfx` at Get and whole-part `intake-v2` int8/int4 derivation. Version 2 also packs the custom duration projection (`cond.seconds_total_weight`, stored under a module-key alias), keeps that projection half/affine and checks it as a sixth independent component. Learned memory-token values stay half because the model concatenates them directly; they do not feed a matrix multiply or embedding lookup. Medium retains its original conditioner arithmetic. Load rejects the unconverted FP32 SAME-S source; half/packed parameters and the decoder stream stay half. Its FP16 DiT path skips the legacy FP32 self-test and promoted-weight fallback. The shared FP32-source policy qualifies the initial SAME-S, duration and padding FP16 casts internally; its separate served FP16 A/A check establishes reproducibility. FP16 is selected, int8 passes and int4 fails its component gate. The retained original PCM bridge is diagnostic; independent vendor half-GPU/component qualification and perceptual equivalence remain unavailable. Quantized tiers are checked against the selected FP16 component inputs. An interrupted Small SFX intake may use `--reference-run <run>` only with an identical FP16 snapshot: it retains the original fixture/reference/A/A floor and replays the current FP16 implementation plus both quantized candidates against them; `reused-reference.json` records the source run. Coverage receipts are rebuilt for each derived tier, never copied from the read-only prior snapshot.

        Quality outcomes carry `stats.qualityIdentity`: SHA256 of the current derive recipe, runner declaration and implementation identity, plus the model folder’s kernel manifest, kernel/header contents and effective load settings where declared. These identities are per model; another model’s folder changes do not invalidate them. Bundled native identities are source-closure receipts (`scripts/quality-identities.py`, shipped in Contents/Resources), scoped per family; folder runners use their executable digest. Missing/mismatched identities turn a pass, and a failure bound to an older identity, into unchecked (stale evidence says nothing either way; unbound legacy failures follow their recorded checks), without discarding historical measurements or runtime failures: int8 stays offered and int4 shows the not-yet-checked warning. The current binding lives in each model’s `gate/evidence/quality-binding-<date>.json`. These receipts justify the saved outcomes against the current recipe and source closure; they are not new measurements. Decision intake uses the same default Load configuration builder as RealModelsService, records the exact requested configuration and worker-reported selected kernels/fallback reasons, and writes the binding itself from the exact completed run, worker, bounds and emitted stats (also retained inside that run). Intake rollout must write this identity after checking the final recipe/build; old unbound outcomes never qualify the warning away.

        ### Approved source downloads and FP32-only rollout models

        The default procedure remains local-source-only. With explicit current approval for Von 1.1, Von 1.2 or RE-USE, `arsenal-stats intake --download-source --model <folder> --helpers <lab-helpers> --support <fresh-worktree-tmp-lab> --go` runs ordinary Get with an exact hash/size allowlist and no local donor stores. Its converter APFS-clones the verified originals into `pinned-source` before the production half cast; this source exists only for the approved intake. The command deletes the temporary model store and retains the original pinned files plus `download-receipt.json`. Pass that regular-file directory as the normal intake `--source`; remove it after verification. Nothing reads or writes the installed profile.

        These adapters calibrate the shared FP32-source policy on GPU BF16 nearest, BF16 truncation and one exact A/A repeat before freezing bounds and executing any FP16 model. Component inputs are saved once and replayed independently, including token IDs, masks and Von option positions/markers. Learned parameters and matrix multiplications stay half; Von probability calibration and RE-USE scan/FFT numerical accumulators may use FP32. Independent vendor parity remains separately identified.

        Von's half converter casts its scorer and final norm as well as the encoder. Both pinned Von revisions and RE-USE failed their frozen FP16 source-format check in this rollout, so they declare BF16 as the native tier (`cast:von-bf16-2` for Von; `cast:re-use` with BF16 for RE-USE). Von's quantized tiers follow BF16: `mlx-affine-g64:von-all-v2` streams every eligible Linear and the token embedding (row blocks), including the scorer dense matrix. RE-USE ships BF16 only: its `intake-v1:re-use` int8 and int4 were withdrawn on 22 Oct 2026, being slower than BF16 for about 0.2 GB less peak memory. A versioned int8 recipe may append `;half=<comma-separated module prefixes>` only for saved-input failures; int4 keeps maximum compression. Decisions view metadata lives in `arsenal-decisions-view.json`; `arsenal-derived.json` is reserved for store provenance. Derived receipts record the exact retained modules and parameter coverage. An int8 fallback cannot inherit the all-packed result.

        Von captures both English states, embedding, every encoder layer and scorer, plus three served probability vectors. RE-USE captures its spectral encoder, each of 30 time/frequency blocks and both decoders (63 outputs). Its ordinary timing workload is the shared three-second 24-kHz tone through `enhance`, with default Load settings, usable WAV delivery, and fresh serving workers for lifetime peaks. Saved source-format captures can be reused for a recipe-only quantization change via `--reference-run`; snapshot hashes and current reference replay must still agree. Bindings and table stats are written only by intake completion. A host-only `--rescore --write-stats --helpers <lab-helpers>` may re-emit enhancement metadata from complete saved captures: it validates the original hashes and exact worker/build/load identities, retains `capture_run` and every original timing/footprint, and starts no worker. This corrects labels and policy metadata without repeating a model run.
        """##,
        "INTEGRATING.md": ##"""
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
        """##

    ]
}
