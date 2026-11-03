# Tool windows for the new capabilities

Every window below is built from the shared tool kit (`Sources/Arsenal/Tools/Kit`) and shares the Get more models / Make an Image design: the band on the traffic lights' line (model pop-up, operation pills, Show in Finder), the live figures, options on the left with the run button at their foot, the result on the right with this window's earlier results under it. A window shows only what the chosen model's operation schemas declare: an operation it lacks is no pill, an option it lacks is no control.

## How a model gets into a window

An integrator declares operations; nothing else. A schema-3 model folder with `tools` absent resolves to the generic tool, and Arsenal places it by its declared operation names (`ToolCatalog`):

- each window in the table below lists every table model that declares one of its operations, installed or not, with the reason a model cannot run here (not installed, approval needed) in the model pop-up; the menu entry is the window's from the start, so it does not rename itself at Get;
- the main menu lists tasks, not models (`Sources/Arsenal/Menu/MenuTasks.swift`): each window is one entry for every model with Show in menu on that serves it, and `Run…` stays only for a declared operation no window offers; models whose operations only serve agents (`ToolCatalog.backgroundOperations`: decide, judge, embed, rerank, end_of_turn, count_tokens) have no entry;
- the names come straight from the folder's manifest (`ModelsService.declaredOperationNames`: no schema read, no admission, nothing reported), read once per folder digest and revision; a folder whose schemas cannot be read is still placed by its declarations and shows the refusal in each window's model pop-up, and its resolved operations are read at most once per change, never twice per refresh;
- the operations already read open each window complete on its first frame (form and pills), refreshed by the API's answer;
- operations the windows drive themselves or that only agents call (`segment_session`, `video_session`, `count_tokens`, `end_of_turn`, `voices`) put a model in no window. A model that declares only those has no menu entry.

Options come from the operation's input schema: `x-arsenal-file` makes the well (images, audio, video, documents, several files where the schema takes an array), and `x-arsenal-control` picks the control: `language` (pop-up, the schema's `enum` when it has one), `lyrics`, `slider`, `json-schema` (named fields, JSON one click away), `image-points` and `image-boxes` (placed on the picture; `x-arsenal-image` names the picture or video). Arrays of short strings are chips, `documents`/`input`/`text` arrays are one per line, `pages` is typed as "1-3, 5" and sent zero-based, number arrays as "0.25, 0.5". A top-level `oneOf`/`anyOf` of required groups is read as alternatives ("an image or images", "a video or two frames"): the window runs when one is filled.

Alternatives are one chooser (`OptionGroup`): a row of pills, only the chosen option's controls shown and only its values sent; every declared option stays one click away. Upscale: Scale | Height. Smooth Video: Factor | Frame rate (Positions only with two frames). Find Objects: Labels | Phrase | Instruction. Extract: Text | File, and Fields | Labels. Separate Audio: Stems | Describe. A file dropped or pasted chooses the option that takes it.

An option left unset sends nothing and says so: the declared default, else a dim "Auto" (sliders rest neutral, unfilled, until moved); the allowed range is the hover. Empty fields whose emptiness means something say it ("Estimated", "First", "Last", "All", "Random"). Choices without a default read "Auto".

## Windows

| Window | Operations | Lives in | Inputs → result |
|---|---|---|---|
| Transcribe | `transcribe`, `diarize`; live through the Realtime socket | Main menu, per model | Clip (drop, choose); language, word times, speakers, task, words to expect. Transcript with each speaker in a colour, a who-spoke-when bar, each line's time playing the clip from there; Copy, Save as text or SRT. Live (microphone button): a line per turn; turns end where an installed `end_of_turn` model says, else at a 0.7 s pause. |
| Separate Audio | `separate_audio` | Main menu | Clip; stems as chips or a description of the sound (main text), extract/remove, steps, seed. Original first, then each stem: Play, envelope, length, Copy, Show in Finder, drag out. Job: progress, Stop cancels the work. |
| Translate | `translate` | Main menu | Text (main), To (required), From (Automatic), context. Translations as a conversation, newest last, with the language read. |
| Read Document | `read_document` | Main menu | PDF or page image; pages, table format, images, annotation schema. Page markdown rendered (headings, tables as grids, lists, code), returned images; page arrows; Copy, Save as Markdown. |
| Extract | `extract` | Main menu | Text (main) or an image or document; the answer's fields (JSON Schema) or labels. The answer as JSON; a break of the fields asked for stated in red; entities as labelled chips. |
| Search | `rerank`, `embed` | API and MCP (background models have no menu entry) | Query (main) and passages one per line (pictures too when the embed input takes images). Ranked list with each score as a bar. Reranker scores itself; an embedding model embeds query and passages (as `query`/`document` tasks when declared) and the window ranks by cosine similarity. |
| Select Objects | `segment` (+ `segment_session`), `track_video` (+ `video_session`) | Main menu | Picture or video with include/exclude points and boxes placed on it, a description when the model takes text, several masks. The mask tinted over the picture (also on the left canvas), the cut-out, or the mask; mask pills with scores. Once a selection shows, each new click selects again quietly: the shown mask stays (no progress card) until the new one replaces it in place, and nothing is saved or added to the history until the selection is kept — by Select, Copy, a drag out, or a new picture. The history shows each kept selection as its cut-out. Sessions are created with the first selection and released on a new picture, model or operation, and on close; a stale session is reopened once. Video: prompts on a chosen frame go as that frame's `prompt_updates`; job with progress and Stop; mask video plays. |
| Find Objects | `detect`, `ground_screen` | Main menu | Picture or screenshot (⌘V); labels as chips, phrase, instruction (main text for grounding), threshold. Boxes and points in each label's colour with scores over the picture, a list beside it (hover gives pixel coordinates). Screen pixels are divided by the reply's `image_size`, never forwarded as normalized. |
| Depth | `estimate_depth` | Main menu | Picture, or several views; relative/metric, point cloud, camera, focal length. Depth in colour (near warm, far cool) or grey, compared with the picture under a handle, legend with near/far or metres, views as pills, other returned files (point cloud) to reveal; focal length figure. |
| Upscale | `upscale_image`, `upscale_video` | Main menu | Picture or video (dropping one picks the operation); scale, height, prompt, seed. Picture: result over original under a handle, size figure. Video: plays, Original one click away. Job. |
| Smooth Video | `interpolate_video` | Main menu | Video, or two frames; factor, frame rate, positions. Smoothed video playing (Original one click away) or the made frames in order between the two. Job. |
| Make 3D | `image_to_3d` | Main menu | Picture; resolution, texture, simplify, seed. GLB shown as a transparent SceneKit still at rest, the live view under the pointer (turns and zooms), triangle count, Open in the Mac's viewer, drag out. A GLB Arsenal cannot read is a stated failure with the saved file's name. Job. |

### Changes to existing windows

| Window | Change |
|---|---|
| Make a Sound | A song model's `generate_music` shows Lyrics (under the prompt), Language and Instrumental from its schema; Instrumental greys the lyrics and sends none. A model without them shows none. |
| Chat | A paperclip (and drop, ⌘V) for pictures and clips only when the model's `chat` schema has `image_url`/`input_audio` content parts (clips in the formats it declares); a braces button for an answer held to named fields only when `response_format` takes `json_schema`. Text-only models (qwen3.8-27b's narrowed schema) show neither. |
| Remove Background | A model of the window that also declares `track_video` takes a video: points and boxes on its first frame, the operation's words field, a job; the result is the model's mask video, titled as a mask, never as an alpha cut-out. BiRefNet's image path is unchanged. |

## Main menu or API only

- In the menu (one entry per task, for every model with its Show in menu switch on): every window above except Search, whose rerank and embed models work in the background.
- API only: raw vectors (`embed` output; Search shows the ranking it makes, not the vectors), `count_tokens`, `segment_session`/`video_session` (the window manages them), and `end_of_turn` (it serves live transcription's turn detection and agents; a file-level end-of-turn probability is not something the user judges by eye).

## Kit pieces

`ToolCatalog` (operation → window routing, result folders by type), `CapabilitySession` (subject well routing files to fields, alternatives, paste/drop, outcomes saved by type, history, figures, preview), `CapabilityView`/`SubjectLead`/`SubjectWell` (image, clip with envelope, video first frame, PDF first page, several files), `PromptCanvas` + `Letterbox` (points/boxes normalized to the picture, never its letterbox), `MaskOverlay`, `DetectionOverlay`, `OutcomePanel`/`OutcomeHistory`, `TranscriptList`/`SpeakerTimeline`/`TranscriptSegment` (SRT), `StemList`, `RankedList`, `MarkdownBlocks`, `JSONView`, `BeforeAfter`, `DepthImage`/`DepthLegend`, `VideoPlayerView`/`VideoFrame`/`VideoInfo`, `GLB` reader + `ModelPreview`, `LanguagePicker`, `TagField`, `SchemaEditor`, `LiveTranscription` (Realtime socket, injectable audio source), and in `ToolSession`: jobs (`Prefer: respond-async`, polled with back-off, Stop cancels on the service, the model loaded first through the memory warning), `makeForm`, `prepareInput`.

Results are saved as they arrive: pictures to ~/Pictures/Arsenal, sound to ~/Music/Arsenal, video to ~/Movies/Arsenal, documents, 3D and text to ~/Documents/Arsenal.

## Tests and renders

`Tests/ArsenalAppTests/CapabilityWindowTests.swift` drives every window through the app's API over the stub service with `CapabilityFixture` (fixture models declaring the kinds' real schemas; runners answering per the kinds' output schemas with files by id): routing and menu entries, declared-only options, semantic controls, file routing, point mapping, each window end to end, sessions and stale sessions, job progress and cancel, screen-pixel conversion, GLB reading, failures, a model that cannot run, the memory warning before a job's load and a direct run. `ARSENAL_QA="render-capabilities <dir>"` renders each window (first frame, ready, busy, result dark and light, minimum size, failure, blocked) and the changed windows; `ARSENAL_QA_ONLY` narrows it.

## Open contracts

- `read_document.pages` is zero-based (documented in the kind schema); the window shows pages as people count them and sends them zero-based.
- File items may carry a display `label` and `order` (kind output schemas); the window titles files from their `id` until it reads them.
