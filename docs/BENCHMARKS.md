<!-- SPDX-License-Identifier: MIT -->
# Benchmarks

Real-model measurements, quality checks and introduction smokes are opt-in: run them only on a quiet Mac (see [AGENTS.md](../AGENTS.md), Running models); model intake checks follow [Adding a model](ADDING-A-MODEL.md). Agents prepare fixtures and append pending runs; they run the queue only on the owner's current go. [AGENTS.md](../AGENTS.md#quality-gate) defines the independent-reference rule: GPU half precision or vendor-published outputs. Historical CPU/FP32 gates do not authorize repeating that route.

## Precision measurements and held queue

`PrecisionStatSet` supplies every kind-specific metric for each actual dtype and runner. New results have TOML front matter in Markdown (`PrecisionStatsResult`, schema 2; `Resources/schemas/precision-stats.schema.json`); existing schema-1 JSON stays intact. Cells contain explicit missing/unavailable states and chip/date/method provenance. Own-16-bit comparison is distinct from task accuracy and independent implementation parity. Energy uses rounded battery percentage per the stated kind unit.

The schema-2 harness uses the ordinary `RealModelsService` Get, Load, operation and Unload path for each dtype. Get derives snapshots using the served converter and recipe; Load receives the same dtype, automatic Fast policy, kernel folder, gates, configuration and context maximum as the app. Quality, elapsed time, first usable output, physical footprint and energy are observed outside the engine. Chat alone adds `measurement_logits` on the already loaded ordinary worker for at most 32 teacher-forced next-token comparisons; it does not load or quantize a second model.

All real work is held in `<support>/benchmarks/pending.json` (`Resources/schemas/pending-benchmark.schema.json`). Introductions, revision changes and runner changes enqueue three precisions at the automatic effective mode. Old pending mode/revision entries become `superseded`; their receipts remain. Saved Exact preferences retire silently. Ranking and current cells require matching nonempty revision/run, chip, date, source and baseline across availability, quality and timing; signed schema-2 history and local results follow the same rule. Historical records without this identity remain unmeasured for the current comparison.

The installed agent entry is `Arsenal.app/Contents/MacOS/Arsenal stats`, dispatched headlessly before AppKit or the single-instance claim. It uses the existing app image without adding a helper or symlink, preserving compatibility with 3.1's release-archive verifier. `stats list --support DIR` inspects the queue; `stats seed --support DIR --models MODEL_ROOT` only introduces entries. `stats run --support DIR --models MODEL_ROOT --source-support APPROVED_WEIGHT_STORE --go` requires the owner's current authorization for the run. Helpers and signed Resources default to their installed bundle locations; development checks can supply `--helpers` and `--resources`. MODEL_ROOT is a writable evidence tree matching the accepted model revisions, not the sealed app Resources. The standalone development `arsenal-stats` executable accepts the same commands without the `stats` prefix.

Each model gets a disposable isolated support directory. Pinned source weights, any already derived precision snapshots and required assets enter through APFS clones; failure to clone or find the source is a named refusal, never a download or full weight copy. Ordinary Get stores its derived snapshots there. The source is retained across source → int8 → source → int4 → source brackets; the entire profile, including derived snapshots, is deleted after the model. Only ordinary stderr logs and `MODEL_ROOT/<id>/results/local/*.md` receipts remain. No residents or saved app settings enter the profile. CPU fixtures never earn Measured; live rootless IOReport energy requires finite, nonnegative CPU/GPU counters and a complete recognized cluster set, otherwise the cell is unavailable.

RE-USE is refused because its ordinary loader widens weights to FP32. Stable Audio 3 is refused in both effective modes: Exact promotes/retains FP32 modules, and Fast's qualification/fallback may execute FP32. Repairs are coordinator follow-ups for later work on those models. Spectral distance remains the audio metric; its exact gate additionally requires identical sample rate, length and samples, so a sign inversion or changed Hann-window endpoint cannot pass merely because distance is zero.

The comparison records weight revision, drainer/native binary hashes, a shared run ID, fixed fixtures and the busy Mac. Three warm ordinary requests precede short ordinary operation windows. Chat sustained speed excludes first-token prefill using the ordinary decode progress frame; first usable text is observed separately. Get conversion memory is distinct from sampled ordinary Load/operation physical footprint. This is a matched comparison, not the legacy quiet-machine policy below; no real model, battery accuracy, perceptual noise threshold or production peak RAM has been qualified by this lane's synthetic tests.

The schema-2 chat timing/energy workload omits sampling overrides and fixes seed zero, so ordinary folder defaults reach ChatRequest and GenerateParameters (currently temperature 0.6, top_p 1); teacher-forced quality is unchanged. Its v3 fixture identity prevents prior greedy v2 timings from sharing provenance.

Before each Get and Load, the drain uses ordinary memory admission with swap disabled and no queue wait. Cold Get reserves twice the verified source-snapshot bytes plus asset bytes, ordinary runtime overhead and the bounded 256 MiB logits payload; existing verified packed snapshots use their own byte estimate and avoid conversion. Admission retains the normal free-memory margin and 512 MB headroom. Refused entries stay pending with `skipped: not enough free memory (needs ~N GB)` and later models continue. These conservative estimates are not measured peaks or a guarantee of model workspace bounds.

## Legacy schema-1 reference

The sections below describe the retained `arsenal-bench` format and earlier procedures. They are not the 3.2 agent entry or an exception to the hold. Internal Fast/Exact fields remain provenance; users choose precision and Arsenal selects mode automatically. Legacy results never become schema-2 precision recommendations.

## Frozen suites

`Tests/Fixtures/Bench/kinds/<kind>/suite-v1.toml` and `fixtures-v1.json` jointly define the suite hash. Quick and full use identical inputs/settings; full repeats ten times rather than once. Both discard one warm-up. Profile remains part of the comparison key; quick numbers are not mixed with full.

| Kind | Fixed workload | Reported metrics |
|---|---|---|
| Voices | Short and long English sentences, af_heart or recorded first available voice | Audio seconds / generation seconds, first-audio seconds, worker peak RAM |
| Decisions | Four labelled yes/no, choice and score questions, plus one batch of the same set | Typed accuracy, median seconds/question, batch questions/s, worker peak RAM |
| Images | Two prompts, seeds 24301/48879, 512×512, four steps | Wall seconds/image, worker peak RAM |
| Chat | Arithmetic, science and syllogism prompts, temperature zero, 64 tokens max | Prefill/decode tokens/s, first-token seconds, normalized exact match, context, worker peak RAM |

Chat quality is a small diagnostic, not a claim of performance on a standard evaluation benchmark. Voices WER is `null`: no fixed version-pinned ASR path has yet been qualified without changing sibling apps. Image reference deviation is `null`: no pinned reference images exist. Prefill throughput is `null` unless the runner exposes actual prefill seconds; first-token time includes other work and is not mislabelled prefill. Unknown memory is `null`, never zero.

## Conditions

Controlled measurement requires three consecutive three-second idle samples: other-process CPU <250% of one core, other GPU <10%, compositing ≤15%. GPU time belonging to WindowServer/ghostty is recorded and excluded, like Vella's lock tool. Missing telemetry is a refusal, not assumed idle. During the workload the sampler excludes the harness and its worker descendants, records three-second samples, and flags any contaminated sample. No contaminated result is ranked or given a badge. Energy is explicitly `not_measured`.

A closed Arsenal GPU gate, an open Vella window, or absence of the exclusive measurement lock refuses a real run. The wrapper also terminates a run when the gate closes. Stub mode exercises schema/aggregation only; synthetic conditions are identified in its note. Full workloads must be scheduled in bounded model windows, not started across Vella's measurement window.

## Result interface

`ResultsFile` in `ArsenalCore` is the Codable representation of `Resources/schemas/results.schema.json`. Fields record model/kind, chip slug, OS build, app build, runner/worker SHA-256, actual dtype (the `precision` field), Fast/Exact mode, suite ID/version/hash, profile/date, gate status, stub flag, conditions, metrics and individual case measurements. `ResultValidation.validate(data:)` returns this type or throws; catalog validation calls it. `ResearchLogValidation.validate(markdown:)` checks log entries.

`BenchmarkRanking.build(results:chip:)` returns model → ordered runner names. Only clean, non-stub results with a passed gate qualify. The latest result fixes suite hash/profile/dtype/mode; incompatible rows are excluded. Voices/chat prefer greater throughput, decisions/images lower latency. Missing speed is never preferred over a measured speed. `ResultsFile.tableMetrics()` supplies kind-specific display strings; `MeasurementBadge.derive(results:chip:)` computes Measured / Measured on another chip / Unverified. Estimates from another chip must remain labelled by the caller.

A model result at `results/<chip>.json` is one selected dtype/mode/profile, not evidence for all dtype cells. Do not overwrite or inherit another dtype's quality gate. A benchmark functional-quality diagnostic is not a model presence/quality gate; `gate_passed` requires separately demonstrated gate evidence.

## Quick checks

Quick checks exist to prove the harness on a real model, not to produce a result anyone ranks. The old `BenchmarkLocalProof.allowedModels` allowance (Kokoro 82M) is superseded by the current hold: every real-model run needs the owner's current go. Controlled `measure` sessions are never started by workers. The shared `HostChip.current` supplies the result slug (for example `m5-max`), exactly as the rows and runner selection do.

```sh
scripts/mlock gpulock quick-kokoro -- perl -e 'alarm 90; exec @ARGV' \
  arsenal-bench run kokoro-82m --precision fp16 --quick-check --profile quick
```

`BenchmarkRequest.quickCheck` is encoded as `quick_check`. Quick checks require a gpulock ancestor, record three idle samples and during-run samples, and set `measurement_lock=false`, `contended=true|false` with a gpulock-quick-proof note. They proceed on a busy machine. Only controlled measure runs refuse contention. `ResultsFile.verificationLabel` returns `quick check`; these results never enter runner rankings or earn Measured.

The executable opens RealModelsService with `--resources DIR` for a verified catalog and `--helpers DIR` for native worker artifacts. `run` never downloads and has a 60-second quick-check shutdown deadline. Its development `get kokoro-82m` command calls RealModelsService.get, including the native Hugging Face download, FP32 to FP16 cast, asset fetch and store manifests. `prepare-proof-catalog --resources /tmp/arsenal-bench/product-resources` only prepares a signed local catalog; Get installs model folders and weights. Preparation requires isolated `/tmp/arsenal-` paths or an explicitly authorized persistent `~/Library/Application Support/Arsenal-dev/bench-support` directory, never the real Arsenal support directory. API hosts should inject locked execution into BenchmarkController rather than infer inside an unlocked app process.
