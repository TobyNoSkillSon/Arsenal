# YuE2 3B

YuE2 generates 48 kHz stereo songs from style tags and sectioned lyrics. `generate_music` supports direct songs, melody-only ABC planning, full melody/chord planning, supplied or edited ABC conditioning, score-only output and instrumental generation. English and Mandarin are the documented languages. YuE2 has no language option: the lyrics carry the vocal language and the style prompt is passed as written, so the operation does not offer `language` (or a time signature, which YuE2 does not take). The Swift port appends `bpm` and `key` to the style tags when the prompt does not already name them. Instrumental requests use wordless sections plus an `instrumental` tag, the Swift port's method; the makers do not document a separate instrumental switch. The standard YuE2-Vae decoder is selected; the separate legacy decoder is outside this snapshot.

Inputs include `prompt` (style), `lyrics`, `duration`, `instrumental`, `bpm`, `key`, seed, batch, flow steps (`steps`, which overrides `yue.ode_steps`) and the `yue` planning, semantic/score sampling and decoder-tile options declared in `ops/generate_music.input.json`. Duration bounds the 25 Hz semantic budget rather than guaranteeing an exact musical ending. The schema allows up to 360 seconds. Default output is one WAV; float WAV, FLAC, M4A and ALAC, optional ABC/metadata and score-only mode are available. Score-only mode draws a random seed unless `seed` is given. Song files count against the service's 4 GiB per-job artifact limit, checked before generation; every schema-allowed request fits. Progress covers bounded native steps; long runs use the universal operation endpoint with `Prefer: respond-async` (durable job, polling, cancellation), not live PCM streaming.

Primary source: `m-a-p/YuE2-3B@8f24312f187763b9854adeec87e437e2b03bbff1`. Verified asset: `m-a-p/YuE2-Vae@152733a19ad43aa67e367f9b5503ef8075bb5126`. Every source filename, hash and byte count is pinned in `model.toml`. Swift port: `arraypress/swift-music-generation@51034b9874a046d4fe6e092524f8daf0940309ef`; the vendoring/adaptation boundary is in `Workers/Music/PORTED-FROM.md`.

Get's `cast:yue2-v1` callback prepares a complete canonical FP16 snapshot: rename transformer time-embedding paths, fold VAE weight normalization, map Sequential keys, transpose convolution axes and retain the synthesis decoder. The unused VAE audio encoder is explicitly excluded. Original headers contain 628 BF16 transformer tensors and 435 FP32 VAE tensors; the full native inventory has 801 tensors (628 transformer and 173 folded decoder). `gate/component-map.json` accounts for every retained tensor and ignored encoder key. Exact Range reads fetched headers only. Names and shapes match after preparation; source BF16/FP32 to FP16 is a conversion plan, not a trained precision qualification.

The FP16, int8 and int4 selections are executable. Group-64 affine tiers cover all 398 eligible matrices, 3,580,100,608 of 3,697,040,320 stored values (96.84%); convolution kernels and small tensors remain FP16. Counts include the complete graph and fixed embeddings, not just the advertised transformer. Random song, score and supplied-score/instrumental paths pass through production result validation at every selection. Synthetic source-format snapshots pass actual component conversion, ordinary loading, generation, tier conversion and stored-tier reload tests. Malformed input, cancellation and unload have separate lifecycle evidence.

Trained prompt/tokenizer agreement, musical quality, FP16 activation range, component/reference fidelity, long-song boundaries and int8/int4 quality still require the approved weight trial. No speed or memory claim is made. Live-test risk: **uncertain**, primarily because the upstream BF16 transformer and FP32 decoder now execute with FP16 weights/activations. Weight terms are CC-BY-NC-4.0 plus the vendor's individual-creator permission; consult the pinned source terms. No licence gate was added.

## Usage notes (2 Nov 2026)

YuE2 3B writes whole songs with vocals: `generate_music` takes a style `prompt` (genre, mood, voice) and `lyrics` in
sections (`[verse]`, `[chorus]`), plus `duration` (up to 360 s), `seed`, `bpm`, `key` and `instrumental`. It returns a
48 kHz stereo WAV (other formats on request) and can plan a melody as ABC notation first. English and Mandarin are the
documented vocal languages; the lyrics set the language. Songs count against the 4 GiB per-job file limit. Terms: CC
BY-NC 4.0, non-commercial use only; the catalog shows that line.

Parity: no maker reference yet (M-A-P's PyTorch pipeline was not run). On 2 Nov a seeded 10-second song came back at
the requested length with a normal level (FP16); it was not listened to. `results/real-check-2026-11-02.md`, golden
in `gate/golden/`.

