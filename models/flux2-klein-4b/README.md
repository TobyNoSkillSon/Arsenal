# FLUX.2 Klein 4B

FLUX.2 [klein] 4B is a 4-billion-parameter rectified flow transformer from Black Forest Labs that turns a text prompt into an image. It is step- and guidance-distilled, so it makes an image in four denoising passes with no classifier-free guidance. The checkpoint has three parts: the image transformer (5 double-stream and 20 single-stream blocks), a Qwen3 text encoder (36 layers, hidden width 2560) and the FLUX.2 autoencoder. Its strength is speed: the makers call it their fastest image model, with end-to-end generation in under a second on modern hardware.

## What it is for

Black Forest Labs built Klein for real-time, interactive and latency-critical image generation on consumer hardware, and placed the 4B model at "maximum speed, consumer hardware, edge deployment". The undistilled Base variants, not this one, are the makers' choice for fine-tuning and LoRA training.

- Fast drafts and iteration on an idea, many images per minute
- Illustrations, product shots, concept art and photo-style scenes from a written description
- Images with short rendered text, such as a sign or a label (text can come out distorted)
- Batches of variations of one prompt from consecutive seeds

## Inputs and outputs

### `generate` (default)

Generates one to eight images from a text prompt.

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | 1–32768 characters; must not be only whitespace. Only the first 512 tokens are used (see below). |
| `width` | integer | no | 64–2048, multiple of 16. Default 1024. |
| `height` | integer | no | 64–2048, multiple of 16. Default 1024. |
| `resolution` | enum | no | `"512x512"`, `"768x768"` or `"1024x1024"`: a square shortcut. An explicit `width` or `height` wins over it. |
| `steps` | integer | no | 1–100. Default 4, the count the model is distilled for. |
| `seed` | integer | no | 0–4294967295. Omitted: a new random seed per request. |
| `guidance` | number | no | Must be `1` if sent. "Distilled Klein has no adjustable classifier-free guidance." Any other value is rejected. |
| `n` | integer | no | 1–8 images. Default 1. |
| `out_dir` | string | no | Managed by Arsenal (the job folder). Leave it out; a folder outside Arsenal's job directory is rejected. |

Returns `images`, an array with one entry per image: `path` (a PNG in the Arsenal job folder), `width`, `height`, `bytes` and `sha256`; plus `seed` (the request's base seed) and `seconds` (wall time for the whole request). The runner also returns `seeds`, the seed used for each image. Each file is an 8-bit RGB PNG in sRGB with no alpha, at exactly the requested width and height.

The prompt is wrapped in the Qwen3 chat template as one user message (generation prompt on, thinking off), tokenised, and truncated or padded to 512 tokens. Hidden states from encoder layers 9, 18 and 27 are concatenated into a 512 × 7680 conditioning; later encoder layers never run. Text past the 512-token window, template tokens included, has no effect. Images start from seeded Gaussian latents of shape 128 × H/16 × W/16, so width and height must be multiples of 16; a 1024 × 1024 image is 4096 image tokens. The schedule is FlowMatch Euler with the vendor's empirical shift. With `n` above 1, images run one after another with seeds `seed`, `seed+1`, … (wrapping at 2^32). The same prompt, size, steps, seed and precision give the same image. There is no image input: editing and reference images are not exposed.

## Run it

MCP: `arsenal_run(model:"flux2-klein-4b", input:{"prompt":"A red ceramic teapot on a wooden table, soft window light","width":1024,"height":1024})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/flux2-klein-4b/ops/generate" -H 'Content-Type: application/json' -d '{"prompt":"A red ceramic teapot on a wooden table, soft window light","width":1024,"height":1024,"seed":7}'`

The settings panel sets Shape (1:1 1024 × 1024, 3:4 864 × 1152, 4:3 1152 × 864, 9:16 768 × 1344, 16:9 1344 × 768), Seed (random by default) and, under More, Width, Height and Steps. A request's own fields override the panel. There is no Quality or Guidance row: the tuned 4 steps is already the fast setting and guidance is fixed. Measured on an M5 Max while the Mac was in use (quick check, 16 Oct 2026, 1024 × 1024, 4 steps, warm): bf16 3.88 s per image, int8 4.12 s, int4 3.91 s. Images take roughly the per-image time times `n`; the first image after Load also pays compilation.

## Limits

- Text to image only. The upstream model also edits images and combines several reference images; Arsenal does not expose those modes, LoRAs or preview streaming.
- Prompt following depends heavily on prompt style, and the model may not match the prompt.
- Rendered text may be inaccurate or distorted.
- It is not a source of facts, and it may reflect or amplify biases in its training data.
- Prompts are tuned for English (the card lists `en`); other languages are *(unverified)*.
- Only the first 512 prompt tokens count, although the field accepts 32768 characters.
- Step counts other than 4 are allowed (1–100) but the model is distilled for 4; quality at other counts is *(unverified)*.
- Sizes from 64 to 2048 pixels per side are accepted; the makers benchmark at 1024 × 1024 and quality at far smaller or larger sizes is *(unverified)*.
- Memory: about 20 GB peak at bf16, 12.8 GB at int8 and 9.0 GB at int4 (measured, see below).
- int4 failed Arsenal's quality check and is offered with a warning.

## Licence and credit

Apache-2.0 (the 4B weights; the 9B Klein variants use a non-commercial licence instead). It allows commercial use, modification and redistribution, provided the licence and notices are kept and changes are marked; it grants no trademark rights, and modified versions must not use the Black Forest Labs or FLUX marks in a way that implies endorsement. The model card's out-of-scope list forbids, among others, unlawful use, content that exploits minors, non-consensual intimate imagery, deceptive content and harassment, and the makers encourage deployers of the 4B model to add NSFW and protected-content filters. Made by Black Forest Labs. The card gives no paper; the code repository asks to be cited as `@misc{flux-2-2025, author={Black Forest Labs}, title={{FLUX.2: Frontier Visual Intelligence}}, year={2025}, howpublished={\url{https://bfl.ai/blog/flux-2}}}`.

## Links

- Model card: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/tree/e7b7dc27f91deacad38e78976d1f2b499d76a294
- Code: https://github.com/black-forest-labs/flux2
- Blog post: https://bfl.ai/blog/flux2-klein-towards-interactive-visual-intelligence
- Licence: https://huggingface.co/black-forest-labs/FLUX.2-klein-4B/blob/e7b7dc27f91deacad38e78976d1f2b499d76a294/LICENSE.md

## How Arsenal runs it

Current precision intake: BF16 and whole-part int8 are qualified; int4 did not pass the quality check. See `log/008-intake-quality-policy.md`.

**Source.** Black Forest Labs `black-forest-labs/FLUX.2-klein-4B`, revision `e7b7dc27f91deacad38e78976d1f2b499d76a294`, Apache-2.0. The pinned vendor snapshot includes the transformer, Qwen3 text encoder, tokenizer and VAE. Source weights and evaluated activations stay bf16. The checkpoint, 18 files and 15,980,140,843 bytes, is unchanged; the root `flux-2-klein-4b.safetensors` duplicates the transformer and is not downloaded. Runner provenance and upstream licence are in `runners/swift/PORTED-FROM.md`: the MMDiT and Qwen3 tap encoder are adapted from xocialize/flux2-klein-swift (MIT), and the VAE decoder, scheduler, pipeline and weight loading were written from the vendor's diffusers implementation.

**Runner.** Get → Load → Generate → Unload. The folder selects `native`, worker family `flux`, at bf16, int8 or int4 (fp16 overflows in this model, so bf16 is its 16-bit format), with no Fast/Exact switch. int8 and int4 pack the transformer and the Qwen3 text encoder matrices and embedding with MLX affine group-64 quantization; the VAE stays bf16. Get derives the int8 and int4 snapshots from the bf16 download (about 16 GB bf16 on disk; int8 about 8.5 GB and int4 about 4.5 GB *(unverified)*). The Swift package is linked by `Workers/Flux` and dispatched by `Workers/Native`; the runtime needs no Python inference environment. Loading only reads a complete local snapshot. The worker validates each request as in the table above, emits per-step progress, checks cancellation between steps and images, and fails on nonfinite output instead of writing a corrupt image.

**Evidence.** On 8 Oct 2026, M5 Max 40-core GPU / 128 GB, macOS build 26A434, three warm images per dtype were alternated at 1024×1024, four steps, after one warm-up each. bf16: 3.814 s/image, 0.716 s/denoising step and 18.70 GB lifetime peak physical footprint. fp16 lab baseline: 3.863 s/image, 0.753 s/step, 18.71 GB peak. The Mac was in use; the small timing difference is inside the observed spread and establishes no speed advantage.

The default bf16 worker's frozen 512×512 quick recipe (one warm-up, photo and text once) has a median of 1.068 s/image and peaks at 15.92 GB. Both images were inspected at 100%: a red ceramic teapot on wood and exactly "ARSENAL". `results/m5-max.json` is an uncontrolled **quick check**, in the runner's single Exact mode, never Measured or ranked.

Intake on 15–16 Oct 2026 (busy M5 Max, warm 1024×1024 four-step, balanced ABBA BAAB against a resident bf16 baseline; `results/intake-m5-max-*-native-exact.json`): bf16 3.88 s/image, 20.40 GB peak; int8 4.12 s/image, 12.77 GB peak, quality check passed; int4 3.91 s/image, 9.01 GB peak, quality check failed. In the component rescore (`log/008`), maximum component relative L2 is 0.000000 for bf16, 0.039887 for int8, 0.162563 for fp16 and 0.264390 for int4; fp16 also produced non-finite pre-clamp residuals (reference maxima reach 98,816). No speed winner is claimed.

All five saved component comparisons and the fixed-image regression bounds pass gate v2 against constrained bf16 mflux captures. The component thresholds follow the captured bf16 floor with a 25% regression margin; they do not establish a seed-noise quality tolerance. bf16's tiny-transformer relative L2 is 0.00710 vs fp16's 0.14520; the fixed captured image has 27.52 dB PSNR vs fp16's 22.63 dB. **Stock mflux fidelity and image-quality tolerance remain unqualified; overall `gate_passed=false`.** Stock mflux explicitly uses fp32 in several operations, so the no-fp32 reference constraint required recorded process-local dtype overrides. The saved reference is reused without a new model run.

Evidence and thresholds live in `gate/`; details and historical fp16 and quantized trials are in `log/006-swift-fp16.md`, the intake component gate in `log/007-intake-components.md` and `log/008-intake-quality-policy.md`. Raw captures and QA images stay outside the signed model folder. Earlier Python (mflux) stand-in logs, `log/001`–`log/005`, are historical.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Text prompt only. A vendor says quality drops noticeably below 768 px, with no test figures. [deAPI.ai blog, 2026-04-29, opinion]
- Prompts, reference audio or voice settings, if relevant: Prose over keywords, 40 to 120 words, describing subject, scene, lighting and materials; negative prompts ignored. [deAPI.ai blog, 2026-04-29, opinion]
- Parameters people recommend: Keep 4 steps and guidance 1, the card's example values. A vendor says other step counts break output. [model card, undated, first-hand; deAPI.ai blog, 2026-04-29, opinion]
- Arsenal call: `arsenal_run(model:"flux2-klein-4b", operation:"generate", input:{"prompt":"A red teapot on a wooden table","steps":4,"guidance":1})`

**Good and bad**
- Good at: Speed: "the fastest model in our tests"; "handled the lighting brief best of any model". [CriticaTV comparison, 2026-09-21 (listing date), opinion]
- Bad at: Rendered text, which the card says may be inaccurate or distorted. A reviewer said it did not handle East Asian faces well. [model card, undated, first-hand; CriticaTV, 2026-09-21, opinion]
- Against the obvious alternatives: Same review: Z-Image Turbo is "the overall winner for our kind of client work"; no SDXL test. [CriticaTV, 2026-09-21, opinion]

**Complaints, bugs and fixes**
- Complaint or known bug: The card says about 13 GB VRAM; a user on a 16 GB V100 got CUDA out-of-memory errors. Fix or workaround: Reported by one user, not confirmed: a replier says an int8 text encoder and an fp8 transformer (about 5 GB each) fit. No maintainer reply. [black-forest-labs/flux2 issue #35, opened 2026-02-02, reply 2026-05-08, first-hand (user report)]

**Evidence**
- Strength: thin; one user report and two opinion pieces; no independent test.
- Sources:
  - https://huggingface.co/black-forest-labs/FLUX.2-klein-4B, undated, first-hand
  - https://github.com/black-forest-labs/flux2/issues/35, 2026-02-02, first-hand (user)
  - https://deapi.ai/blog/prompting-flux-2-klein-what-works-what-doesnt-and-why, 2026-04-29, opinion
  - https://www.criticatv.com/z-image-vs-flux-klein-vs-qwen-image-krea-2-comfyui/, 2026-09-21 (listing date), opinion
