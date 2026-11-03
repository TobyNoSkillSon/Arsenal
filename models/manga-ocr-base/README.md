# Manga OCR Base

Reads the Japanese text in one manga speech bubble or text crop, vertical or horizontal, and returns it as a string. A ViT image encoder with a small character-level BERT-Japanese decoder, decoded with the maker's 4-beam search and post-processing (whitespace removed, ASCII and digits made full-width).

Input: one image (`image`, PNG, JPEG or HEIC), ideally a single bubble or text block. Output: `text`, the Japanese text read.

Example request:

```json
{"image": {"path": "/absolute/path/to/bubble.jpg"}}
```

On the maker's example bubble it returns よかったじゃないわよ！何逃げてるのよ！！早くあいつを退治してよ！, the card's own answer.

Limits: one text block per image, with no text detector, so crop to the bubble first. Trained for Japanese only, not Polish. A tiny or blurred crop gets invented text rather than a refusal. At most 299 characters.

Parity: on 6 fixed inputs the output tokens are identical to the maker's transformers generate (4 beams, MPS fp16); first-step logits differ by at most 0.016 (cosine 0.9999995). Details in log/003-parity.md.

Source: [kha-white/manga-ocr-base](https://huggingface.co/kha-white/manga-ocr-base/tree/82c77fb84369e0fce0d6e8f0ad96fa82646d0346), Apache-2.0, the safetensors conversion in refs/pr/4 of the same weights. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The maker documents general-purpose printed Japanese OCR, not only manga [model card, 2026-10-31 fetch, first-hand].
- Prompts, reference audio or voice settings, if relevant: Not relevant.
- Parameters people recommend: For poor or small images, read 5 to 10 characters at a time and upscale [issue #80 comment, 2024-08-30, opinion].
- Arsenal call: `POST /arsenal/v1/models/manga-ocr-base/ops/read`, or the same operation through MCP. Repo README: `from manga_ocr import MangaOcr; mocr = MangaOcr(); mocr('/path/to/img')` [kha-white/manga-ocr README, 2026-10-31 fetch, first-hand].

**Good and bad**
- Good at: No independent result found. The card gives no benchmark [model card, 2026-10-31 fetch, first-hand].
- Bad at: Issue #80's opener (2024-08-16) calls two reads "pretty bad", with no correct transcript. A 2024-08-30 reply says "100% accurate" [opinion]. White text on a black outline is a reported struggle [issue #81, 2024-08-19, opinion].
- Against the obvious alternatives: A fork's README gives about 14.4% character error rate for its own model, with no eval set named [gnurt2041/MangaOCR README, undated, opinion].

**Complaints, bugs and fixes**
- Complaint or known bug: Won't run after NVIDIA driver reinstall (issue #101, 2025-09-10). Full manga pages fail (issue #98, 2025-08-22). Warnings on transformers 4.46 and later (issue #85, 2024-12-13) [kha-white/manga-ocr issues, read 2026-10-31, opinion].
- Fix or workaround: Reported by one user, not confirmed: pinning transformers 4.45.2 avoids the #85 error (2024-12-19). Reported by one user, not confirmed: inverting the image "works flawlessly" on white-on-black text (#81, 2024-08-30). Mokuro is suggested for full pages (#98); none reports trying it. No maintainer comment.

**Evidence**
- Strength: thin. No maker benchmark; independent evidence is mostly user reports. Reddit not re-searched.
- Sources: model card and repo README (2026-10-31, first-hand); issues #80, #81, #85, #98, #101 (2026-10-31, opinion); gnurt2041/MangaOCR README (undated, opinion).
