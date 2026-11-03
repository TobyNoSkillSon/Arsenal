<!-- SPDX-License-Identifier: MIT -->
# FLUX 3 Image Klein

Black Forest Labs' distilled FLUX 3 image model, on this Mac in Swift and MLX: a 5.6B transformer with the Qwen3-VL-4B-Instruct text encoder and the FLUX.2 autoencoder, 16 steps, no guidance. One model makes images from text, edits a photo, and composes a new image from up to 10 reference images. It renders text inside images and takes layout boxes in the prompt.

## Operations

- **generate**: `prompt`, `width` and `height` (multiples of 16, 64 to 4096, at most 4 megapixels; default 1024 x 1024), `steps` (default 16), `seed`, `n` (1 to 4).
- **edit**: `prompt` plus `images`, 1 to 10 reference images (PNG, JPEG, HEIC or WebP). One reference is fitted to at most 4 megapixels on a canvas padded to multiples of 16, and the result keeps the reference's size and framing. Several references are each scaled to about one megapixel and cropped to multiples of 16; the result takes the first one's size. `width` and `height` choose another size; left empty (the default), the edit keeps the reference's size. `steps`, `seed` and `n` as above.

Results are PNG files. The same seed, prompt and references give the same image.

## Prompts

The model takes full scene captions, not short requests: medium and style, each subject with its colours, materials and position, background, lighting and camera. Black Forest Labs says not to use it without a prompt upsampler; theirs is a separate 9B model (`black-forest-labs/flux-3-image-klein-pup`) that Arsenal does not include. Any language model can write the caption instead, or write it by hand. A short prompt gives a generic image.

- For an edit, describe the result: a complete caption of the reference with the change folded in. A one-line instruction such as "make it blue" reframes the whole shot.
- Put text to render in quotes.
- For layout, append a JSON list of elements, each `{"id": ..., "bbox": [top, left, bottom, right], "desc": ...}` on a 0 to 1000 grid, and tag the element in the caption as `<id>`. In edits, refer to the n-th reference as `<ref_image_n>` and give an element `from`, `src_bbox` and `tgt_bbox` (a null target removes it). Boxes set placement and scale, not a hard mask.

## Install

This model is not in Arsenal's signed catalog, because its licence is not open. It is an add-on folder:

1. Copy this folder to `~/Library/Application Support/Arsenal/local-models/flux3-image-klein/`. Arsenal lists it with a Local badge (Refresh if the Models table is open).
2. Read and accept the FLUX Kommunity License on the model's Hugging Face page: the repository is gated. Then give Arsenal access in one of two ways: add a Hugging Face read token in Arsenal's Settings › Hugging Face Token… (it is kept in the login keychain), or download the eight pinned files yourself with `hf download black-forest-labs/flux-3-image-klein --revision 337337621b8526e120f2275d250242f4a6bedd75` and those file names; Get then takes them from the Hugging Face cache instead of downloading again.
3. Get the model: about 20 GB (transformer 11.1 GB, text encoder 8.9 GB, autoencoder 0.3 GB) at the pinned revision.

The bundled flux worker runs it, so there is no code to approve.

## Memory and precision

It runs in BF16, the release's precision. mflux, the reference port, reports feed-forward outputs near 3 x 10^5 and attention outputs near 3 x 10^4 in BF16 runs; the first is beyond FP16's 65,504, so no FP16 copy is offered. Black Forest Labs measured a peak of 31 GB at 1024 x 1024 with their reference code. No figure has been measured in Arsenal yet.

## Licence

Weights: [FLUX Kommunity License v1.0](https://huggingface.co/black-forest-labs/flux-3-image-klein/blob/337337621b8526e120f2275d250242f4a6bedd75/LICENSE.md). Non-commercial use; commercial output generation is allowed only for Qualifying Users with less than US$5M in gross annualized revenue, including affiliates. It also restricts uses (its Section 4) and covers export controls. Use of the Black Forest Labs and FLUX trademarks needs prior written permission, except required attribution and necessary descriptive references. The Qwen3-VL-4B-Instruct text encoder is Apache-2.0.

The Swift code is Arsenal's (MIT), ported from mflux (MIT); see `PORTED-FROM.md` beside the flux worker's source.
