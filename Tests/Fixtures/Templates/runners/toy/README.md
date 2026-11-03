<!-- SPDX-License-Identifier: MIT -->
# CPU toy

The three runner templates implement `evaluate`: sum a nonempty list of at most 64 numbers, multiply by the load-time `scale` setting, and optionally repeat the arithmetic for cancellation testing. The fixture loads `scale = 2`, so `[1, 2, 3]` returns `result = 12`. Both modes use the same math.

This is a local protocol fixture, not a Hugging Face model. Its zero-byte weight and dummy revision are placeholders; Get must not be used on it. It is outside `models/` and never enters the product catalog. Run the checker as shown in each template README. No model-quality, GPU or benchmark evidence is implied.
