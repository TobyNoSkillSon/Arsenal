+++
hypothesis = "Native half and affine graphs match the pinned checkpoint inventory."
change = "Add a native MLX retrieval adapter and schemas."
measurement = "Header-only inspection and synthetic tests; trained output remains pending."
verdict = "inconclusive"
numbers = { tensor_payload_bytes = 0 }
links = ["../README.md"]
+++

# Native integration, 18 October 2026

Pinned checkpoint headers inspected with exact Range reads; no tensor payloads fetched. The source is BF16. The native loader casts source tensors to FP16 before execution; int8/int4 use group-64 affine packs. This cast has not been qualified on trained activations.

See README for the actual port scope and pending trained-weight checks.
