+++
hypothesis = "The pinned native TRELLIS graph can load original checkpoint layouts through shared streamed precision conversion."
change = "Adapt the generation, mesh and xatlas closure and expose image_to_3d with PBR GLB and mesh preview."
measurement = "No trained weights; exact safetensors header ranges, independent parameter inventory and isolated random worker graphs."
verdict = "inconclusive"
links = ["../README.md", "../../../docs/INTEGRATING.md"]
numbers = { trained_weights_downloaded = 0, precision_tiers = 3, components = 9, parameters = 4265 }
+++

# Native integration

2026-10-18. The original Microsoft weights are ungated safetensors. DINOv3's ungated mirror has the same weight hash as its gated original. FP16 is the serving default; cast and quantization quality await the trained reference trial. Structural counts and eligibility are retained in gate/header-check.json. No performance benchmark ran.
