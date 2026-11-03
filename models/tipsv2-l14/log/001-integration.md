+++
hypothesis = "The pinned Swift graph matches the complete checkpoint inventory and can execute every declared operation without trained weights."
change = "Vendor the model closure and add bounded native operations with FP16/int8/int4 loading."
measurement = "Header-only inventory, isolated random graph operations, model folder validation and family tests; no trained payloads."
verdict = "kept"
numbers = { weights_downloaded = 0, declared_operations = 3 }
links = ["../README.md"]
+++

# Native integration

18 October 2026. Source frozen from Hugging Face metadata and header-only Range reads. No tensor payloads read.

Implementation provenance: `Workers/Perception/licenses/`. FP16 is the default; int8/int4 cover every eligible Linear/Embedding in both encoders and any decoder. No trained-weight quality, speed or memory claim is made. Verification results are added after the real worker checks.

Verification: 491 full-graph parameters matched header names/shapes; 145 full-graph eligible matrices. All 3 declared operations passed each of FP16/int8/int4 on the real worker with a shrunk graph. Malformed input, responsive cancellation, session closure where applicable, and exit-0 unload passed. Source dtype differences are recorded, not treated as trained cast-parity evidence.
