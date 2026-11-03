+++
hypothesis = "The pinned Swift graph matches the complete checkpoint inventory and can execute every declared operation without trained weights."
change = "Vendor the model closure and add bounded native operations with FP16/int8/int4 loading."
measurement = "Header-only inventory, isolated random graph operations, model folder validation and family tests; no trained payloads."
verdict = "kept"
numbers = { weights_downloaded = 0, declared_operations = 2 }
links = ["../README.md"]
+++

# Native integration

18 October 2026. Source frozen from Hugging Face metadata and header-only Range reads. No tensor payloads read.

Implementation provenance: `Workers/Perception/licenses/`. FP16 is the default; int8/int4 cover every eligible Linear/Embedding in both encoders and any decoder. No trained-weight quality, speed or memory claim is made. Verification results are added after the real worker checks.

Historical initial verification (superseded by the review corrections below): 769 full-graph parameters matched header names/shapes; 336 full-graph eligible matrices. All 2 declared operations passed each of FP16/int8/int4 on the real worker with a shrunk graph. Malformed input, responsive cancellation, session closure where applicable, and exit-0 unload passed. Source dtype differences are recorded, not treated as trained cast-parity evidence.

Review correction: the initial random generations were incorrectly accepted as empty detections. Strict termination/grammar checks now reject them. No successful detection-output smoke is claimed from those random generations. All 769 source names/shapes still match, but BF16→FP16 header/quality qualification remains outstanding.
