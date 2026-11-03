+++
hypothesis = "Folder-declared resolution, steps and seed can use the existing reviewed mflux path."
change = "Added schema 2 settings, accepted runner selection, input/output schemas and resolution-to-dimensions translation."
measurement = "CPU mocked wire fixture tests cover resolution defaults, explicit dimensions and malformed resolution. No FLUX weights were run by this worker."
verdict = "inconclusive"
numbers = { schema = 2, model_runs = 0, seed_max = 4294967295 }
links = ["../runners/mflux/test_wire.py", "../../../docs/evidence/mb-v2-schema/slice4-tests.txt", "../ops/generate.input.json"]
+++

The resolution setting chooses a square canvas; explicitly supplied width
and height always win. Steps/seed defaults apply at request time. This is a
wire/schema change, not a claim that the production 4B model has been tried
or measured. The existing real same-architecture stand-in record is preserved.
