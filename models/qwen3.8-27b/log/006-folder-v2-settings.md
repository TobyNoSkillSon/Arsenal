+++
hypothesis = "Explicit settings and op schemas can expose this model without on-device runner racing."
change = "Migrated to folder schema 2, declared native default, context/thinking/temperature/system prompt settings and input/output schemas."
measurement = "CPU schema fixtures and signed bundle activation passed. This worker did not load weights or check longer context."
verdict = "inconclusive"
numbers = { schema = 2, model_runs = 0, default_context = 32768 }
links = ["../../../docs/evidence/mb-v2-schema/slice4-tests.txt", "../model.toml", "../ops/chat.input.json"]
+++

Context choices are only 8192 and 32768, as directed by the owner: those are the
values run on this Mac. The vendor native_context = 262144 is descriptive,
not evidence that a longer context is verified locally. 131072/262144 are
not offered until a separate long-context check exists. The chat worker owns
engine support and proof for the thinking toggle and system prompt; this
entry records schema work, not a new model-quality claim.
