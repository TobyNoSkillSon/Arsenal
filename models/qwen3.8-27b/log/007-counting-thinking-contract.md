+++
hypothesis = "Token counting should use the same explicit or saved thinking setting as chat."
change = "Declared optional thinking:boolean in the model count_tokens input schema and shared chat schema; the Anthropic adapter forwards mapped thinking."
measurement = "CPU service/schema regression proves an omitted caller field takes saved true and explicit false wins; API adapter tests cover forwarding. No tokenizer/model inference was run in this review-fix round."
verdict = "kept"
numbers = { real_model_runs = 0, mapping_cases = 2 }
links = ["../ops/count_tokens.input.json", "../../../Tests/ArsenalCoreTests/ReviewFixATests.swift", "../../../Tests/ArsenalAPITests/ReviewFixTests.swift"]
+++

<!-- SPDX-License-Identifier: MIT -->

The request schema previously did not declare `thinking`, so request-setting merge
skipped that default even though the chat template uses it. This fixes contract
coverage, not a claim about numerical/tokenizer accuracy or performance. The worker
remains responsible for counting the actual vendor template.
