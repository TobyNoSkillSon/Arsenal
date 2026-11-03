+++
hypothesis = "The pinned Qwen vendor template can switch thinking per request while the native runner keeps reasoning separate from the answer and respects folder-delivered defaults."
change = "Pass enable_thinking to the vendor Jinja, split closing reasoning markers across detokenizer chunks, expose reasoning events/final reasoning, merge a default system message, and prioritize canonical load settings."
measurement = "18 CPU contract/tokenizer tests pass. Real int4 settings smoke answered correctly with thinking off/on, streamed 49 reasoning events before 3 answer events, followed the system prompt, cancelled during reasoning, and exited on unload."
verdict = "kept"
numbers = { cpu_tests = 18, supported_context_max = 262144, network_weight_bytes = 0, int4_load_s = 1.303874, off_ttft_s = 3.488445, on_ttft_s = 0.387377, on_tokens_per_s = 25.935618, reasoning_events = 49, answer_events = 3, reasoning_chars = 161, peak_gib = 14.983852, cancel_settled_s = 0.044688 }
links = ["../../../docs/notes/v2-chat.md", "../../../Workers/Chat/evidence/v2-settings/"]
+++

The only source is the existing Arsenal-dev int4 derivative of the verified vendor bf16 copy. No downloads,
community quants, sibling-app model mutations or dependency-pin changes are needed.

CPU tests exercise the actual pinned vendor Jinja with thinking off/on, with an injected system prompt and
with an explicit system message that wins. Thinking-on adds the vendor's own xhigh reasoning-effort preamble;
this is preserved, not overridden. Delimiter tests split `</think>` at every character boundary, including
an output cut off before reasoning ends. Stop strings operate on answer text only. Nonstreaming requests
still return the separate `reasoning` field.

Load reads `config.settings.context_limit` before legacy `config.context_limit`; missing settings retain
the v1 8192-token fallback. The runner accepts up to the vendor's 262144 limit so unsupported UI choices
do not require a future runner rewrite. **128K/256K context: not verified on this Mac.** These are not offered
in the folder's popover: offered choices remain 8K/32K. A short ≤180-second smoke does not qualify long
prefill, quality at long context, or peak memory at long context.

The verification harness uses the real ArsenalCore.Supervisor, whose existing first-response and stall
deadlines refresh on reasoning events as on other stream events. Cancellation and memory/status paths
are unchanged. There is no new idle timer. The bounded gpulock proof checks thinking off/on, system-prompt
override, canonical context precedence, cancellation during reasoning, unload and PID disappearance.

The first final-build attempt was stopped by the coordinator while replacing the old Vella work-wrapper
wait policy; exit 241 was not a code failure. All build/test commands use `scripts/mlock work`, and the
real inference proof uses `scripts/mlock gpulock chat2`. Figures from that proof are functional-smoke
diagnostics, not ranked/controlled benchmark numbers. No `measure` job is run by this worker.

**Ran for real, int4:** preflight found 71.5 GiB free plus estimated reclaimable (24 GiB conservative
threshold), without evicting any sibling-app model. Canonical context 4096 beat legacy 2048. Thinking-off
returned `17 plus 25 equals 42.` in 11 answer fragments with no reasoning. Thinking-on returned 161
reasoning characters in 49 fragments, then the answer `42` in 3 fragments; final fields exactly matched
their separate streams. The system default changed the arithmetic answer to `PINEAPPLE`. Cancellation
during the first reasoning event settled in 0.045 seconds, status/memory worked, and unload verified
the worker PID was gone. Load was 1.304 seconds, peak lifetime footprint 14.984 GiB. Cold thinking-off
TTFT includes first-request kernel work; do not infer a thinking speed advantage from those TTFTs.

Evidence: `Workers/Chat/evidence/v2-settings/{cpu-checks,memory-preflight,smoke-settings-int4}.json`.
Raw CPU/build/smoke logs are retained locally alongside those files, outside /tmp, and ignored by Git.
**Did not run:** bf16/int8 for this change, long-context prefill or quality checks, controlled benchmarks,
vision, tool decoding. No worker, proof, gpulock or build child remains running after completion.
