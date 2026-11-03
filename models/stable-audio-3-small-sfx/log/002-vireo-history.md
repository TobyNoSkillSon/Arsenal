+++
hypothesis = "Preserve the accepted half residual stream; withhold the failed int4 tier and rejected decoder/DiT candidates."
change = "Ported accepted paths and preserved the source rejection decisions."
measurement = "Historical Vireo source evidence on M5 Max, September 2026; no new Arsenal measurement."
verdict = "kept"
numbers = { historical_stock_render_10s_ms = 175, historical_fast_render_10s_ms = 126, historical_fast_mel_db_max = 0.16, historical_int4_mel_db_min = 6, historical_int4_mel_db_max = 16, historical_int8_active_mb = 1247 }
links = ["Workers/Sound/PORTED-FROM.md", "docs/evidence/mb-v2-sound/vireo-history-excerpts.txt"]
+++

Kept in Vireo: half DiT residual stream; 5/10/30-second render medians 146/175/292→99/126/206 ms, 0.07–0.16 dB log-mel distance against a 6–11 dB seed spread. SAME-S has no SAME-L sliding window, so that lever does not apply.

Rejected/withheld: int4 produced 6–16 dB deviation (another seed's scale) and is not offered. Int8 remained offered for memory rather than speed (140–145 ms at 10 seconds, 0.45–1.2 dB; 1247 MB versus fp16 1622 MB). FP16 decoder GEMMs did not materially improve time; batching windows changed decode 80→84 ms. SmallMGEMM improved the 5-second case 97→89 ms but tied at 10/30 seconds and did not save energy; it is not a default. These are source decisions, not newly qualified Arsenal dtypes.

Source: Vireo `lab/perf/NOTES.md`, read from the local research working tree with HEAD `9366da5e14dcafe846fc41565c94f4f145eaf7cc` (the lab note is not Git-tracked); excerpts are retained in Arsenal evidence. No historical result is placed in Arsenal `results/` or used for its badge/ranking.
