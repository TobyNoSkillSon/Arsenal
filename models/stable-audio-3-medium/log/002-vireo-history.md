+++
hypothesis = "Preserve gated fp16 residual arithmetic, exact promoted-weight copies, and exact strided decoder windows."
change = "Ported accepted paths and preserved the source rejection decisions."
measurement = "Historical Vireo source evidence on M5 Max, September 2026; no new Arsenal measurement."
verdict = "kept"
numbers = { historical_stock_step_10s_ms = 47, historical_fp16_step_10s_ms = 23.5, historical_exact_step_10s_ms = 31, historical_stock_render_30s_ms = 619, historical_views_render_30s_ms = 548, historical_fast_music_mel_db = 0.260 }
links = ["Workers/Sound/PORTED-FROM.md", "docs/evidence/mb-v2-sound/vireo-history-excerpts.txt"]
+++

Kept in Vireo: the fp16 DiT stream (≤1 dB self-test, 0.127 dB SFX / 0.260 dB music against 12.8 / 7.7 dB seed spread); exact FP32 copies only of promoted weights; SAME-L strided windows (three decoder cases bit-identical; 30-second render 619→548 ms, decode 302→234 ms). Arsenal ports these paths and their gates. The self-test runs on this Mac at load; historical evidence is not a cached Arsenal gate verdict.

Rejected in Vireo: fp16 decoder GEMMs (30-second render 612→612 ms, energy arms overlapped); batching decoder windows (609→643 ms); DiT SmallMGEMM (554→565 ms at 30 seconds, energy tied). These are not default paths in this port.

Source: Vireo `lab/perf/NOTES.md`, read from the local research working tree with HEAD `9366da5e14dcafe846fc41565c94f4f145eaf7cc` (the lab note is not Git-tracked); excerpts are retained in Arsenal evidence. No historical result is placed in Arsenal `results/` or used for its badge/ranking.
