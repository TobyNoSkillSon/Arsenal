+++
hypothesis = "The native graph matches the pinned checkpoint and exposes the documented tasks."
change = "Add the language-family adapter, model schemas and pinned source files."
measurement = "Header-only source inspection; trained-weight fidelity and precision quality are pending."
verdict = "inconclusive"
numbers = { header_tensors = 2130 }
links = ["../README.md"]
+++

# Native integration, 18 October 2026

No trained weights downloaded. Exact checkpoint headers were fetched using validated HTTP 206 Range reads. Shape mapping, random graph execution and trained-weight fidelity have separate outcomes; see the final G1 report.

Current checks: exact header comparison passes with explicit Get cast plans; graph tensors 1324; ignored source tensors 806. Int8/int4 packed 47/47 eligible reduced-graph matrices. All 12 mode/input fixtures satisfy their declared expectations; malformed input, cancellation and unload pass. Parser fixture tests cover positive schema-valid structured results. Trained-weight/reference/precision qualification remains pending.
