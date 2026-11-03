+++
hypothesis = "The MLX kit runner reproduces the maker's AutoGluon Mitra zero-shot classification with the Mitra-v2 weights."
change = "None: measurement only."
measurement = "autogluon.tabular 1.6.3 Mitra Preprocessor + Tab2D on MPS under torch.autocast fp16 vs the kit runner on the same fixed inputs, m5-max, 2026-10-31."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { wine_same_labels = "36/36", wine_prob_cosine = 0.99999985, wine_prob_max_abs_diff = 0.0016, awkward_same_labels = "2/2", stress_same_labels = "699/700", stress_prob_cosine = 0.9999962, stress_prob_max_abs_diff = 0.0123, wine_ref_fp16_vs_fp32_max_abs_diff = 0.0035 }
+++
# 003 Parity with the maker's code

Reference: the maker's Mitra code from the autogluon.tabular 1.6.3 wheel (`models/mitra/_internal`: `Preprocessor`,
`Tab2D.from_pretrained` on the pinned FP32 snapshot edada0d), run on MPS under `torch.autocast` float16 (the maker's
`predict` uses autocast with bfloat16; fp16 and fp32 were run too). This is MitraClassifier.predict_proba's zero-shot
path: fine_tune=False, one estimator, and the random preprocessing switched off (shuffle_classes, shuffle_features and
random_mirror_x False), so the reference is deterministic. Labels are encoded in sorted order; softmax over the real
classes. The rest of autogluon (core, features) was not installed: it would downgrade torch in the shared env.

Candidate: the kit runner (`runners/swift`, MLX, FP16 weights; column quantile embedding on the host in Float32), driven
over the wire. Compared: labels, then every class probability.

| input | rows | same labels | prob max abs diff | prob cosine |
|---|---|---|---|---|
| wine (card example: load_wine, stratified 80/20, seed 42; 142 labelled, 36 to label, 13 columns, 3 classes) | 36 | 36 | 0.0016 | 0.99999985 |
| awkward (3 labelled rows, one constant column dropped, 2 classes) | 2 | 2 | 0.0008 | 0.9999996 |
| stress (make_classification seed 0: 2048 labelled, 700 to label, 20 columns, 4 integer-rounded, 5 classes) | 700 | 699 | 0.0123 | 0.9999962 |

The one stress difference, row 505, is an exact tie in the fp16 reference (0.4935 for classes 0 and 4); the maker's code
in fp32 picks class 4, as the runner does. For scale, the reference in fp16 and fp32 disagrees on 2 of the 700 labels, and
bf16 (the maker's default) on 5; the runner disagrees with fp32 on 1. Wine accuracy against the truth is 36/36 for every
variant. Verdict: pass (identical labels wherever the reference is not tied; probabilities at the fp16 floor).

Scripts: lab/night2/mitra-classifier-2/{make_data.py,make_big.py,ref.py,drive.py,compare.py} (not tracked).
