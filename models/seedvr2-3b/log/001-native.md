+++
hypothesis = "The pinned SeedVR2 Swift closure can serve image and video restoration without a runtime download path."
change = "Vendor the source closure, add model-owned native adapter, explicit schemas, checkpoint mapping and whole-component affine recipes."
measurement = "Header-only counts and shapes matched the full native graph; small random graphs exercised both operations at all tiers through production output validation. Trained weights were not fetched."
verdict = "inconclusive"
numbers = { checkpoint_tensors = 996, eligible_matrices = 267, eligible_values = 3969155072 }
links = ["../README.md", "../gate/header-check.json"]
+++

Random graph wiring and exact inventory are verified. Trained-output correctness, FP16 adaptation and precision quality remain unqualified. No benchmark ran.
