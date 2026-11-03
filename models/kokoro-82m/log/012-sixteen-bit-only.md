+++
hypothesis = "A model whose 16-bit weights are under 1 GB gains nothing worth offering from int4."
change = "model.toml drops [derive] int4; the model ships its 16-bit format only. The withdrawn tiers' table results are removed; the intake runs and quality bindings stay as evidence for the 16-bit figures."
measurement = "No new run. Policy from Models whose 16-bit weights are under 1 GB ship 16-bit only (16-bit weights here 0.171 GB, from the pinned file sizes)."
verdict = "rejected"
numbers = { sixteen_bit_weights_gb = 0.171 }
links = ["../README.md", "011-int8-withdrawn.md"]
+++

# INT4 withdrawn: 16-bit only

26 October 2026. Models whose 16-bit weights are under 1 GB ship 16-bit only, with no 8-bit or 4-bit offered; every Mac has the memory for the 16-bit copy. This model's 16-bit weights are 0.171 GB, so int4 is withdrawn. Copies already on a Mac show as old precisions the settings card can remove.
