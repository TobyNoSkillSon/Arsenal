+++
hypothesis = "The Geneformer-V2-104M files can be pinned without the repository's root V2-316M checkpoint."
change = "Scaffold with the kit's own code, replacing only its file selection with the 104M files and their gene token dictionary."
measurement = "Pins checked against Hugging Face at the revision of 001-plan; the root checkpoint was not downloaded."
verdict = "kept"
numbers = { token_dictionary_bytes = 425590 }
links = ["../model.toml", "001-plan.md"]
+++
# 004 Variant pins (2026-10-31)

The kit's scaffold pins root checkpoints only, and this repo's root model.safetensors is Geneformer-V2-316M (1.27 GB).
So the folder was scaffolded with the kit's own scaffold code through lab/night2/geneformer-v2-104m/scaffold_variant.py,
which only replaces the kit's file selection: it pins the planned Geneformer-V2-104M files (same revision and SHA-256
as 001-plan) plus geneformer/token_dictionary_gc104M.pkl (425,590 bytes), the 104M gene token dictionary the runner
needs to map Ensembl IDs to tokens. Download, hash checks against Hugging Face and model.toml are the kit's.
The root V2-316M checkpoint was not downloaded.
