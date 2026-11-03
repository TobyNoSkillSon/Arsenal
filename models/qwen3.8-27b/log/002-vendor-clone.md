+++
hypothesis = "The existing local vendor checkpoint is exactly the pinned Hugging Face source, so no network weight transfer or community quant source is needed."
change = "Used Arsenal's catalog pin tool for HF metadata, cp -c for our independent APFS copies, and checked every cloned file's SHA256 and byte count."
measurement = "All 31 vendor files matched the immutable revision; source size/mtime/ctime stayed unchanged."
verdict = "kept"
numbers = { vendor_files = 31, vendor_bytes = 55586113293, network_weight_bytes = 0 }
links = ["runners/native/vendor-verified.json", "../../Workers/Chat/scripts/clone-vendor.py"]
+++

Source: `~/.omlx/models/Qwen/Qwen3.8-27B`, read-only for this worker.
Owned copy: `~/Library/Application Support/Arsenal-dev/weights/chat/qwen3.8-27b/bf16/`.
`arsenal-catalog pin qwen3.8-27b` used pinned HF LFS metadata and small Git files. Hashing the existing cloned
bytes then confirmed tensor identity. No weights were downloaded, no source file or oMLX/MTPLX model folder
was mutated, and our copy has independent inodes (APFS clones, not hardlinks). The `.gitattributes` control
file is not required at runtime; all 31 checkpoint files present in the source are pinned and verified.
