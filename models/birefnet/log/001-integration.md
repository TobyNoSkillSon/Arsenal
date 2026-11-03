+++
hypothesis = "The pinned vendor implementation runs unchanged in half precision on Apple's GPU."
change = "Wrap the reviewed implementation in ArsenalWire, add removal schemas and stored affine precisions."
measurement = "Tiny fixed-tensor MPS deformable convolution and 128-square vendor smoke passed; full intake follows."
verdict = "inconclusive"
numbers = { source_bytes = 444473596 }
links = ["../README.md"]
+++

Source: ZhengPeng7/BiRefNet e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4.

Historical hypothesis: superseded by Swift MLX in log/004-native.md.
