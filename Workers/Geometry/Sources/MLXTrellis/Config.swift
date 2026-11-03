// SPDX-License-Identifier: MIT
import MLX

/// Every neural stage uses native FP16 SDPA, including the selected affine tiers.
public enum TRELLIS2Config {
    public static let sdpaCastDtype: DType? = .float16
}
