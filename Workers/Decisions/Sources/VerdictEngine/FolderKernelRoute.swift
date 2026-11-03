// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import ArsenalKernels
import VerdictSmallMGEMM

/// Folder kernels for the decisions runner (docs/KERNELS.md). One site: `linear.down`, the K > N Linears
/// (Laya mlp.Wo 2624→1024 + residual, head linear2 4096→1024 + bias + residual, Multilingual 1152→768 + residual).
/// Two kernels the folder may resolve to it, told apart by `entry`:
/// - `smallm_tile` (Fast): the SmallMGEMM tile-1 kernel; served through SmallMRoute and gated by SmallMGate as before.
/// - `splitk_exact` (Exact): MLX's own NAX split-K GEMM in one dispatch with the bias/residual epilogue; serves only
///   the calls MLX itself routes to split-K, with MLX's partition size, so the numbers are MLX's; gated bit for bit.
public enum FolderKernelRoute {
    public static let site = "linear.down"

    /// MLX 0.32's routing for x[M, K] · W[N, K]ᵀ (backend/metal/matmul.cpp, Matmul/AddMM::eval_gpu): NAX split-K when
    /// NAX is available, the dtype is half, the batch collapses into M, M ≥ 16 (below that `gemv_wide`/`gemv` run) and
    /// K ≥ 3·max(M, N) or (max(M, N) ≤ 1024 and K > 2·max(M, N)). Partition size as `steel_gemm_splitk_axpby_nax`.
    /// nil: MLX would not split this GEMM, so the exact kernel must not run it.
    public static func splitKPlan(m: Int, n: Int, k: Int, dtype: DType) -> (partition: Int, partitions: Int)? {
        guard SmallMGEMM.tensorOpsAvailable, dtype == .float16 || dtype == .bfloat16, m >= 16, n >= 2 else { return nil }
        let mn = max(m, n)
        guard k >= 3 * mn || (mn <= 1024 && k > 2 * mn) else { return nil }
        let partition = k <= 1024 ? k / 2 : k <= 2048 ? 1024 : k <= 4096 ? 2048 : 4096
        guard partition > 0 else { return nil }
        let partitions = (k + partition - 1) / partition
        // Threadgroup memory holds one 32 × 32 float tile per partition (4 = 16 KB); 2 simdgroups per partition.
        return partitions <= 4 ? (partition, partitions) : nil
    }

    /// `residual + layer(x)` (or `layer(x)`) on the exact split-K kernel, or nil (caller: its stock expression).
    static func splitK(_ x: MLXArray, _ layer: Linear, residual: MLXArray?, kernel: FolderMLXKernel) -> MLXArray? {
        guard type(of: layer) == Linear.self else { return nil }
        let w = layer.weight
        guard w.ndim == 2, x.dtype == w.dtype else { return nil }
        let k = x.dim(-1), n = w.dim(0), m = x.size / max(k, 1)
        guard w.dim(1) == k, let plan = splitKPlan(m: m, n: n, k: k, dtype: x.dtype) else { return nil }
        if let bias = layer.bias, bias.dtype != x.dtype || bias.size != n { return nil }
        if let residual, residual.dtype != x.dtype || residual.size != m * n { return nil }
        let placeholder = MLXArray([Float(0)])
        guard let y = kernel([x.reshaped([m, k]), w, layer.bias ?? placeholder, residual?.reshaped([m, n]) ?? placeholder],
                             template: [("T", x.dtype), ("PS", plan.partition), ("P", plan.partitions),
                                        ("HAS_BIAS", layer.bias != nil), ("HAS_RES", residual != nil)],
                             grid: ((m + 31) / 32 * 64 * plan.partitions, (n + 31) / 32, 1), threadGroup: (64 * plan.partitions, 1, 1),
                             outputShapes: [[m, n]], outputDTypes: [x.dtype])?.first else { return nil }
        return y.reshaped(Array(x.shape.dropLast()) + [n])
    }
}
