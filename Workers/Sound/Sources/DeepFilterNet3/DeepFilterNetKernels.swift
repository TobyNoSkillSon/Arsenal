// SPDX-License-Identifier: MIT
// GPU replacements for the parts the mlx-audio-swift port ran on the CPU: the GRU recurrence (it used Accelerate's
// vDSP_mmul per frame) and libdf's running feature normalisation (exact sequential loops). Both are sequential in time,
// so each runs as one Metal kernel launch that walks the frames, with float accumulators.
import Foundation
import MLX

public enum DeepFilterNetKernels {
    /// One PyTorch GRU layer over time for batch 1 (gate order r, z, n; n = tanh(x_n + r * (W_hn h + b_hn))).
    /// One threadgroup of `hidden` threads; thread k owns hidden unit k and reads h from threadgroup memory.
    static let gruKernel = MLXFast.metalKernel(
        name: "arsenal_dfn3_gru_sequence",
        inputNames: ["gx", "whh_t", "bhh", "steps"],
        outputNames: ["out"],
        source: """
            const uint k = thread_position_in_threadgroup.x;
            const uint H = threads_per_threadgroup.x;
            const int T = steps[0];
            threadgroup float h[1024];
            h[k] = 0.0f;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (int t = 0; t < T; t++) {
                float ar = 0.0f, az = 0.0f, an = 0.0f;
                for (uint i = 0; i < H; i++) {
                    const float hi = h[i];
                    const uint row = i * 3 * H;
                    ar += hi * float(whh_t[row + k]);
                    az += hi * float(whh_t[row + H + k]);
                    an += hi * float(whh_t[row + 2 * H + k]);
                }
                const uint g = uint(t) * 3 * H;
                const float r = 1.0f / (1.0f + exp(-(float(gx[g + k]) + ar + float(bhh[k]))));
                const float z = 1.0f / (1.0f + exp(-(float(gx[g + H + k]) + az + float(bhh[H + k]))));
                const float n = tanh(float(gx[g + 2 * H + k]) + r * (an + float(bhh[2 * H + k])));
                const float next = (1.0f - z) * n + z * h[k];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                h[k] = next;
                out[uint(t) * H + k] = T_OUT(next);
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            """,
        ensureRowContiguous: true)

    /// `gx` [T, 3H] (input projection plus input bias), `whhT` [H, 3H], `bhh` [3H] → [T, H] in `gx`'s dtype.
    public static func gru(gx: MLXArray, whhT: MLXArray, bhh: MLXArray, hidden: Int) -> MLXArray {
        precondition(hidden <= 1024 && gx.dim(1) == 3 * hidden && whhT.shape == [hidden, 3 * hidden])
        let steps = gx.dim(0)
        guard steps > 0 else { return MLXArray.zeros([0, hidden], dtype: gx.dtype) }
        return gruKernel(
            [gx, whhT.asType(gx.dtype), bhh.asType(gx.dtype), MLXArray([Int32(steps)])],
            template: [("T_OUT", gx.dtype)],
            grid: (hidden, 1, 1), threadGroup: (hidden, 1, 1),
            outputShapes: [[steps, hidden]], outputDTypes: [gx.dtype])[0]
    }

    /// libdf's exponential running mean: s_t = x_t (1 - a) + s_{t-1} a from `initial`, one thread per column.
    static let emaKernel = MLXFast.metalKernel(
        name: "arsenal_dfn3_running_mean",
        inputNames: ["x", "initial", "alpha", "steps"],
        outputNames: ["out"],
        source: """
            const uint c = thread_position_in_grid.x;
            const uint C = threads_per_grid.x;
            const int T = steps[0];
            const float a = alpha[0];
            float s = initial[c];
            for (int t = 0; t < T; t++) {
                const uint i = uint(t) * C + c;
                s = x[i] * (1.0f - a) + s * a;
                out[i] = s;
            }
            """,
        ensureRowContiguous: true)

    /// `x` [T, C] float32 → running state [T, C] float32.
    public static func runningMean(_ x: MLXArray, initial: [Float], alpha: Float) -> MLXArray {
        let steps = x.dim(0), columns = x.dim(1)
        precondition(initial.count == columns)
        guard steps > 0 else { return x.asType(.float32) }
        return emaKernel(
            [x.asType(.float32), MLXArray(initial), MLXArray([alpha]), MLXArray([Int32(steps)])],
            grid: (columns, 1, 1), threadGroup: (min(columns, 256), 1, 1),
            outputShapes: [[steps, columns]], outputDTypes: [.float32])[0]
    }
}
