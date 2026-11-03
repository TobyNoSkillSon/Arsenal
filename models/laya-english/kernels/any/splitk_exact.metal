    // SPDX-License-Identifier: MIT
    // out[M, N] = x[M, K] . w[N, K]^T (+ bias) (+ res), exactly as MLX computes it when it routes the GEMM to its NAX
    // split-K kernel (steel_gemm_splitk_nax + steel_gemm_splitk_accum[_axpby], then the separate residual Add):
    // partition p sums K in [p*PS, min((p+1)*PS, K)) in 32-wide steps of two 16-wide mma's; partials are added in
    // partition order onto 0.0f; bias is MLX's axpby epilogue T(acc*1 + T(1)*bias); the residual is a T add.
    // One threadgroup = one 32 x 32 output tile; simdgroup s = (partition s / 2, row half s % 2) owns 16 x 32 of it
    // (each output element still sees MLX's exact sequence of mma steps). Partials meet in threadgroup memory, so the
    // fp32 scratch, the accumulate dispatch and the residual dispatch disappear. The host routes only shapes MLX
    // itself splits (same rule and partition size), or the numbers would differ.
    const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
    const int sg = int(simdgroup_index_in_threadgroup), p = sg / 2, half_row = sg % 2;
    const int row0 = int(threadgroup_position_in_grid.x) * 32 + half_row * 16, col0 = int(threadgroup_position_in_grid.y) * 32;
    const int k0 = p * PS, k1 = min(k0 + PS, K);
    const short lane = short(thread_index_in_simdgroup), qid = lane >> 2;
    const short fm = (qid & 4) | ((lane >> 1) & 3);
    const short fn = ((qid & 2) | (lane & 1)) * 4;
    const bool tile_full = row0 + 16 <= M && col0 + 32 <= N;
    vec<float, 8> C0 = vec<float, 8>(0.0f), C1 = vec<float, 8>(0.0f);
    if (row0 < M) {
        for (int k = k0; k < k1; k += 32) {
            vec<T, 8> A0, A1, B00, B01, B10, B11;
            if (tile_full && k + 32 <= k1) {
                sk_load(A0, x, K, row0, k, fm, fn);       sk_load(A1, x, K, row0, k + 16, fm, fn);
                sk_load(B00, w, K, col0, k, fm, fn);      sk_load(B01, w, K, col0, k + 16, fm, fn);
                sk_load(B10, w, K, col0 + 16, k, fm, fn); sk_load(B11, w, K, col0 + 16, k + 16, fm, fn);
            } else {
                sk_load_safe(A0, x, K, row0, k, M, k1, fm, fn);       sk_load_safe(A1, x, K, row0, k + 16, M, k1, fm, fn);
                sk_load_safe(B00, w, K, col0, k, N, k1, fm, fn);      sk_load_safe(B01, w, K, col0, k + 16, N, k1, fm, fn);
                sk_load_safe(B10, w, K, col0 + 16, k, N, k1, fm, fn); sk_load_safe(B11, w, K, col0 + 16, k + 16, N, k1, fm, fn);
            }
            sk_mma(C0, C1, A0, B00, B10);
            sk_mma(C0, C1, A1, B01, B11);
        }
    }
    threadgroup float part[P][32 * 32];
    for (short i = 0; i < 8; i++) {
        const int r = half_row * 16 + fm + (i >> 2) * 8, c = fn + (i & 3);
        part[p][r * 32 + c] = C0[i];
        part[p][r * 32 + 16 + c] = C1[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int tile_row = int(threadgroup_position_in_grid.x) * 32;
    for (int e = int(thread_index_in_threadgroup); e < 32 * 32; e += 64 * P) {
        const int m = tile_row + e / 32, n = col0 + e % 32;
        if (m >= M || n >= N) continue;
        float acc = 0.0f;
        for (int q = 0; q < P; q++) acc += part[q][e];
        T v;
        if (HAS_BIAS) v = static_cast<T>(acc * 1.0f + (static_cast<T>(1.0f) * static_cast<T>(bias[n])));
        else v = static_cast<T>(acc);
        if (HAS_RES) v = static_cast<T>(res[m * N + n]) + v;
        out[m * N + n] = v;
    }
