    // SPDX-License-Identifier: MIT
    // SmallMGEMM tile kernel tile-1, body (MLXFast.metalKernel source). x [M, K] and w [N, K] row-major T; out [M, N] =
    // x . w^T (+ bias[n]) (+ res[m, n]). One threadgroup per 32 x 32 output tile; K split in 16-wide chunks over the SG
    // simdgroups (split-K: float partial sums added in threadgroup memory, rounded once). Not bit-identical to MLX: Fast.
    const int M = x_shape[0], K = x_shape[1], N = w_shape[0];
    const int tm = threadgroup_position_in_grid.x, tn = threadgroup_position_in_grid.y;
    const int sg = simdgroup_index_in_threadgroup;
    const int chunks = K / 16, base = chunks / SG, extra = chunks % SG;
    const int k0 = 16 * (sg * base + min(sg, extra)), kc = 16 * (base + (sg < extra ? 1 : 0));
    auto A = tensor<device T, dextents<int32_t, 2>, tensor_inline>((device T*)x + k0, dextents<int32_t, 2>(kc, M), array<int, 2>{1, K});
    auto B = tensor<device T, dextents<int32_t, 2>, tensor_inline>((device T*)w + k0, dextents<int32_t, 2>(kc, N), array<int, 2>{1, K});
    constexpr auto desc = matmul2d_descriptor(BM, BN, static_cast<int>(dynamic_extent), false, true, false);
    matmul2d<desc, execution_simdgroup> op;
    auto mA = A.slice(0, tm * BM);
    auto mB = B.slice(0, tn * BN);
    auto cT = op.template get_destination_cooperative_tensor<decltype(mA), decltype(mB), float>();
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) { if (cT.is_valid_element(i)) cT[i] = 0; }
    op.run(mA, mB, cT);
    threadgroup float red[SG][BM * BN];
    #pragma unroll
    for (uint16_t i = 0; i < cT.get_capacity(); ++i) {
        if (!cT.is_valid_element(i)) continue;
        auto idx = cT.get_multidimensional_index(i);
        red[sg][idx[1] * BN + idx[0]] = cT[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int e = thread_index_in_threadgroup; e < BM * BN; e += SG * 32) {
        float s = 0.0f;
        #pragma unroll
        for (int j = 0; j < SG; ++j) s += red[j][e];
        const int n = tn * BN + e % BN, m = tm * BM + e / BN;
        if (n < N && m < M) {
            if (HAS_BIAS) s += static_cast<float>(bias[n]);
            if (HAS_RES) s += static_cast<float>(res[m * N + n]);
            out[m * N + n] = static_cast<T>(s);
        }
    }
