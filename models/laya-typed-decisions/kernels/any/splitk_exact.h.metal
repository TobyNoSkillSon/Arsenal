// SPDX-License-Identifier: MIT
// MLX's NAX split-K GEMM fused into one dispatch, bit-identical to it (Exact). Helpers below replay
// mlx::steel::BaseNAXFrag (load / load_safe / mma) and tile_matmad_nax from MLX 0.32 (mlx-swift 9019419,
// Copyright (c) 2025 Apple Inc., MIT): the same 16x16 fragment layout, the same matmul2d(16, 32, 16,
// relaxed, multiply_accumulate) descriptor and the same K order, so every partial sum is MLX's.
#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
using namespace metal;

// One 16x16 fragment of src (row-major, leading dimension ld) at (r0, c0): lane element i*4+j is
// row r0 + fm + 8i, column c0 + fn + j. Out-of-range elements are 0 (MLX's load_safe), in-range ones as MLX's load.
template <typename T, typename P>
METAL_FUNC void sk_load(thread vec<T, 8>& dst, P src, int ld, int r0, int c0, short fm, short fn) {
  src += (r0 + fm) * ld + c0 + fn;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; i++) {
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; j++) dst[i * 4 + j] = static_cast<T>(src[i * 8 * ld + j]);
  }
}
template <typename T, typename P>
METAL_FUNC void sk_load_safe(thread vec<T, 8>& dst, P src, int ld, int r0, int c0, int rlim, int clim, short fm, short fn) {
  src += (r0 + fm) * ld + c0 + fn;
  const int lr = rlim - r0 - fm, lc = clim - c0 - fn;
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 2; i++) {
    _Pragma("clang loop unroll(full)")
    for (short j = 0; j < 4; j++) dst[i * 4 + j] = (i * 8 < lr && j < lc) ? static_cast<T>(src[i * 8 * ld + j]) : T(0);
  }
}

// C[0], C[1] (two 16x16 output fragments side by side in N) += A (16x16) * [B0; B1]^T: MLX's mma.
template <typename T>
METAL_FUNC void sk_mma(thread vec<float, 8>& C0, thread vec<float, 8>& C1, const thread vec<T, 8>& A,
                       const thread vec<T, 8>& B0, const thread vec<T, 8>& B1) {
  constexpr auto desc = mpp::tensor_ops::matmul2d_descriptor(16, 32, 16, false, true, true,
      mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate);
  mpp::tensor_ops::matmul2d<desc, metal::execution_simdgroup> op;
  auto a = op.template get_left_input_cooperative_tensor<T, T, float>();
  auto b = op.template get_right_input_cooperative_tensor<T, T, float>();
  auto c = op.template get_destination_cooperative_tensor<metal::remove_addrspace_t<decltype(a)>, metal::remove_addrspace_t<decltype(b)>, float>();
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) { a[i] = A[i]; b[i] = B0[i]; b[8 + i] = B1[i]; c[i] = C0[i]; c[8 + i] = C1[i]; }
  op.run(a, b, c);
  _Pragma("clang loop unroll(full)")
  for (short i = 0; i < 8; i++) { C0[i] = c[i]; C1[i] = c[8 + i]; }
}
