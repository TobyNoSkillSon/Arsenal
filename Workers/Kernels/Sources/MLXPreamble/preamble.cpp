// SPDX-License-Identifier: MIT
#include "MLXPreamble.h"

// Defined in mlx-swift's Cmlx (mlx-generated/utils.cpp); declared here because the header is internal to MLX.
namespace mlx::core::metal {
const char* utils();
}

extern "C" const char *arsenal_mlx_metal_utils(void) { return mlx::core::metal::utils(); }
