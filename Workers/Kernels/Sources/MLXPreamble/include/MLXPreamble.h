// SPDX-License-Identifier: MIT
#ifndef ARSENAL_MLX_PREAMBLE_H
#define ARSENAL_MLX_PREAMBLE_H
#ifdef __cplusplus
extern "C" {
#endif
/// MLX's Metal preamble (mlx::core::metal::utils()): the text MLX prepends to every custom-kernel source.
const char *arsenal_mlx_metal_utils(void);
#ifdef __cplusplus
}
#endif
#endif
