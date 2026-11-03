# RE-USE engine: provenance and changes

The files below are adapted from SpeechLens (https://github.com/faraday/SpeechLens, `Sources/Inference/`,
commit `d20e11bcafbc32435ef18a17ed56caf22fca6833`), Copyright 2026 Çağatay Çallı, Apache-2.0
(`Sources/VireoEngine/Vendor/LICENSES/SpeechLens-Apache-2.0.txt`, notice in `SpeechLens-NOTICE.txt`).
`UPSTREAM-README.md` is SpeechLens' README for that module, kept for reference; the documents it links are not vendored.

| Files | Vireo changes |
|---|---|
| `ReUse{SEMamba,Layers,Mamba,MambaBlock,MetalSelectiveScan,MetalCausalConv1d,STFTFraming,StridedViewBounds,Settings,ScanMode,EnhancementSession,Enhancer,Log}.swift` | Types renamed with a `ReUse` prefix (they share the VireoEngine module); `import Numerics` dropped; the metallib probe and the process-wide `Memory.cacheLimit` removed (the helper owns both); weights mapped with `update(parameters:verify: .all)` and an FP32 check instead of an unverified update; PReLU per channel (`count: 64`), because the weights hold 64 alphas where upstream declared one; `ReUseKernels` (`metal`/`stock`) threaded through the module tree; OSLog subsystem renamed |
| `ReUseStockSelectiveScan.swift`, `ReUseKernelGate.swift`, `ReUseEngine.swift` | New in Vireo: plain-MLX reference scan (stock path and fallback), the on-device kernel self-test, the engine adapter |

Weights: NVIDIA's `nvidia/NVIDIA-NemotronLabs-RE-USE` checkpoint, OpenMDW-1.1 at the pinned revision `31f82f2e`
(`Workers/Sound/licenses/RE-USE-NVIDIA-OpenMDW-1.1.txt`; earlier revisions and the `faraday/re-use-mlx` conversion were
NVIDIA NSCLv1, non-commercial). Not bundled; downloaded on Get.

16 Oct 2026: the kernels run on half activations and follow the stock path's rounding points. The causal conv
accumulates in FP32, rounds once to the activation dtype and adds the bias in that dtype; the scan keeps FP32 softplus,
decay and state (MLX's `logAddExp`, `exp`, sequential sum, no FMA contraction) but evaluates `u * D` and `SiLU(z)` in the
activation dtype and rounds its output once. The stock causal conv pads explicitly and uses MLX's depthwise conv1d kernel
(`ReUseMamba.stockCausalConv`) instead of the padded `Conv1d` call, whose grouped GEMM rounded half products more coarsely.
