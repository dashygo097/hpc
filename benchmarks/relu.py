import torch
from torch.utils.cpp_extension import load
from utils import Benchmark

torch.set_grad_enabled(False)

lib = load(
    name="relu_lib",
    sources=["../src/cuda/kernels/relu.cu"],
    extra_include_paths=["../include"],
    extra_cuda_cflags=[
        "-O3",
        "-U__CUDA_NO_HALF_OPERATORS__",
        "-U__CUDA_NO_HALF_CONVERSIONS__",
        "-U__CUDA_NO_HALF2_OPERATORS__",
        "-U__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "--expt-relaxed-constexpr",
        "--expt-extended-lambda",
        "--use_fast_math",
    ],
    extra_cflags=["-std=c++17"],
)

benchmark = Benchmark()

Ss = [1024, 2048, 4096]
Ks = [1024, 2048, 4096]
SKs = [(S, K) for S in Ss for K in Ks]

for S, K in SKs:
    print("-" * 85)
    print(" " * 40 + f"S={S}, K={K}")
    x = torch.randn((S, K)).cuda().float().contiguous()
    benchmark.meval(
        [lib.relu_fp32, lib.relu_fp32x2, lib.relu_fp32x4, torch.relu],
        ["fp32", "fp32x2", "fp32x4", "fp32_th"],
        [x],
    )

    print("-" * 85)
    x_f16 = x.half().contiguous()
    benchmark.meval(
        [lib.relu_fp16, lib.relu_fp16x2, lib.relu_fp16x8, torch.relu],
        ["fp16", "fp16x2", "fp16x8", "fp16_th"],
        [x_f16],
    )
    print("-" * 85)
