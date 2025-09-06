import torch
from torch.utils.cpp_extension import load
from utils import Benchmark

torch.set_grad_enabled(False)

lib = load(
    name="reduce_lib",
    sources=["../src/cuda/kernels/reduce.cu"],
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
        [lib.reduce_sum_fp32, torch.sum],
        ["fp32", "fp32_th"],
        [x],
    )

    print("-" * 85)
    x_f16 = x.half().contiguous()
    benchmark.meval(
        [lib.reduce_sum_fp16, torch.sum],
        ["fp16", "fp16_th"],
        [x_f16],
    )
    print("-" * 85)
