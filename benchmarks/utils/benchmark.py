import time
from typing import Callable, List, Optional, Tuple

import torch
from termcolor import colored


class Benchmark:
    def __init__(self) -> None:
        self.kernel: Optional[Callable] = None
        self.tag: Optional[str] = None

    def init(self, kernel: Callable, tag: str) -> None:
        self.kernel = kernel
        self.tag = tag

    def deinit(self) -> None:
        self.kernel = None
        self.tag = None

    def eval(
        self,
        inputs: List[torch.Tensor],
        warmup: int = 10,
        iterations: int = 100,
        info: bool = False,
    ) -> Tuple[Optional[torch.Tensor], float]:
        out = None
        if self.kernel is None or self.tag is None:
            assert False, colored(
                "[ERROR] No kernel or no tag is set up for benchmarking!",
                "red",
                attrs=["bold"],
            )

        # warmup
        for _ in range(warmup):
            if out is None:
                out = self.kernel(*inputs)
            else:
                _ = self.kernel(*inputs)

        # torch.cuda.synchronize()

        start = time.time()
        # iters
        for _ in range(warmup):
            if out is None:
                out = self.kernel(*inputs)
            else:
                _ = self.kernel(*inputs)

        # torch.cuda.synchronize()
        end = time.time()
        total_time = (end - start) * 1000  # ms
        mean_time = total_time / iterations
        time_info = f"[INFO] `{self.tag}`: elasped time: {mean_time:.8f}ms"
        print(
            colored(
                time_info + f", type: {out.dtype}" if out is not None else time_info,
                "light_green",
                attrs=["bold"],
            )
        )
        if info:
            if out is not None:
                print(f"Shape: {tuple(out.shape)}")
                print(f"Output: {out}")
        self.deinit()
        return out, mean_time

    def meval(
        self,
        kernels: List[Callable],
        tags: List[str],
        inputs: List[torch.Tensor],
        warmup: int = 10,
        iterations: int = 100,
        info: bool = False,
    ) -> None:
        assert len(kernels) == len(tags), colored(
            "[ERROR] The number of kernels should be same as the number of tags!"
        )
        for i in range(len(kernels)):
            self.init(kernels[i], tags[i])
            self.eval(inputs, warmup=warmup, iterations=iterations, info=info)

        self.deinit()
