"""
Probing manager

#### License
This source code is licensed under the terms specified in the `LICENSE` file,
located in the root directory of this repository.

@ 2025, Meta
"""

import uuid
from pathlib import Path, PosixPath
from types import TracebackType
from typing import Any

import torch
import torch.nn as nn
from torch._ops import OpOverload
from torch.autograd.function import FunctionCtx
from torch.utils._python_dispatch import TorchDispatchMode

PROBE_MODE: bool = False


# ------------------------------------------------------------------------------
# Probe hook
# ------------------------------------------------------------------------------


@torch.library.custom_op("probe::hook", mutates_args=(), device_types=None)
def _log(x: torch.Tensor, name: str, uid: str) -> None:
    pass


class ProbeModule(torch.autograd.Function):
    """
    Module applying a probe operator in a computational graph.
    """

    @staticmethod
    def forward(ctx: FunctionCtx, x: torch.Tensor, name: str) -> torch.Tensor:
        uid = str(uuid.uuid4())
        torch.ops.probe.hook(x, name, uid)
        ctx.name = name
        ctx.uid = uid
        return x

    @staticmethod
    def backward(ctx: FunctionCtx, grad: torch.Tensor) -> tuple:
        torch.ops.probe.hook(grad, f"{ctx.name}.g", ctx.uid)
        return grad, None, None


def log_stats(x: torch.Tensor, name: str) -> torch.Tensor:
    """
    Register a probe module in a computational graph.

    ### Parameters
    x: torch.Tensor to log
    name: name to log the tensor information

    The probe will call a operator register as torch.ops.probe,
    this operator should take as input the tensor (or its gradient), name (or {name}.g), and a uid as arguments.
    Its logic may be implemented in a torch_dispatcher that catch and define the probe operation.
    """
    if PROBE_MODE:
        return ProbeModule.apply(x, name)
    return x


# ------------------------------------------------------------------------------
# Examples of function to hook to the probe
# ------------------------------------------------------------------------------


def probe_stats(x: torch.Tensor) -> dict[str, Any]:
    shape = x.shape
    x = x.flatten()
    mean = x.mean()
    std = x.std()
    metrics = {
        "shape": shape,
        "mean": mean,
        "std": std,
        "skew": (((x - mean) / std) ** 3).mean(dtype=torch.double),
        "kurtosis": (((x - mean) / std) ** 4).mean(dtype=torch.double),
        "max": x.max(),
        "min": x.min(),
    }
    return metrics


def print_stats(x: torch.Tensor, name: str, uid: str) -> None:
    """
    Default probe logging function.

    ### Parameters
    x: torch.Tensor to log
    name: name to log the tensor information
    uid: unique identifier for the tensor
    """
    print(f"Logging {name} with UID {uid}:")
    stats = probe_stats(x)
    for key, value in stats.items():
        print(f"{key:10}: {value}")


SAVE_DIR: PosixPath = Path.home() / "probe"


def saved_probe_tensor(x: torch.Tensor, name: str, uid: str) -> None:
    """
    Default probe logging function.

    ### Parameters
    x: torch.Tensor to log
    name: name to log the tensor information
    uid: unique identifier for the tensor
    """
    if name.endswith(".g"):
        return
    save_dir = SAVE_DIR / name
    save_dir.mkdir(parents=True, exist_ok=True)
    torch.save(x, save_dir / f"{uid}.pt")


# ------------------------------------------------------------------------------
# Probe Context Manager
# ------------------------------------------------------------------------------


class Probe(TorchDispatchMode):
    """
    Probing Context Manager.

    ### Parameters
    func_hook: dictionary to hook operators to functions to call when the operator is called.
    """

    def __init__(self, func_hook: dict[OpOverload, callable] = None):
        super().__init__()
        self.func_hook = func_hook if func_hook else {}

    def __enter__(self) -> "Probe":
        global PROBE_MODE
        super().__enter__()
        PROBE_MODE = True
        return self

    def __exit__(self, exc: type[BaseException], value: BaseException, tb: TracebackType):
        global PROBE_MODE
        PROBE_MODE = False
        super().__exit__(exc, value, tb)

    @torch.no_grad()
    def __torch_dispatch__(self, func: OpOverload, types: tuple, args: tuple = (), kwargs: dict | None = None) -> Any:
        kwargs = kwargs if kwargs else {}
        out = func(*args, **kwargs)

        operator = func._overloadpacket

        if operator in self.func_hook:
            self.func_hook[operator](*args)
        return out


# ------------------------------------------------------------------------------
# Examples
# ------------------------------------------------------------------------------


if __name__ == "__main__":

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(10, 10)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            x = log_stats(x, "input")
            x = self.linear(x)
            x = log_stats(x, "output")
            return x

    # Example usage
    model = SimpleModel()
    x = torch.randn(1, 10)

    with Probe({torch.ops.probe.hook: saved_probe_tensor}):
        SAVE_DIR = SAVE_DIR / "epoch0"
        output = model(x)
        SAVE_DIR = SAVE_DIR.parent / "epoch1"
        output = model(x)

    with Probe({torch.ops.probe.hook: print_stats}):
        output = model(x)
