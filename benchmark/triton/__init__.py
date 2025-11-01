import sys
import time
from types import ModuleType, SimpleNamespace


class _Testing:
    def do_bench(self, fn, rep=50, warmup=10):
        """Lightweight replacement for triton.testing.do_bench.

        Returns average latency in milliseconds across `rep` runs after an optional warmup.
        """
        for _ in range(max(0, warmup)):
            fn()

        start = time.perf_counter()
        for _ in range(max(1, rep)):
            fn()
        end = time.perf_counter()

        return (end - start) * 1000.0 / max(1, rep)


testing = _Testing()


def cdiv(a, b):
    """Ceiling division helper used by the benchmark harness."""
    return (a + b - 1) // b


def jit(fn=None, **_kwargs):
    """Dummy decorator to keep the benchmark script importable without Triton."""

    if fn is None:
        def wrapper(func):
            return func

        return wrapper

    return fn


# Minimal placeholder for triton.language as `tl`.  The benchmark never executes
# the Triton kernels in this environment, so an empty namespace is sufficient.
_language_module = ModuleType("triton.language")


def _stub(*_args, **_kwargs):
    raise NotImplementedError("Triton kernels are not available in this benchmarking environment.")


_language_module.constexpr = lambda x: x
_language_module.program_id = _stub
_language_module.load = _stub
_language_module.arange = _stub
_language_module.zeros = _stub
_language_module.cdiv = lambda a, b: (a + b - 1) // b
_language_module.minimum = _stub
_language_module.dot = _stub
_language_module.where = _stub
_language_module.trans = _stub
_language_module.maximum = _stub
_language_module.exp = _stub
_language_module.sum = _stub
_language_module.store = _stub
_language_module.log = _stub
_language_module.float32 = "float32"

language = tl = _language_module

sys.modules[__name__ + ".language"] = _language_module


__all__ = ["testing", "cdiv", "jit", "language", "tl"]
