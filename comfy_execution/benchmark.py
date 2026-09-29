"""Opt-in benchmark capture for a single prompt execution.

Records, for one workflow run: per-node timings (the operation timeline), a
CPU/RAM/GPU time series, and a static device/env snapshot — written as a JSON
record and announced via a `benchmark` server event.

Activation is opt-in and zero-cost when off: `BenchmarkSession.maybe_start`
returns ``None`` unless the run requested it via ``extra_data["benchmark"]`` or
the server was launched with ``--benchmark``. When it returns ``None`` no thread
is started, no hooks run, and callers no-op — so a normal run pays nothing.

Everything here is best-effort: a failure in sampling or writing is logged and
never propagates into workflow execution.
"""
from __future__ import annotations

import datetime
import json
import logging
import os
import platform
import subprocess
import sys
import time
from queue import Empty, Queue
from threading import Thread
from typing import Callable, Optional

import psutil
import torch

import comfy.cli_args
import comfy.model_management
import folder_paths

SCHEMA_VERSION = 1

# System sampler columns. RAM in bytes; CPU is percent of all cores since the
# previous tick. Captured on every backend, including CPU-only.
PSUTIL_QUERY = ["timestamp", "ram_used_bytes", "ram_total_bytes", "cpu_percent"]

# NVIDIA telemetry (memory MiB, utilization %, temperature C, power W, clocks MHz).
NVIDIA_SMI_QUERY = [
    "timestamp", "memory.used", "memory.total", "utilization.gpu", "utilization.memory",
    "temperature.gpu", "power.draw", "power.limit", "clocks.current.sm", "clocks.current.memory",
    "pcie.link.gen.current", "pcie.link.width.current",
]

# Apple Silicon: torch reports bytes held by live tensors and total bytes the
# Metal driver reserved. Both are unified system memory, not dedicated VRAM.
DEVICE_MEMORY_QUERY = ["timestamp", "current_allocated_bytes", "driver_allocated_bytes"]


def _cpu_model() -> str:
    try:
        if sys.platform == "darwin":
            return subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
        if sys.platform.startswith("linux"):
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if line.lower().startswith("model name"):
                        return line.split(":", 1)[1].strip()
        return platform.processor() or platform.machine()
    except Exception:
        return platform.machine()


def _comfyui_version() -> Optional[str]:
    try:
        import comfyui_version
        return getattr(comfyui_version, "__version__", None)
    except Exception:
        return None


def _call_psutil() -> str:
    mem = psutil.virtual_memory()
    return f"{time.perf_counter()},{mem.used},{mem.total},{psutil.cpu_percent(interval=None)}"


def _call_nvidia_smi(query_list: list[str]) -> str:
    return subprocess.check_output(query_list, stderr=subprocess.STDOUT).decode("utf-8").strip()


def _call_torch_mps() -> str:
    return f"{time.perf_counter()},{torch.mps.current_allocated_memory()},{torch.mps.driver_allocated_memory()}"


def _sampler_thread(out_q: Queue, in_q: Queue, interval: float, gpu_probe: Optional[Callable[[], str]]):
    while True:
        try:
            if gpu_probe is not None:
                out_q.put(("gpu", gpu_probe()))
            out_q.put(("psutil", _call_psutil()))
        except Exception as e:
            logging.debug(f"[benchmark] sampler stopping: {e}")
            break
        try:
            if in_q.get(timeout=interval) == "stop":
                break
        except Empty:
            pass
        except Exception:
            break


class BenchmarkSession:
    def __init__(self, check_interval: float):
        self._check_interval = check_interval
        self._t0 = time.perf_counter()
        self.schema_version = SCHEMA_VERSION
        self.created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        self.device_info = self._device_info()
        self.node_timings: list[dict] = []
        self.psutil_query_params = ", ".join(PSUTIL_QUERY)
        self.psutil_data: list[str] = []
        self.nvidia_smi_query_params = ", ".join(NVIDIA_SMI_QUERY)
        self.nvidia_smi_data: list[str] = []
        self.device_memory_query_params = ", ".join(DEVICE_MEMORY_QUERY)
        self.device_memory_data: list[str] = []
        self.total_execution_seconds: Optional[float] = None
        self._node_start: dict[str, float] = {}
        self._thread = None
        self._in_q: Optional[Queue] = None
        self._out_q: Optional[Queue] = None
        # Prime CPU% so the first sampled value is meaningful (see psutil docs).
        try:
            psutil.cpu_percent(interval=None)
        except Exception:
            pass

    @staticmethod
    def _backend() -> str:
        if comfy.model_management.is_nvidia():
            return "nvidia"
        if torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _device_info(self) -> dict:
        info = {
            "backend": self._backend(),
            "pytorch_version": str(getattr(torch, "__version__", "")),
            "python_version": sys.version.split()[0],
            "comfyui_version": _comfyui_version(),
            "operating_system": sys.platform,
            "platform": platform.platform(),
            "architecture": platform.machine(),
            "cpu_model": _cpu_model(),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "cpu_count_logical": psutil.cpu_count(logical=True),
        }
        try:
            info["name"] = comfy.model_management.get_torch_device_name(comfy.model_management.get_torch_device())
            info["total_vram_bytes"] = int(comfy.model_management.total_vram * (1024 ** 2))
            info["total_ram_bytes"] = int(comfy.model_management.total_ram * (1024 ** 2))
        except Exception as e:
            logging.debug(f"[benchmark] device info partial: {e}")
        return info

    def _gpu_probe(self) -> Optional[Callable[[], str]]:
        backend = self.device_info.get("backend")
        if backend == "nvidia":
            cuda_device = comfy.cli_args.args.cuda_device
            smi_id = [] if cuda_device is None else [f"--id={cuda_device}"]
            query_list = ["nvidia-smi", "--query-gpu=" + ",".join(NVIDIA_SMI_QUERY), "--format=csv,noheader,nounits", *smi_id]
            try:
                _call_nvidia_smi(query_list)  # verify it works before relying on it
            except Exception as e:
                logging.debug(f"[benchmark] nvidia-smi unavailable, GPU series disabled: {e}")
                return None
            return lambda: _call_nvidia_smi(query_list)
        if backend == "mps":
            return _call_torch_mps
        return None

    @classmethod
    def maybe_start(cls, extra_data: dict) -> Optional["BenchmarkSession"]:
        enabled = bool(extra_data.get("benchmark")) or getattr(comfy.cli_args.args, "benchmark", False)
        if not enabled:
            return None
        try:
            cfg = extra_data.get("benchmark") if isinstance(extra_data.get("benchmark"), dict) else {}
            interval = float(cfg.get("check_interval", 0.25))
            session = cls(interval)
            session._start_sampler()
            return session
        except Exception as e:
            logging.error(f"[benchmark] failed to start, skipping capture: {e}")
            return None

    def _start_sampler(self):
        gpu_probe = self._gpu_probe()
        self._out_q, self._in_q = Queue(), Queue()
        self._thread = Thread(target=_sampler_thread, args=(self._out_q, self._in_q, self._check_interval, gpu_probe))
        self._thread.daemon = True
        self._thread.start()

    def node_start(self, node_id: str):
        self._node_start[node_id] = time.perf_counter()

    def node_end(self, node_id: str, class_type: Optional[str]):
        start = self._node_start.pop(node_id, None)
        if start is None:
            return
        self.node_timings.append({
            "node_id": str(node_id),
            "class_type": class_type,
            "start_time": start,
            "elapsed_seconds": time.perf_counter() - start,
        })

    def _drain(self):
        if self._out_q is None:
            return
        while not self._out_q.empty():
            try:
                kind, line = self._out_q.get_nowait()
            except Exception:
                break
            if kind == "psutil":
                self.psutil_data.append(line)
            elif self.device_info.get("backend") == "mps":
                self.device_memory_data.append(line)
            else:
                self.nvidia_smi_data.append(line)

    def finish(self, server=None, prompt_id: Optional[str] = None) -> Optional[str]:
        self.total_execution_seconds = time.perf_counter() - self._t0
        try:
            if self._thread is not None and self._in_q is not None:
                self._in_q.put("stop")
                self._thread.join(timeout=5)
                self._drain()
        except Exception as e:
            logging.debug(f"[benchmark] error stopping sampler: {e}")
        return self._write(server, prompt_id)

    def _record(self, prompt_id: Optional[str]) -> dict:
        return {
            "schema_version": self.schema_version,
            "prompt_id": prompt_id,
            "created_at": self.created_at,
            "total_execution_seconds": self.total_execution_seconds,
            "device_info": self.device_info,
            "node_timings": self.node_timings,
            "psutil_query_params": self.psutil_query_params,
            "psutil_data": self.psutil_data,
            "nvidia_smi_query_params": self.nvidia_smi_query_params,
            "nvidia_smi_data": self.nvidia_smi_data,
            "device_memory_query_params": self.device_memory_query_params,
            "device_memory_data": self.device_memory_data,
        }

    def _write(self, server, prompt_id: Optional[str]) -> Optional[str]:
        record = self._record(prompt_id)
        path = None
        try:
            out_dir = os.path.join(folder_paths.get_output_directory(), "benchmark")
            os.makedirs(out_dir, exist_ok=True)
            stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            path = os.path.join(out_dir, f"benchmark_{stamp}.json")
            with open(path, "w") as f:
                json.dump(record, f, indent=2, default=str)
            logging.info(f"[benchmark] wrote {path}")
        except Exception as e:
            logging.error(f"[benchmark] failed to write record: {e}")
        try:
            if server is not None and hasattr(server, "send_sync"):
                server.send_sync("benchmark", {
                    "prompt_id": prompt_id,
                    "file": path,
                    "total_execution_seconds": self.total_execution_seconds,
                    "backend": self.device_info.get("backend"),
                    "node_count": len(self.node_timings),
                }, getattr(server, "client_id", None))
        except Exception as e:
            logging.debug(f"[benchmark] failed to emit event: {e}")
        return path
