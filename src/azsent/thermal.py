"""GPU thermal guard: pauses training when the GPU runs hot, gates job starts.

Uses NVML (nvidia-ml-py). Degrades gracefully to a no-op when no NVIDIA GPU or
NVML is unavailable (e.g. CPU smoke tests).
"""
from __future__ import annotations

import csv
import time
from pathlib import Path

from .utils import log


class ThermalGuard:
    def __init__(self, cfg_thermal, log_path: str | Path | None = None):
        self.cfg = cfg_thermal
        self.enabled = bool(cfg_thermal.get("enabled", True))
        self.h = None
        self.log_path = Path(log_path) if log_path else None
        self._last_log = 0.0
        self.pauses = 0
        self.pause_seconds = 0.0
        self.max_temp = 0
        if not self.enabled:
            return
        try:
            import os

            import pynvml

            pynvml.nvmlInit()
            self.nvml = pynvml
            # NVML ignores CUDA_VISIBLE_DEVICES; multi-GPU workers pass the
            # physical index via AZSENT_NVML_INDEX so each guard watches its own card.
            idx = int(os.environ.get("AZSENT_NVML_INDEX", "0"))
            self.h = pynvml.nvmlDeviceGetHandleByIndex(idx)
        except Exception:
            self.enabled = False

    def read(self) -> dict | None:
        if not self.enabled or self.h is None:
            return None
        try:
            t = self.nvml.nvmlDeviceGetTemperature(self.h, self.nvml.NVML_TEMPERATURE_GPU)
            p = self.nvml.nvmlDeviceGetPowerUsage(self.h) / 1000.0
            u = self.nvml.nvmlDeviceGetUtilizationRates(self.h).gpu
            m = self.nvml.nvmlDeviceGetMemoryInfo(self.h)
            self.max_temp = max(self.max_temp, t)
            row = {"t": time.time(), "temp_c": t, "power_w": round(p, 1), "util": u,
                   "vram_gb": round(m.used / 2**30, 2)}
            if self.log_path and time.time() - self._last_log >= float(self.cfg.get("log_every_s", 30)):
                self._last_log = time.time()
                new = not self.log_path.exists()
                self.log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(self.log_path, "a", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(row.keys()))
                    if new:
                        w.writeheader()
                    w.writerow(row)
            return row
        except Exception:
            return None

    def maybe_pause(self) -> None:
        """Call periodically from the training loop."""
        r = self.read()
        if r is None:
            return
        pause_at = int(self.cfg.get("pause_temp_c", 80))
        resume_at = int(self.cfg.get("resume_temp_c", 70))
        if r["temp_c"] >= pause_at:
            self.pauses += 1
            t0 = time.time()
            log.warning("Thermal pause: GPU %d degC >= %d. Cooling to %d ...", r["temp_c"], pause_at, resume_at)
            while True:
                time.sleep(10)
                r2 = self.read()
                if r2 is None or r2["temp_c"] <= resume_at:
                    break
            self.pause_seconds += time.time() - t0
            log.info("Thermal resume after %.0fs", time.time() - t0)

    def gate_job_start(self) -> None:
        """Called by the runner between jobs."""
        if not self.enabled:
            return
        target = int(self.cfg.get("job_gap_temp_c", 60))
        max_wait = float(self.cfg.get("job_gap_max_wait_s", 900))
        t0 = time.time()
        while time.time() - t0 < max_wait:
            r = self.read()
            if r is None or r["temp_c"] <= target:
                return
            time.sleep(15)

    def summary(self) -> dict:
        return {"thermal_pauses": self.pauses, "thermal_pause_seconds": round(self.pause_seconds, 1),
                "max_temp_c": self.max_temp}
