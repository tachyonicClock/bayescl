from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Literal, Protocol

from avalanche.training.plugins import SupervisedPlugin
from loguru import logger
from zeus.monitor.energy import ZeusMonitor

from bayescl.config import ZeusMonitorConfig


class ScalarWriter(Protocol):
    def add_scalar(self, name: str, value: float, step: int) -> None: ...


@dataclass
class ZeusMeasurementRecord:
    phase: Literal["train", "eval"]
    scope: Literal["experience", "stream"]
    window_name: str
    experience: int | None
    step: int
    time: float
    total_energy: float
    average_power: float | None
    gpu_energy: dict[int, float]
    total_gpu_energy: float
    cpu_energy: dict[int, float] | None
    total_cpu_energy: float | None
    dram_energy: dict[int, float] | None
    total_dram_energy: float | None


class ZeusMonitorPlugin(SupervisedPlugin):
    supports_distributed = False

    def __init__(
        self,
        config: ZeusMonitorConfig,
        writer: ScalarWriter,
    ) -> None:
        super().__init__()
        self.config = config
        self.writer = writer
        self._records: list[ZeusMeasurementRecord] = []
        self._eval_index = 0
        self._monitor_factory = ZeusMonitor
        self._monitor: Any | None = None

    def before_training_exp(self, strategy: Any, *args, **kwargs) -> None:
        self._begin_window(self._train_window_name(strategy))

    def after_training_exp(self, strategy: Any, *args, **kwargs) -> None:
        self._record_measurement(
            phase="train",
            scope="experience",
            window_name=self._train_window_name(strategy),
            strategy=strategy,
            experience=self._experience_index(strategy),
        )

    def before_eval(self, strategy: Any, *args, **kwargs) -> None:
        if not self.config.measure_eval:
            return
        self._begin_window(self._eval_stream_window_name())

    def before_eval_exp(self, strategy: Any, *args, **kwargs) -> None:
        if not self.config.measure_eval:
            return
        self._begin_window(self._eval_experience_window_name(strategy))

    def after_eval_exp(self, strategy: Any, *args, **kwargs) -> None:
        if not self.config.measure_eval:
            return
        self._record_measurement(
            phase="eval",
            scope="experience",
            window_name=self._eval_experience_window_name(strategy),
            strategy=strategy,
            experience=self._experience_index(strategy),
        )

    def after_eval(self, strategy: Any, *args, **kwargs) -> None:
        if not self.config.measure_eval:
            return
        self._record_measurement(
            phase="eval",
            scope="stream",
            window_name=self._eval_stream_window_name(),
            strategy=strategy,
            experience=None,
        )
        self._eval_index += 1

    def result(self) -> dict[str, Any]:
        records = [asdict(record) for record in self._records]
        return {
            "records": records,
            "summary": self._summarize(records),
        }

    def _monitor_instance(self) -> Any:
        if self._monitor is None:
            self._monitor = self._monitor_factory(
                gpu_indices=self.config.gpu_indices,
                cpu_indices=self.config.cpu_indices,
                approx_instant_energy=self.config.approx_instant_energy,
                sync_execution_with="torch",
            )
        return self._monitor

    def _begin_window(self, name: str) -> None:
        self._monitor_instance().begin_window(
            name,
            sync_execution=self.config.sync_execution,
            restart=True,
        )

    def _record_measurement(
        self,
        phase: Literal["train", "eval"],
        scope: Literal["experience", "stream"],
        window_name: str,
        strategy: Any,
        experience: int | None,
    ) -> None:
        measurement = self._monitor_instance().end_window(
            window_name,
            sync_execution=self.config.sync_execution,
        )
        gpu_energy = dict(getattr(measurement, "gpu_energy", {}) or {})
        cpu_energy = self._normalize_optional_energy(
            getattr(measurement, "cpu_energy", None)
        )
        dram_energy = self._normalize_optional_energy(
            getattr(measurement, "dram_energy", None)
        )
        record = ZeusMeasurementRecord(
            phase=phase,
            scope=scope,
            window_name=window_name,
            experience=experience,
            step=self._global_step(strategy),
            time=float(getattr(measurement, "time", 0.0)),
            total_energy=float(getattr(measurement, "total_energy", 0.0)),
            average_power=self._average_power(measurement),
            gpu_energy=gpu_energy,
            total_gpu_energy=sum(gpu_energy.values()),
            cpu_energy=cpu_energy,
            total_cpu_energy=self._sum_optional_energy(cpu_energy),
            dram_energy=dram_energy,
            total_dram_energy=self._sum_optional_energy(dram_energy),
        )
        self._records.append(record)
        self._log_record(record)

    def _log_record(self, record: ZeusMeasurementRecord) -> None:
        logger.info(
            "Zeus {} {} energy: {:.3f} J in {:.3f} s",
            record.phase,
            record.scope,
            record.total_energy,
            record.time,
        )
        base = f"zeus/{record.phase}/{record.scope}"
        self.writer.add_scalar(f"{base}/time_s", record.time, record.step)
        self.writer.add_scalar(f"{base}/energy_j", record.total_energy, record.step)
        if record.average_power is not None:
            self.writer.add_scalar(
                f"{base}/average_power_w", record.average_power, record.step
            )
        self.writer.add_scalar(
            f"{base}/gpu_energy_j_total", record.total_gpu_energy, record.step
        )
        if record.total_cpu_energy is not None:
            self.writer.add_scalar(
                f"{base}/cpu_energy_j_total", record.total_cpu_energy, record.step
            )
        if record.total_dram_energy is not None:
            self.writer.add_scalar(
                f"{base}/dram_energy_j_total", record.total_dram_energy, record.step
            )

    def _summarize(
        self, records: list[dict[str, Any]]
    ) -> dict[str, dict[str, float | int]]:
        summary: dict[str, dict[str, float | int]] = {}
        for phase in ("train", "eval"):
            for scope in ("experience", "stream"):
                subset = [
                    record
                    for record in records
                    if record["phase"] == phase and record["scope"] == scope
                ]
                if not subset:
                    continue
                summary[f"{phase}_{scope}"] = {
                    "count": len(subset),
                    "time": sum(float(record["time"]) for record in subset),
                    "total_energy": sum(
                        float(record["total_energy"]) for record in subset
                    ),
                    "total_gpu_energy": sum(
                        float(record["total_gpu_energy"]) for record in subset
                    ),
                }
        return summary

    def _global_step(self, strategy: Any) -> int:
        clock = getattr(strategy, "clock", None)
        return int(getattr(clock, "train_iterations", 0))

    def _experience_index(self, strategy: Any) -> int:
        experience = getattr(strategy, "experience", None)
        current = getattr(experience, "current_experience", None)
        if current is not None:
            return int(current)
        clock = getattr(strategy, "clock", None)
        return int(getattr(clock, "train_exp_counter", 0))

    def _train_window_name(self, strategy: Any) -> str:
        return f"train_exp_{self._experience_index(strategy)}"

    def _eval_stream_window_name(self) -> str:
        return f"eval_stream_{self._eval_index}"

    def _eval_experience_window_name(self, strategy: Any) -> str:
        return f"eval_stream_{self._eval_index}_exp_{self._experience_index(strategy)}"

    def _average_power(self, measurement: Any) -> float | None:
        time = float(getattr(measurement, "time", 0.0))
        if time <= 0:
            return None
        return float(getattr(measurement, "total_energy", 0.0)) / time

    def _normalize_optional_energy(
        self, energy: dict[int, float] | None
    ) -> dict[int, float] | None:
        if energy is None:
            return None
        return dict(energy)

    def _sum_optional_energy(self, energy: dict[int, float] | None) -> float | None:
        if energy is None:
            return None
        return sum(energy.values())
