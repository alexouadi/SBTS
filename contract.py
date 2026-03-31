from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Mapping, Optional
import pickle

import numpy as np

from models.sbts_multi import simulate_kernel_vectorized
from models.sbts_multi_markovian import (
    sample_last_mark_multi,
    simulate_kernel_vectorized_mark,
)
from models.sbts_uni import simulate_kernel
from models.sbts_uni_markovian import sample_last_mark, simulate_kernel_mark


class GenerationMode(str, Enum):
    UNCONDITIONAL = "unconditional"
    FORECAST = "forecast"


@dataclass
class GenerationResult:
    samples: Any
    diagnostics: Optional[Mapping[str, Any]] = None


@dataclass
class ModelCapabilities:
    supported_modes: frozenset[GenerationMode]
    supports_multivariate_targets: bool = True
    supports_known_covariates: bool = False
    supports_observed_covariates: bool = False
    supports_static_covariates: bool = False
    supports_constraints: bool = False


@dataclass
class FitReport:
    train_metrics: Optional[Mapping[str, float]] = None
    val_metrics: Optional[Mapping[str, float]] = None
    fit_time_sec: Optional[float] = None
    peak_memory_mb: Optional[float] = None
    n_parameters: Optional[int] = None
    diagnostics: Optional[Mapping[str, Any]] = None


def build_estimator(
    *,
    h: float = 0.2,
    N_pi: int = 100,
    deltati: float = 1.0,
    markov_order: int = 1,
    unconditional_variant: str = "standard",
    random_seed: int | None = None,
) -> "SBTSEstimator":
    """Build a duck-typed SBTS estimator."""
    return SBTSEstimator(
        h=h,
        N_pi=N_pi,
        deltati=deltati,
        markov_order=markov_order,
        unconditional_variant=unconditional_variant,
        random_seed=random_seed,
    )


def load_generator(path: str | Path) -> "SBTSGenerator":
    """Load a previously saved fitted SBTS generator."""
    with Path(path).open("rb") as handle:
        payload = pickle.load(handle)
    return SBTSGenerator.from_state(payload)


@dataclass
class SBTSGenerator:
    """Fitted SBTS generator returned by the thin estimator."""

    h: float
    N_pi: int
    deltati: float
    markov_order: int
    unconditional_variant: str
    random_seed: Optional[int]
    fitted_mode: GenerationMode
    target_dim: int
    horizon: int
    context_length: int
    reference_bank: np.ndarray

    @staticmethod
    def _mode_name(value: object) -> str:
        if hasattr(value, "value"):
            return str(getattr(value, "value"))
        return str(value)

    def capabilities(self) -> ModelCapabilities:
        return ModelCapabilities(
            supported_modes=frozenset({self.fitted_mode}),
            supports_multivariate_targets=True,
            supports_known_covariates=False,
            supports_observed_covariates=False,
            supports_static_covariates=False,
            supports_constraints=False,
        )

    def sample(self, request: Any) -> GenerationResult:
        request_mode = self._mode_name(getattr(request.task, "mode", None))
        if request_mode != self.fitted_mode.value:
            raise ValueError(
                f"SBTS generator was fit for mode '{self.fitted_mode.value}' "
                f"but received generation request for mode '{request_mode}'."
            )
        request_horizon = getattr(request.task, "horizon", None)
        if request_horizon is not None and int(request_horizon) != self.horizon:
            raise ValueError(
                f"SBTS generator was fit for horizon {self.horizon} but received "
                f"generation request for horizon {int(request_horizon)}."
            )
        series_values = np.asarray(request.series.values, dtype=np.float64)
        num_samples = int(request.num_samples)

        if self.fitted_mode == GenerationMode.UNCONDITIONAL:
            samples = self._sample_unconditional(num_samples=num_samples)
        else:
            if series_values.shape[0] != self.context_length:
                raise ValueError(
                    f"SBTS forecast generator expects context length {self.context_length}, "
                    f"received {series_values.shape[0]}."
                )
            samples = self._sample_forecast(context=series_values, num_samples=num_samples)

        return GenerationResult(samples=samples)

    def _sample_unconditional(self, *, num_samples: int) -> np.ndarray:
        if self.target_dim == 1:
            bank = self.reference_bank
            assert bank.ndim == 2
            bank_size = int(bank.shape[0])
            generated = np.zeros((num_samples, self.horizon), dtype=np.float64)
            for idx in range(num_samples):
                if self.unconditional_variant == "markovian":
                    path = simulate_kernel_mark(
                        self.horizon,
                        bank_size,
                        self.markov_order,
                        bank,
                        self.N_pi,
                        self.h,
                        self.deltati,
                    )
                else:
                    path = simulate_kernel(
                        self.horizon,
                        bank_size,
                        bank,
                        self.N_pi,
                        self.h,
                        self.deltati,
                    )
                generated[idx] = path[1:]
            samples = generated[:, :, np.newaxis]
        else:
            bank = self.reference_bank
            assert bank.ndim == 3
            bank_size = int(bank.shape[0])
            generated = np.zeros((num_samples, self.horizon, self.target_dim), dtype=np.float64)
            for idx in range(num_samples):
                if self.unconditional_variant == "markovian":
                    path = simulate_kernel_vectorized_mark(
                        self.horizon,
                        bank_size,
                        self.target_dim,
                        self.markov_order,
                        bank,
                        self.N_pi,
                        self.h,
                        self.deltati,
                    )
                else:
                    path = simulate_kernel_vectorized(
                        self.horizon,
                        bank_size,
                        self.target_dim,
                        bank,
                        self.N_pi,
                        self.h,
                        self.deltati,
                    )
                generated[idx] = path[1:]
            samples = generated

        return samples

    def _sample_forecast(self, *, context: np.ndarray, num_samples: int) -> np.ndarray:
        samples = np.zeros((num_samples, self.horizon, self.target_dim), dtype=np.float64)

        if self.target_dim == 1:
            bank = self.reference_bank
            assert bank.ndim == 2
            bank_size = int(bank.shape[0])
            history_seed = context[:, 0]
            for sample_idx in range(num_samples):
                history = np.zeros((self.context_length + self.horizon,), dtype=np.float64)
                history[: self.context_length] = history_seed
                for step in range(self.horizon):
                    curr_length = self.context_length + step
                    history[curr_length] = sample_last_mark(
                        bank_size,
                        self.markov_order,
                        bank,
                        history[:curr_length],
                        self.N_pi,
                        self.h,
                        self.deltati,
                    )
                samples[sample_idx, :, 0] = history[self.context_length :]
        else:
            bank = self.reference_bank
            assert bank.ndim == 3
            bank_size = int(bank.shape[0])
            history_seed = context
            for sample_idx in range(num_samples):
                history = np.zeros((self.context_length + self.horizon, self.target_dim), dtype=np.float64)
                history[: self.context_length] = history_seed
                for step in range(self.horizon):
                    curr_length = self.context_length + step
                    history[curr_length] = sample_last_mark_multi(
                        bank_size,
                        self.target_dim,
                        self.markov_order,
                        bank,
                        history[:curr_length],
                        self.N_pi,
                        self.h,
                        self.deltati,
                    )
                samples[sample_idx] = history[self.context_length :]

        return samples

    def save(self, path: str | Path) -> None:
        payload = self.state_dict()
        with Path(path).open("wb") as handle:
            pickle.dump(payload, handle)

    def state_dict(self) -> dict[str, Any]:
        return {
            "h": self.h,
            "N_pi": self.N_pi,
            "deltati": self.deltati,
            "markov_order": self.markov_order,
            "unconditional_variant": self.unconditional_variant,
            "random_seed": self.random_seed,
            "fitted_mode": self.fitted_mode.value,
            "target_dim": self.target_dim,
            "horizon": self.horizon,
            "context_length": self.context_length,
            "reference_bank": self.reference_bank,
        }

    @classmethod
    def from_state(cls, payload: Mapping[str, Any]) -> "SBTSGenerator":
        return cls(
            h=float(payload["h"]),
            N_pi=int(payload["N_pi"]),
            deltati=float(payload["deltati"]),
            markov_order=int(payload["markov_order"]),
            unconditional_variant=str(payload["unconditional_variant"]),
            random_seed=None if payload.get("random_seed") is None else int(payload["random_seed"]),
            fitted_mode=GenerationMode(str(payload["fitted_mode"])),
            target_dim=int(payload["target_dim"]),
            horizon=int(payload["horizon"]),
            context_length=int(payload["context_length"]),
            reference_bank=np.asarray(payload["reference_bank"], dtype=np.float64),
        )


class SBTSEstimator:
    """Duck-typed trainable estimator for SBTS."""

    def __init__(
        self,
        *,
        h: float,
        N_pi: int,
        deltati: float,
        markov_order: int,
        unconditional_variant: str,
        random_seed: Optional[int],
    ):
        self.h = h
        self.N_pi = N_pi
        self.deltati = deltati
        self.markov_order = markov_order
        self.unconditional_variant = unconditional_variant
        self.random_seed = random_seed

    def fit(
        self,
        train: Any,
        *,
        schema: Any,
        task: Any,
        valid: Optional[Any] = None,
        runtime: Optional[Any] = None,
    ) -> tuple[SBTSGenerator, FitReport]:
        mode = GenerationMode(str(getattr(task.mode, "value", task.mode)))
        horizon = int(task.horizon)
        examples = train.examples

        if mode == GenerationMode.UNCONDITIONAL:
            reference_paths = np.stack(
                [np.asarray(example.target.values, dtype=np.float64) for example in examples],
                axis=0,
            )
            fitted_context_length = 0
        else:
            fitted_context_length = int(np.asarray(examples[0].context.values).shape[0])
            reference_path_length = fitted_context_length + horizon
            reference_sequences: list[np.ndarray] = []
            for example in examples:
                full_path = np.concatenate(
                    [
                        np.asarray(
                            example.history.values if example.history is not None else example.context.values,
                            dtype=np.float64,
                        ),
                        np.asarray(example.target.values, dtype=np.float64),
                    ],
                    axis=0,
                )
                if full_path.shape[0] < reference_path_length:
                    raise ValueError(
                        "SBTS forecast examples must provide enough realized history to form at least "
                        f"one fixed-length reference sequence of length {reference_path_length}; "
                        f"received length {full_path.shape[0]}."
                    )
                for start in range(full_path.shape[0] - reference_path_length + 1):
                    reference_sequences.append(full_path[start : start + reference_path_length])
            reference_paths = np.stack(reference_sequences, axis=0)

        if int(schema.target_dim) == 1:
            reference_bank: np.ndarray = reference_paths[:, :, 0]
        else:
            reference_bank = reference_paths

        fitted = SBTSGenerator(
            h=self.h,
            N_pi=self.N_pi,
            deltati=self.deltati,
            markov_order=self.markov_order,
            unconditional_variant=self.unconditional_variant,
            random_seed=self.random_seed,
            fitted_mode=mode,
            target_dim=int(schema.target_dim),
            horizon=horizon,
            context_length=fitted_context_length,
            reference_bank=reference_bank,
        )
        return (
            fitted,
            FitReport(
                diagnostics={
                    "reference_bank_shape": tuple(int(dim) for dim in reference_bank.shape),
                    "fitted_mode": fitted.fitted_mode.value,
                    "context_length": fitted.context_length,
                    "horizon": fitted.horizon,
                }
            ),
        )


__all__ = [
    "SBTSEstimator",
    "SBTSGenerator",
    "build_estimator",
    "load_generator",
]
