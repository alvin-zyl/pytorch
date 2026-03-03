#!/usr/bin/env python3
"""
Failure Simulator for ULFM Fault-Tolerant Training

A minibatch-based stochastic failure simulator. Register possible failure
locations with decorators or context managers. At each minibatch, the simulator
decides whether to fail and randomly picks one registered location.

Usage:
    from failure_simulator import FailureSimulator, set_failure_simulator

    sim = FailureSimulator(
        seed=42,
        desired_failures=2,      # Expected total failures across all ranks
        total_minibatches=100,   # Total minibatches in training
        target_ranks={1, 2},     # Ranks that can fail
    )
    set_failure_simulator(sim)
    sim.initialize(rank=dist.get_rank(), world_size=dist.get_world_size())

    # Register locations with decorator (registers at decoration time)
    @sim.may_fail("forward_pass")
    def forward(x):
        return model(x)

    # Or context manager (registers on first use)

    # Training loop
    for minibatch in range(total_minibatches):
        sim.begin_minibatch(minibatch)  # Decides if/where to fail
        # Note: minibatch 0 is skipped to allow context managers to register

        output = forward(x)  # May fail here if selected

        with sim.may_fail_here("backward"):
            loss.backward()  # May fail here if selected
"""

import os
import signal
import random
import threading
import logging
import time
from contextlib import contextmanager
from functools import wraps
from typing import Optional, Set, List
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class SimulatorState:
    """Current state for decision making."""
    rank: int = -1
    world_size: int = -1
    minibatch: int = 0
    # Per-minibatch failure decision
    should_fail_this_minibatch: bool = False
    target_location: Optional[str] = None


@dataclass
class FailureRecord:
    """Record of an injected failure."""
    location: str
    rank: int
    minibatch: int
    timestamp: float = field(default_factory=time.time)


class FailureSimulator:
    """
    Stochastic failure simulator for ULFM testing.

    Failure injection is tied to minibatches:
    1. Register possible failure locations with decorators or context managers
    2. Call begin_minibatch() at the start of each minibatch
    3. The simulator decides if this minibatch should fail (minibatch 0 is skipped
       to allow context managers to register on first use)
    4. If failing, one registered location is randomly chosen to inject SIGKILL

    The probability is auto-computed from desired_failures, total_minibatches,
    and number of target ranks using: p = 1 - (1 - F/N)^(1/S)

    Args:
        seed: Random seed for reproducibility. Different seeds produce
              different failure sequences.
        desired_failures: Expected total number of failures across ALL ranks.
                         The probability is computed to achieve this on average.
        total_minibatches: Total number of minibatches in training.
        target_ranks: Set of ranks that can fail. None means any rank can fail.
        enabled: Whether injection is active. Set to False to disable.
    """

    @staticmethod
    def compute_probability(
        total_minibatches: int,
        desired_failures: int,
        num_target_ranks: int,
    ) -> float:
        """
        Compute per-minibatch failure probability to achieve desired total failures.

        Uses the formula: p = 1 - (1 - F/N)^(1/S)

        This ensures that across all target ranks and minibatches, the expected
        number of failures equals desired_failures.

        Args:
            total_minibatches: Total number of minibatches in training
            desired_failures: Target number of total failures across all ranks
            num_target_ranks: Number of ranks that can fail (len(target_ranks))

        Returns:
            Probability value between 0.0 and 1.0
        """
        if desired_failures <= 0:
            return 0.0
        if num_target_ranks <= 0:
            return 0.0
        if total_minibatches <= 0:
            return 0.0
        if desired_failures >= num_target_ranks:
            # Can't have more failures than target ranks (each dies once)
            # Set high probability to ensure all target ranks fail
            return 1.0 - (1e-9) ** (1.0 / total_minibatches)

        # p = 1 - (1 - F/N)^(1/S)
        survival_ratio = 1.0 - desired_failures / num_target_ranks
        prob = 1.0 - survival_ratio ** (1.0 / total_minibatches)
        return prob

    def __init__(
        self,
        seed: int = 42,
        desired_failures: int = 1,
        total_minibatches: int = 100,
        target_ranks: Optional[Set[int]] = None,
        enabled: bool = True,
    ):
        self.seed = seed
        self.desired_failures = desired_failures
        self.total_minibatches = total_minibatches
        self.target_ranks = target_ranks
        self.enabled = enabled

        # Computed in initialize() once we know world_size
        self._failure_probability: float = 0.0

        self._lock = threading.RLock()
        self._rng: Optional[random.Random] = None
        self._has_failed = False  # Each rank can only fail once
        self._history: List[FailureRecord] = []
        self._state = SimulatorState()
        self._initialized = False

        # Registered failure locations (populated by decorators/context managers)
        self._registered_locations: Set[str] = set()

    def initialize(self, rank: int, world_size: int) -> None:
        """
        Initialize with distributed context.

        Must be called after dist.init_process_group() or equivalent.
        Each rank gets a unique RNG seed (base_seed + rank) for independent
        random sequences across processes.

        Computes failure probability based on desired_failures, total_minibatches,
        and number of target ranks.

        Args:
            rank: This process's rank in the distributed group
            world_size: Total number of processes
        """
        with self._lock:
            self._state.rank = rank
            self._state.world_size = world_size

            # Compute number of target ranks
            if self.target_ranks is not None:
                num_target_ranks = len(self.target_ranks)
            else:
                num_target_ranks = world_size

            # Compute probability to achieve desired_failures
            self._failure_probability = self.compute_probability(
                total_minibatches=self.total_minibatches,
                desired_failures=self.desired_failures,
                num_target_ranks=num_target_ranks,
            )

            # Unique seed per rank for independent random sequences
            self._rng = random.Random(self.seed + rank)
            self._initialized = True

            logger.info(
                f"[Rank {rank}] FailureSimulator initialized: "
                f"seed={self.seed}, desired_failures={self.desired_failures}, "
                f"total_minibatches={self.total_minibatches}, "
                f"computed_probability={self._failure_probability:.6f}, "
                f"target_ranks={self.target_ranks}"
            )

    def register_location(self, location: str) -> None:
        """
        Register a location as a possible failure point.

        Locations are automatically registered when using may_fail() decorator
        or may_fail_here() context manager. This method allows manual registration.

        Args:
            location: Name of the failure point
        """
        with self._lock:
            self._registered_locations.add(location)

    def get_registered_locations(self) -> Set[str]:
        """Get all registered failure locations."""
        with self._lock:
            return self._registered_locations.copy()

    def begin_minibatch(self, minibatch: int) -> None:
        """
        Begin a new minibatch and decide if failure should occur.

        Call this at the start of each minibatch. The simulator will:
        1. Decide if this minibatch should have a failure (based on probability)
        2. If yes, randomly select one registered location as the target

        Note: Minibatch 0 is always skipped to allow location registration to
        complete (context managers register on first use).

        Args:
            minibatch: Current minibatch index (0-based)
        """
        with self._lock:
            self._state.minibatch = minibatch
            self._state.should_fail_this_minibatch = False
            self._state.target_location = None

            if not self.enabled:
                return
            if not self._initialized or self._rng is None:
                logger.warning("FailureSimulator not initialized, skipping minibatch decision")
                return
            if self._has_failed:
                return
            if self.target_ranks is not None and self._state.rank not in self.target_ranks:
                return
            if not self._registered_locations:
                logger.warning("No locations registered, cannot inject failure")
                return

            # Skip first minibatch to allow location registration to complete
            if minibatch == 0:
                return

            # Decide if this minibatch should fail
            if self._rng.random() < self._failure_probability:
                self._state.should_fail_this_minibatch = True
                # Randomly pick one registered location
                locations = list(self._registered_locations)
                self._state.target_location = self._rng.choice(locations)
                logger.debug(
                    f"[Rank {self._state.rank}] Minibatch {minibatch}: "
                    f"scheduled failure at '{self._state.target_location}'"
                )

    def _inject(self, location: str) -> None:
        """Execute SIGKILL to simulate process failure."""
        with self._lock:
            record = FailureRecord(
                location=location,
                rank=self._state.rank,
                minibatch=self._state.minibatch,
            )
            self._history.append(record)
            self._has_failed = True

            logger.warning(
                f"[Rank {self._state.rank}] INJECTING FAILURE at '{location}', "
                f"minibatch={self._state.minibatch}"
            )

        # SIGKILL for immediate process termination
        os.kill(os.getpid(), signal.SIGKILL)

    def check(self, location: str) -> bool:
        """
        Check if failure should be injected at this location.

        Only injects if:
        1. This minibatch was selected for failure (via begin_minibatch)
        2. This location was randomly chosen as the target

        Args:
            location: Name of the failure point

        Returns:
            True if failure was injected (process will die, so won't return)
            False if no failure was injected
        """
        with self._lock:
            if (self._state.should_fail_this_minibatch and
                self._state.target_location == location):
                self._inject(location)
                return True  # Won't reach here due to SIGKILL
            return False

    def may_fail(self, location: str):
        """
        Decorator to mark a function as a possible failure point.

        Registers the location and checks for failure at function entry.

        Usage:
            @simulator.may_fail("forward_pass")
            def forward(self, x):
                return self.model(x)

        Args:
            location: Name for this failure point
        """
        self.register_location(location)

        def decorator(func):
            @wraps(func)
            def wrapper(*args, **kwargs):
                self.check(location)
                return func(*args, **kwargs)
            return wrapper
        return decorator

    @contextmanager
    def may_fail_here(self, location: str):
        """
        Context manager to mark a code block as a possible failure point.

        Registers the location and checks for failure at context entry.

        Usage:
            with simulator.may_fail_here("backward"):
                loss.backward()

        Args:
            location: Name for this failure point
        """
        self.register_location(location)
        self.check(location)
        yield

    @property
    def failure_probability(self) -> float:
        """Computed failure probability per check point."""
        return self._failure_probability

    @property
    def has_failed(self) -> bool:
        """Whether this rank has already failed."""
        with self._lock:
            return self._has_failed

    def get_stats(self) -> dict:
        """
        Get failure statistics.

        Returns:
            Dictionary with configuration and history.
        """
        with self._lock:
            return {
                "desired_failures": self.desired_failures,
                "total_minibatches": self.total_minibatches,
                "failure_probability": self._failure_probability,
                "registered_locations": list(self._registered_locations),
                "has_failed": self._has_failed,
                "history": [
                    {
                        "location": r.location,
                        "rank": r.rank,
                        "minibatch": r.minibatch,
                        "timestamp": r.timestamp,
                    }
                    for r in self._history
                ],
            }

    def reset(self) -> None:
        """
        Reset simulator for a new run.

        Clears failure history and resets RNG. Useful for running
        multiple test scenarios with the same simulator instance.
        Note: registered locations are preserved.
        """
        with self._lock:
            self._has_failed = False
            self._history.clear()
            self._state.minibatch = 0
            self._state.should_fail_this_minibatch = False
            self._state.target_location = None
            if self._state.rank >= 0:
                self._rng = random.Random(self.seed + self._state.rank)
            logger.info(f"[Rank {self._state.rank}] FailureSimulator reset")

    def __repr__(self) -> str:
        return (
            f"FailureSimulator(seed={self.seed}, desired_failures={self.desired_failures}, "
            f"total_minibatches={self.total_minibatches}, probability={self._failure_probability:.6f}, "
            f"target_ranks={self.target_ranks}, locations={self._registered_locations}, "
            f"enabled={self.enabled})"
        )


# Global singleton for convenience
_simulator: Optional[FailureSimulator] = None


def get_failure_simulator() -> Optional[FailureSimulator]:
    """Get the global failure simulator instance."""
    return _simulator


def set_failure_simulator(sim: FailureSimulator) -> None:
    """Set the global failure simulator instance."""
    global _simulator
    _simulator = sim


def clear_failure_simulator() -> None:
    """Clear the global failure simulator instance."""
    global _simulator
    _simulator = None
