#!/usr/bin/env python3
"""
Fault Tolerance Policy Manager

This module encapsulates all fault tolerance decision-making logic, making the
training manager agnostic to specific fault tolerance strategies.

The Policy determines:
1. When failures require quiescing vs. just communicator repair
2. How to adjust gradient accumulation windows based on failure patterns
3. Whether gradient restoration should be blocking or non-blocking
4. Whether a step can be committed or needs additional forward/backward passes
"""

from dataclasses import dataclass
from typing import Optional, List
from enum import Enum
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from orchestrator import StepTxnOrchestrator


logger = logging.getLogger(__name__)


class GradRestoreMode(Enum):
    """How gradients should be restored after failure."""

    BLOCKING = "blocking"  # Synchronous restoration with re-reduction
    NON_BLOCKING = "non_blocking"  # Asynchronous restoration overlapped with forward
    SKIP = "skip"  # No restoration needed


class FailureResponse(Enum):
    """How to respond to a detected failure."""

    QUIESCE_AND_RESTORE = (
        "quiesce_and_restore"  # Stop all comms, restore grads next iter
    )
    REPAIR_AND_CONTINUE = "repair_and_continue"  # Just repair comm, keep going
    ABORT = "abort"  # Unrecoverable failure


@dataclass
class PolicyDecision:
    """
    Decision output from the policy about how to handle current state.
    """

    # Failure handling
    failure_response: FailureResponse
    should_quiesce: bool
    should_manual_repair: bool

    # Gradient management
    grad_restore_mode: GradRestoreMode
    should_skip_step: bool  # Skip optimizer.step() this iteration

    # Gradient accumulation adjustment
    grad_accum_steps: int  # Current window size (may change after failures)
    need_extra_microbatch: bool  # Add one more forward/backward to window

    # State tracking
    at_iteration_boundary: bool  # Are we at the start/end of accumulation window
    hook_invocation_count: int  # How many buckets reduced before failure


@dataclass
class FailureEvent:
    """Information about a detected failure."""

    failed_ranks: List[int]
    hook_invocation_count: int  # How many hook calls before this failure
    current_microbatch_idx: int  # Which microbatch in accumulation window
    total_microbatches: int  # Total microbatches in window
    world_epoch: int  # ProcessGroup epoch (increments on repair)


class FaultTolerancePolicy:
    """
    Base class for fault tolerance policies.

    Subclass this to implement different fault tolerance strategies.
    """

    def __init__(
        self, initial_grad_accum_steps: int = 1, enable_auto_repair: bool = True
    ):
        """
        Args:
            initial_grad_accum_steps: Initial gradient accumulation window size
            enable_auto_repair: Whether to enable automatic communicator repair
        """
        self.grad_accum_steps = initial_grad_accum_steps
        self.enable_auto_repair = enable_auto_repair

        # State tracking
        self._current_microbatch = 0
        self._failures_this_window = 0
        self._total_failures = 0
        self._total_recoveries = 0

    def on_window_start(self):
        """
        Called at the start of each accumulation window.
        """
        self._current_microbatch = 0
        self._failures_this_window = 0

    def on_failure(
        self, failure_event: FailureEvent, orchestrator: "StepTxnOrchestrator"
    ) -> PolicyDecision:
        """
        Decide how to handle a failure event.

        Args:
            failure_event: Information about the failure
            orchestrator: StepTxnOrchestrator for state queries and actions

        Returns:
            PolicyDecision with instructions for training manager
        """
        raise NotImplementedError("Subclasses must implement on_failure()")

    def on_microbatch_complete(self, microbatch_idx: int) -> PolicyDecision:
        """
        Called after each microbatch completes successfully.

        Args:
            microbatch_idx: Index of completed microbatch in window

        Returns:
            PolicyDecision about whether to commit or continue accumulating
        """
        self._current_microbatch = microbatch_idx

        at_boundary = (microbatch_idx + 1) >= self.grad_accum_steps

        return PolicyDecision(
            failure_response=FailureResponse.REPAIR_AND_CONTINUE,
            should_quiesce=False,
            should_manual_repair=False,
            grad_restore_mode=GradRestoreMode.SKIP,
            should_skip_step=not at_boundary,
            grad_accum_steps=self.grad_accum_steps,
            need_extra_microbatch=False,
            at_iteration_boundary=at_boundary,
            hook_invocation_count=0,  # No failure
        )

    def should_restore_gradients(
        self, orchestrator: "StepTxnOrchestrator"
    ) -> GradRestoreMode:
        """
        Determine if and how gradients should be restored.

        Args:
            orchestrator: StepTxnOrchestrator for state queries

        Returns:
            GradRestoreMode indicating restoration strategy
        """
        return orchestrator.get_restore_plan()

    def get_stats(self):
        """Get policy statistics."""
        return {
            "total_failures": self._total_failures,
            "total_recoveries": self._total_recoveries,
            "success_rate": self._total_recoveries / max(1, self._total_failures),
            "current_grad_accum_steps": self.grad_accum_steps,
        }


class AdaptiveWorldSizePolicy(FaultTolerancePolicy):
    """
    Adaptive World-Size Policy: Simple repair and continue, agnostic to world size changes.

    This policy:
    - Accepts any number of process failures
    - Simply repairs communicator and continues training
    - Adjusts gradients for new world size automatically
    - No gradient restoration needed (always uses current surviving processes)
    - Suitable for elastic training where world size can vary

    Strategy:
    - On failure: Repair communicator, continue with survivors
    - No quiescing or gradient restoration
    - Training adapts to whatever world size remains
    """

    def __init__(
        self, initial_grad_accum_steps: int = 1, enable_auto_repair: bool = True
    ):
        super().__init__(initial_grad_accum_steps, enable_auto_repair)

    def on_failure(
        self, failure_event: FailureEvent, orchestrator: "StepTxnOrchestrator"
    ) -> PolicyDecision:
        """
        Handle failure with simple repair and continue strategy.

        No matter when or how many failures occur, just repair and keep going.
        """
        self._total_failures += 1
        self._failures_this_window += 1

        hook_count = failure_event.hook_invocation_count
        num_failed = len(failure_event.failed_ranks)

        # Policy decision logging (INFO level - detailed trace)
        logger.info(
            f"[AdaptiveWorldSize] Failure detected: {num_failed} processes failed, "
            f"hook_count={hook_count}, repairing and continuing, "
            f"gradient restoration needed: {hook_count > 0}"
        )

        # Always just repair and continue - no quiesce, no restoration
        decision = PolicyDecision(
            failure_response=FailureResponse.REPAIR_AND_CONTINUE,
            should_quiesce=False,
            should_manual_repair=not self.enable_auto_repair,
            grad_restore_mode=(
                GradRestoreMode.BLOCKING if hook_count > 0 else GradRestoreMode.SKIP
            ),  # No restoration needed
            should_skip_step=False,  # Continue with current step
            grad_accum_steps=self.grad_accum_steps,
            need_extra_microbatch=False,
            at_iteration_boundary=False,  # Not used in this policy
            hook_invocation_count=hook_count,
        )

        self._total_recoveries += 1
        return decision


class FixedWorldSizePolicy(FaultTolerancePolicy):
    """
    Fixed World-Size Policy: Complex policy that tries to maintain a target world size.

    This policy has configurable "boundaries" that define acceptable operating conditions:
    - min_world_size: Minimum acceptable number of processes
    - max_failures_per_window: Maximum failures allowed before taking corrective action
    - target_world_size: Desired world size to maintain

    When within boundaries:
    - Gradients may be corrupted, need restoration
    - Quiesce and restore to maintain consistency

    When crossing boundaries:
    - Policy may need to adjust grad_accum_steps
    - May need to trigger more aggressive recovery (e.g., restart failed processes)
    - Different restoration strategies based on severity

    Strategy based on hook_invocation_count:
    - hook_count == 0: Failure during forward/backward, no grad corruption
    - hook_count > 0: Failure during grad sync, gradients corrupted, need restoration
    """

    def __init__(
        self,
        initial_grad_accum_steps: int = 1,
        enable_auto_repair: bool = True,
        target_world_size: Optional[int] = None,
        min_world_size: Optional[int] = None,
        max_failures_per_window: int = 1,
        adaptive_grad_accum: bool = True,
    ):
        """
        Args:
            target_world_size: Target world size to maintain (None = use initial world size)
            min_world_size: Minimum acceptable world size (None = no minimum)
            max_failures_per_window: Max failures per window before boundary crossed
            adaptive_grad_accum: Adjust grad accumulation based on world size changes
        """
        super().__init__(initial_grad_accum_steps, enable_auto_repair)

        self.target_world_size = target_world_size
        self.min_world_size = min_world_size
        self.max_failures_per_window = max_failures_per_window
        self.adaptive_grad_accum = adaptive_grad_accum

        self._initial_grad_accum_steps = initial_grad_accum_steps
        self._current_world_size = target_world_size  # Will be updated on first failure

    def on_failure(
        self, failure_event: FailureEvent, orchestrator: "StepTxnOrchestrator"
    ) -> PolicyDecision:
        """
        Handle failure with boundary-aware logic.

        Decision tree:
        1. Check if hook_invocation_count == 0 (failure during forward/backward)
           -> Simple repair, no restoration needed
        2. Check if crossing policy boundaries (world size, failure budget)
           -> If yes: More aggressive action (quiesce, restore, maybe adjust config)
           -> If no: Standard quiesce and restore
        """
        self._total_failures += 1
        self._failures_this_window += 1

        hook_count = failure_event.hook_invocation_count
        num_failed = len(failure_event.failed_ranks)

        # TODO: Get current world size from process group
        # For now, estimate: previous_world_size - num_failed
        # self._current_world_size = get_current_world_size(orchestrator)

        # Check if we're at a boundary condition
        at_boundary = self._is_at_boundary(failure_event, orchestrator)

        # Case 1: Failure during forward/backward (no gradient corruption)
        if hook_count == 0:
            logger.info(
                f"[FixedWorldSize] Failure during forward/backward (hook_count=0), "
                f"no gradient corruption, simple repair"
            )

            decision = PolicyDecision(
                failure_response=FailureResponse.REPAIR_AND_CONTINUE,
                should_quiesce=False,
                should_manual_repair=not self.enable_auto_repair,
                grad_restore_mode=GradRestoreMode.SKIP,
                should_skip_step=False,
                grad_accum_steps=self.grad_accum_steps,
                need_extra_microbatch=False,
                at_iteration_boundary=at_boundary,
                hook_invocation_count=hook_count,
            )
            self._total_recoveries += 1
            return decision

        # Case 2: At boundary - need special handling
        if at_boundary:
            logger.warning(
                f"[FixedWorldSize] BOUNDARY CROSSED: {num_failed} failures, "
                f"hook_count={hook_count}, world_size={self._current_world_size}, "
                f"failures_this_window={self._failures_this_window}"
            )

            # At boundary: Quiesce and restore, potentially adjust policy
            decision = self._handle_boundary_crossing(failure_event, orchestrator)

        # Case 3: Within boundaries - standard recovery
        else:
            logger.info(
                f"[FixedWorldSize] Mid-window failure (hook_count={hook_count}), "
                f"within boundaries, quiesce and restore"
            )

            decision = PolicyDecision(
                failure_response=FailureResponse.QUIESCE_AND_RESTORE,
                should_quiesce=True,
                should_manual_repair=not self.enable_auto_repair,
                grad_restore_mode=GradRestoreMode.NON_BLOCKING,
                should_skip_step=True,
                grad_accum_steps=self.grad_accum_steps,
                need_extra_microbatch=True,
                at_iteration_boundary=False,
                hook_invocation_count=hook_count,
            )

        self._total_recoveries += 1
        return decision

    def _is_at_boundary(
        self, failure_event: FailureEvent, orchestrator: "StepTxnOrchestrator"
    ) -> bool:
        """
        Determine if this failure crosses a policy boundary.

        Boundaries can be:
        - World size falls below min_world_size
        - Too many failures in current window (> max_failures_per_window)
        - TODO: Other policy-specific boundaries
        """
        # Boundary 1: Too many failures in this window
        if self._failures_this_window > self.max_failures_per_window:
            logger.warning(
                f"[FixedWorldSize] Boundary: failures_this_window ({self._failures_this_window}) "
                f"> max_failures_per_window ({self.max_failures_per_window})"
            )
            return True

        # Boundary 2: World size too small (if configured)
        if self.min_world_size is not None:
            # TODO: Get actual current world size from process group
            # estimated_world_size = self._current_world_size - len(failure_event.failed_ranks)
            # if estimated_world_size < self.min_world_size:
            #     return True
            pass

        return False

    def _handle_boundary_crossing(
        self, failure_event: FailureEvent, orchestrator: "StepTxnOrchestrator"
    ) -> PolicyDecision:
        """
        Handle a boundary crossing event.

        At boundary, we might need to:
        - Adjust gradient accumulation steps to compensate for world size change
        - Use blocking restoration for correctness
        - Reset failure counters
        - TODO: Potentially trigger process restart/replacement
        """
        logger.info(
            f"[FixedWorldSize] Handling boundary crossing, "
            f"may adjust grad_accum_steps"
        )

        # Adjust grad accumulation if enabled
        new_grad_accum = self.grad_accum_steps
        if self.adaptive_grad_accum:
            new_grad_accum = self._compute_adjusted_grad_accum(failure_event)
            if new_grad_accum != self.grad_accum_steps:
                logger.info(
                    f"[FixedWorldSize] Adjusting grad_accum_steps: "
                    f"{self.grad_accum_steps} -> {new_grad_accum}"
                )
                self.grad_accum_steps = new_grad_accum

        return PolicyDecision(
            failure_response=FailureResponse.QUIESCE_AND_RESTORE,
            should_quiesce=True,
            should_manual_repair=not self.enable_auto_repair,
            grad_restore_mode=GradRestoreMode.BLOCKING,  # Use blocking at boundary for safety
            should_skip_step=True,
            grad_accum_steps=new_grad_accum,
            need_extra_microbatch=True,
            at_iteration_boundary=True,  # Signal this is a boundary event
            hook_invocation_count=failure_event.hook_invocation_count,
        )

    def _compute_adjusted_grad_accum(self, failure_event: FailureEvent) -> int:
        """
        Compute adjusted gradient accumulation steps based on world size change.

        Goal: Maintain similar effective batch size after world size changes.

        TODO: Implement adaptive logic based on:
        - Current vs target world size
        - Available memory
        - Training stability requirements
        """
        # Placeholder: keep same for now
        return self.grad_accum_steps


# Factory function for easy policy creation
def create_policy(policy_type: str = "adaptive", **kwargs) -> FaultTolerancePolicy:
    """
    Create a fault tolerance policy.

    Args:
        policy_type: "adaptive" (AdaptiveWorldSizePolicy) or "fixed" (FixedWorldSizePolicy)
        **kwargs: Policy-specific arguments

    Returns:
        FaultTolerancePolicy instance
    """
    policies = {
        "adaptive": AdaptiveWorldSizePolicy,
        "fixed": FixedWorldSizePolicy,
    }

    if policy_type not in policies:
        raise ValueError(
            f"Unknown policy type: {policy_type}. Choose from {list(policies.keys())}"
        )

    return policies[policy_type](**kwargs)
