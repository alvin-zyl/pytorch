#!/usr/bin/env python3
"""
ULFM Communication Hook for PyTorch DDP

This module provides a reusable communication hook for fault-tolerant distributed training
with ULFM (User Level Failure Mitigation). The hook integrates with PyTorch DDP and the
StepTxnOrchestrator for gradient snapshotting and recovery.
"""

import os
import signal
import logging
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from policy import FaultTolerancePolicy

try:
    import ulfm_collectives as ULFM
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )

from orchestrator import StepTxnOrchestrator

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """State object passed to the ULFM communication hook."""

    pg: Union[ULFM.ProcessGroupULFM, dist.ProcessGroup]
    orchestrator: StepTxnOrchestrator


def create_ulfm_recovery_hook(failure_strategy: str = "continue"):
    """
    Create a ULFM communication hook with comprehensive failure recovery logic.

    Now policy-driven: consults the FaultTolerancePolicy for all fault tolerance decisions.
    The policy is accessed from the orchestrator (via hstate.orchestrator.policy).

    This hook integrates with the StepTxnOrchestrator to:
    - Snapshot gradients before reduction
    - Detect and handle process failures
    - Consult policy for quiesce/repair decisions
    - Track successful reductions for restoration decisions

    Args:
        failure_strategy: "continue", "restart", or "abort" (legacy, kept for ULFM options)

    Returns:
        Callable hook function compatible with DDP.register_comm_hook()
        The hook expects state to be a HookState object with pg and orchestrator.
        The orchestrator must have a policy attribute.

    Example:
        >>> from ulfm_hook import create_ulfm_recovery_hook, HookState
        >>> from policy import create_policy
        >>> pg = dist.group.WORLD
        >>> rank = dist.get_rank()
        >>> policy = create_policy("adaptive", ...)
        >>> orchestrator = StepTxnOrchestrator(rank=rank, policy=policy)
        >>> hook = create_ulfm_recovery_hook("continue")
        >>> hstate = HookState(pg=pg, orchestrator=orchestrator)
        >>> ddp_model.register_comm_hook(state=hstate, hook=hook)
    """
    strategy_map = {
        "continue": ULFM.ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS,
        "restart": ULFM.ULFMFailureHandlingStrategy.RESTART_FAILED_PROCESSES,
        "abort": ULFM.ULFMFailureHandlingStrategy.ABORT_ON_FAILURE,
    }

    if failure_strategy not in strategy_map:
        raise ValueError(f"Invalid failure strategy: {failure_strategy}")

    def hook(hstate: HookState, bucket: dist.GradBucket):
        """ULFM communication hook with comprehensive recovery logic."""
        pg = hstate.pg
        orch = hstate.orchestrator
        policy = orch.policy  # Get policy from orchestrator

        # print(f"ULFM Hook invoked on rank {orch.rank}, current hook counter {orch.get_hook_counter()}, current macrobatch idx {orch._current_macrobatch_idx}")
        # if orch.rank == 1 and orch.get_hook_counter() > 0 and orch._current_macrobatch_idx == 2:
        #     logger.warning(f"[Rank {orch.rank}] Simulating process failure in mid of grad sync")
        #     os.kill(os.getpid(), signal.SIGKILL)

        # 1) If a previous failure quiesced comms, NOOP this bucket
        if getattr(pg, "is_quiesced", lambda: False)():
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        # 2) Snapshot the entire bucket buffer (pre-reduce)
        bucket_index = bucket.index()
        orch.on_bucket_snapshot(bucket.buffer(), bucket_index, pg)

        # 3) Normal ULFM allreduce (SUM; scale once at commit)
        opts = torch.distributed.AllreduceOptions()
        opts.reduceOp = torch.distributed.ReduceOp.SUM

        # Get ULFM options from policy
        ulfm_opts = ULFM.ULFMOptions()
        ulfm_opts.auto_repair = policy.enable_auto_repair
        ulfm_opts.failure_strategy = strategy_map[failure_strategy]
        ulfm_opts.max_retries = 3
        ulfm_opts.retry_delay_ms = 100

        # work = dist.ulfm_all_reduce(bucket.buffer(), async_op=True, ulfm_opts=ulfm_opts)
        work = pg.ulfm_allreduce([bucket.buffer()], opts, ulfm_opts)

        def on_done(fut):
            if (
                orch._rank == 1
                and orch.get_hook_counter() > 0
                and orch._current_macrobatch_idx == 2
            ):
                logger.warning(
                    f"[Rank {orch._rank}] Simulating process failure in mid of grad sync, hook counter {orch.get_hook_counter()}"
                )
                os.kill(os.getpid(), signal.SIGKILL)

            # Use orchestrator's unified entry point for handling work completion
            # This encapsulates all failure detection, policy consultation, and recovery logic
            orch.handle_work_completion(
                work=work,
                bucket_index=bucket_index,
            )

            # Increment hook invocation counter
            orch.increment_hook_counter()

            return fut.value()[0]

        return work.get_future().then(on_done)

    return hook
