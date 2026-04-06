#!/usr/bin/env python3
"""
ULFM Communication Hook for PyTorch DDP

This module provides a reusable communication hook for fault-tolerant distributed training
with ULFM (User Level Failure Mitigation). The hook integrates with PyTorch DDP and the
StepTxnOrchestrator for gradient snapshotting and recovery.
"""

import contextlib
import os
import signal
import logging
import torch
import torch.distributed as dist
from dataclasses import dataclass
from typing import Union, TYPE_CHECKING

if TYPE_CHECKING:
    from .policy import FaultTolerancePolicy

try:
    import ulfm_collectives as ULFM
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )

from .orchestrator import StepTxnOrchestrator
from .failure_simulator import get_failure_simulator

logger = logging.getLogger(__name__)


@dataclass
class HookState:
    """State object passed to the ULFM communication hook."""

    pg: Union[ULFM.ProcessGroupULFM, dist.ProcessGroup]
    orchestrator: StepTxnOrchestrator


def create_ulfm_recovery_hook(ulfm_opts: ULFM.ULFMOptions = None):
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

    """ULFMOptions setup with AllreduceOp SUM for gradient reduction."""
    opts = torch.distributed.AllreduceOptions()
    opts.reduceOp = torch.distributed.ReduceOp.SUM
    ulfm_opts = ulfm_opts if ulfm_opts is not None else ULFM.ULFMOptions()

    def hook(hstate: HookState, bucket: dist.GradBucket):
        """ULFM communication hook with comprehensive recovery logic."""
        pg = hstate.pg
        orch = hstate.orchestrator
        bucket_index = bucket.index()

        logger.debug(
            f"[Rank {orch._rank}] Hook entered for bucket {bucket_index}, "
            f"numel={bucket.buffer().numel()}, dtype={bucket.buffer().dtype}"
        )

        # 1) If a previous failure quiesced comms, NOOP this bucket
        if getattr(pg, "is_quiesced", lambda: False)():
            logger.warning(
                f"[Rank {orch._rank}] Communicator is quiesced before submitting MPI request."
            )
            fut = torch.futures.Future()
            fut.set_result(bucket.buffer())
            return fut

        # 2) Ensure all GPU work (TP allreduces from pipeline fwd/bwd) has
        #    truly completed before we snapshot or enter MPI.  Without this,
        #    the CPU can race ahead of a stuck GPU stream: if a TP partner is
        #    dead the NCCL ops on this rank's stream never finish, but the CPU
        #    would still enter MPI — dragging the healthy DP partner into a
        #    blocked collective.  With the sync, a stuck rank blocks HERE
        #    (never enters MPI) and eventually dies via NCCL watchdog, letting
        #    the DP partner discover the failure through ULFM comm_agree.
        torch.cuda.synchronize()

        # 3) Snapshot the entire bucket buffer (pre-reduce)
        logger.debug(
            f"[Rank {orch._rank}] Snapshotting bucket {bucket_index} before allreduce"
        )
        orch.on_bucket_snapshot(bucket.buffer(), bucket_index, pg)

        # work = dist.ulfm_all_reduce(bucket.buffer(), async_op=True, ulfm_opts=ulfm_opts)
        logger.debug(
            f"[Rank {orch._rank}] Submitting ulfm_allreduce for bucket {bucket_index}"
        )
        work = pg.ulfm_allreduce([bucket.buffer()], opts, ulfm_opts)

        def on_done(fut):
            logger.debug(
                f"[Rank {orch._rank}] Allreduce completed for bucket {bucket_index}, "
                f"entering work completion handler"
            )
            # Use orchestrator's unified entry point for handling work completion
            # This encapsulates all failure detection, policy consultation, and recovery logic
            # get_failure_simulator() is called here (not at hook-registration time) so that
            # simulators set after DDP construction are picked up correctly.
            _sim = get_failure_simulator()
            ctx = _sim.may_fail_here("post-allreduce") if _sim is not None else contextlib.nullcontext()
            with ctx:
                orch.handle_work_completion(
                    work=work,
                    bucket_index=bucket_index,
                )

            # Increment hook invocation counter
            orch.increment_hook_counter()
            logger.debug(
                f"[Rank {orch._rank}] Hook done for bucket {bucket_index}, "
                f"hook_count={orch._hook_invocation_counter}"
            )

            return fut.value()[0]

        return work.get_future().then(on_done)

    return hook
