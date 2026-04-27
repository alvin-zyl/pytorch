"""HSDPULFMTrainingManager: ULFMTrainingManager adapted for FSDP1 HYBRID_SHARD.

Differences from the DDP-based parent:
  - Model is already FSDP-wrapped before being passed in (no internal DDP wrap).
  - Orchestrator's process group is replicate_pg (cross-replica), not world.
  - Registers create_ulfm_hsdp_hook instead of create_ulfm_recovery_hook.

Everything else — policy, orchestrator, train_step microbatch state machine,
no_sync, restore modes, optimizer commit — is inherited unchanged.
"""

import logging
import contextlib
import torch
import torch.distributed as dist
from torch.distributed.fsdp._traversal_utils import _get_fsdp_states

import ulfm_collectives as ULFM
from .training_manager import ULFMTrainingManager
from .orchestrator import StepTxnOrchestrator
from .policy import create_policy, GradRestoreMode
from .ulfm_hook import HSDPHookState, create_ulfm_hsdp_hook

logger = logging.getLogger(__name__)


class HSDPULFMTrainingManager(ULFMTrainingManager):
    """ULFMTrainingManager for FSDP1 HYBRID_SHARD.

    Args:
        fsdp_model: FullyShardedDataParallel(hf_model, sharding_strategy=HYBRID_SHARD, ...)
        replicate_pg: cross-replica ULFM process group
        grad_accum_steps: number of microbatches per optimizer step
        policy_type: "static" or "adaptive"
        **policy_kwargs: forwarded to create_policy
    """

    def __init__(
        self,
        fsdp_model,
        replicate_pg,
        grad_accum_steps: int = 1,
        policy_type: str = "static",
        **policy_kwargs,
    ):
        if not isinstance(replicate_pg, ULFM.ProcessGroupULFM):
            raise ValueError(
                f"replicate_pg must be a ProcessGroupULFM, got {type(replicate_pg)}"
            )

        self.failure_strategy = "continue"
        self.process_group = replicate_pg
        # Derived world PG (sibling of replicate_pg / shard_pg, all rooted at
        # MPI_COMM_WORLD), mirroring nanotron's ParallelContext.world_pg.
        # Falls back to dist.group.WORLD (the default group) when not provided.

        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=True,
            **policy_kwargs,
        )

        ulfm_opts = self._create_ulfm_opts(policy_type)

        rank = dist.get_rank()
        self.txn = StepTxnOrchestrator(
            rank=rank, pg=replicate_pg, policy=policy, ulfm_opts=ulfm_opts
        )

        # Already-wrapped FSDP model. Keep the attribute name `ddp_model` so
        # inherited train_step() code (which calls self.ddp_model.no_sync(),
        # self.ddp_model(data), self.ddp_model.parameters()) works unchanged —
        # FSDP1 provides all three.
        self.ddp_model = fsdp_model

        # FSDP defaults to dividing grads by (shard × replicate). We keep its
        # SHARD scaling (shard_pg is stable under our failure model — failures
        # kill replicas, not shards) and strip out the replicate factor, so
        # the parent's _get_grad_div_factor (= target_replicate × grad_accum
        # under StaticWorldPolicy) is the only replicate-axis divisor and
        # stays constant across rank failures.
        self._strip_fsdp_replicate_scaling(fsdp_model)

        self._hook_state = HSDPHookState(pg=replicate_pg, orchestrator=self.txn)
        self._register_ulfm_hook(ulfm_opts=ulfm_opts)

        self._micro_in_window = 0

        logger.debug(
            f"[Rank {rank}] HSDPULFMTrainingManager initialized: "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

    @staticmethod
    def _strip_fsdp_replicate_scaling(fsdp_model) -> None:
        """Set predivide=1, postdivide=shard_size on every FSDP unit so
        FSDP divides only by the (stable) shard dimension; the replicate-axis
        divisor is then applied solely by the parent's _get_grad_div_factor
        (= target_replicate × grad_accum under StaticWorldPolicy, constant
        across failures).
        """
        states = _get_fsdp_states(fsdp_model)
        if not states:
            raise RuntimeError("HSDP manager: no FSDP states found in model")
        for state in states:
            if state.process_group is None:
                raise RuntimeError(
                    "HSDP manager: FSDP state missing shard process_group"
                )
            shard_size = state.process_group.size()
            state._gradient_predivide_factor = 1.0
            state._gradient_postdivide_factor = float(shard_size)

    def _register_ulfm_hook(self, ulfm_opts):
        """Register the FSDP-shaped ULFM hook on the FSDP model."""
        hook = create_ulfm_hsdp_hook(ulfm_opts=ulfm_opts)
        self.ddp_model.register_comm_hook(state=self._hook_state, hook=hook)
        logger.debug(
            f"[Rank {self.txn._rank}] HSDP ULFM hook registered on replicate_pg"
        )
        self._hook = hook

    def train_step(self, batch_idx, data, target, criterion, optimizer, scaler=None, grad_clipping: float = 0.0):
        """Reset per-step unit counter, then delegate to the parent."""
        self._hook_state.reset_unit_counter()
        self.txn.update_progress(
            microbatch_idx=self._micro_in_window,
            total_microbatches=self._get_grad_accum_steps(),
            macrobatch_idx=batch_idx,
        )

        # === Window start: initialize ===
        if self._micro_in_window == 0:
            # Notify policy of window start
            self._notify_window_start()

            # Zero gradients at start of window
            optimizer.zero_grad(set_to_none=False)

        # === Check if orchestrator flagged need for restoration (set by hook) ===
        # If at policy boundary: use non-blocking restoration during forward
        # (This flag is set by hook when policy.on_failure() returns at_iteration_boundary=True)
        restore_mode = self._get_restore_mode()

        if restore_mode == GradRestoreMode.NON_BLOCKING:
            # At policy boundary: Start async restoration during forward
            logger.info(
                f"[Rank {self.txn._rank}] At policy boundary - starting non-blocking grad restoration"
            )
            self._start_restore_gradients_non_blocking()

        # === Backward (use no_sync on non-last microbatches) ===
        if self._is_at_grad_sync_step:
            ctx = contextlib.nullcontext()
            self._on_grad_sync_step()
        else:
            ctx = self.ddp_model.no_sync()

        with ctx:
            logger.debug(
                f"[Rank {self.txn._rank}] Backward pass at microbatch {self._micro_in_window} "
                f"no_sync={not self._is_at_grad_sync_step}"
            )
            # === Forward ===
            output = self.ddp_model(data)

            # === Wait for async restoration if it was started ===
            if restore_mode == GradRestoreMode.NON_BLOCKING:
                self._wait_restore_before_backward()
                logger.debug(
                    f"[Rank {self.txn._rank}] Non-blocking restoration completed before backward"
                )

            loss = self._may_zero_grad(criterion(output, target))
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

        # === After backward: check with policy if we should commit ===
        state = self._on_microbatch_complete(self._micro_in_window)
        restore_mode = self._get_restore_mode()
        stepped = False

        # === Decide whether to commit optimizer step ===
        if state.at_iteration_boundary:
            logger.debug(
                f"[Rank {self.txn._rank}] Microbatch index: {self._micro_in_window}, grad_acc_step: {self._get_grad_accum_steps()}, "
                f"at iteration boundary"
            )
            self._on_grad_sync_step()

            # If NOT at policy boundary but need restoration: blocking restore before optimizer
            if restore_mode == GradRestoreMode.BLOCKING:
                logger.info(
                    f"[Rank {self.txn._rank}] Not at policy boundary - blocking grad restoration before optimizer"
                )
                self._start_restore_gradients_blocking()
                logger.debug(f"[Rank {self.txn._rank}] Blocking restoration finished.")

            # Optimizer step
            if scaler is None:
                for p in self.ddp_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(self._get_grad_div_factor())
                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.ddp_model.parameters(), grad_clipping
                    )
                grad_norm = self._compute_grad_norm()
                optimizer.step()
            else:
                if hasattr(scaler, "unscale_"):
                    scaler.unscale_(optimizer)
                for p in self.ddp_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(self._get_grad_div_factor())
                if grad_clipping != 0.0:
                    torch.nn.utils.clip_grad_norm_(
                        self.ddp_model.parameters(), grad_clipping
                    )
                grad_norm = self._compute_grad_norm()
                scaler.step(optimizer)
                scaler.update()

            optimizer.zero_grad(set_to_none=False)
            stepped = True

            # Notify orchestrator
            self._on_step_committed()

            # Reset for next window
            self._micro_in_window = 0

        else:
            # Not at window boundary: continue accumulation
            self._micro_in_window += 1
            grad_norm = 0.0

        return float(loss.detach()), stepped, grad_norm

    def _compute_grad_norm(self) -> float:
        """Mirror the NCCL baseline's grad_norm: sum of per-param L2 norms on
        each rank's local shards. Not a true global norm — kept identical to
        baseline so wandb curves are comparable."""
        total = 0.0
        for p in self.ddp_model.parameters():
            if p.grad is not None:
                total += float(torch.norm(p.grad.detach()).cpu())
        return total
