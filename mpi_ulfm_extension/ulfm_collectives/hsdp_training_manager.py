"""HSDPULFMTrainingManager: ULFMTrainingManager adapted for FSDP1 HYBRID_SHARD.

Differences from the DDP-based parent:
  - Model is already FSDP-wrapped before being passed in (no internal DDP wrap).
  - Orchestrator's process group is replicate_pg (cross-replica), not world.
  - Registers create_ulfm_hsdp_hook instead of create_ulfm_recovery_hook.

Everything else — policy, orchestrator, train_step microbatch state machine,
no_sync, restore modes, optimizer commit — is inherited unchanged.
"""

import logging

import torch.distributed as dist

import ulfm_collectives as ULFM
from .training_manager import ULFMTrainingManager
from .orchestrator import StepTxnOrchestrator
from .policy import create_policy
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

        self._hook_state = HSDPHookState(pg=replicate_pg, orchestrator=self.txn)
        self._register_ulfm_hook(ulfm_opts=ulfm_opts)

        self._micro_in_window = 0

        logger.debug(
            f"[Rank {rank}] HSDPULFMTrainingManager initialized: "
            f"policy={policy_type}, grad_accum={grad_accum_steps}"
        )

    def _register_ulfm_hook(self, ulfm_opts):
        """Register the FSDP-shaped ULFM hook on the FSDP model."""
        hook = create_ulfm_hsdp_hook(ulfm_opts=ulfm_opts)
        self.ddp_model.register_comm_hook(state=self._hook_state, hook=hook)
        logger.debug(
            f"[Rank {self.txn._rank}] HSDP ULFM hook registered on replicate_pg"
        )
        self._hook = hook

    def train_step(self, *args, **kwargs):
        """Reset per-step unit counter, then delegate to the parent."""
        self._hook_state.reset_unit_counter()
        return super().train_step(*args, **kwargs)
