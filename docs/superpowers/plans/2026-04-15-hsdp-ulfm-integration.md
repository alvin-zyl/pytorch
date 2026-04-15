# HSDP + ULFM Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a standalone HSDP training binary (`main_hsdp.py`) with two variants — NCCL baseline and ULFM-fault-tolerant — built on PyTorch FSDP1 `HYBRID_SHARD` and reusing the existing ULFM orchestrator/policy stack.

**Architecture:** FSDP1 `HYBRID_SHARD` with a 2D group layout built from `dist.new_group`: intra-replica shard PG (NCCL) and cross-replica replicate PG (NCCL or ULFM). The ULFM variant registers a new FSDP-shaped comm hook (`create_ulfm_hsdp_hook`) that snapshots the reduce-scattered bf16 shard grad, submits `ulfm_allreduce` on `replicate_pg`, and routes through the existing `StepTxnOrchestrator`. A thin `HSDPULFMTrainingManager(ULFMTrainingManager)` subclass replaces the DDP wrap with an already-wrapped FSDP model and registers the new hook. Orchestrator, policy, failure simulator, and C++ extension are unchanged.

**Tech Stack:** PyTorch (FSDP1 HYBRID_SHARD, `register_comm_hook`), MPI via the existing `ulfm_collectives` extension, HuggingFace Transformers (Llama), OpenMPI 5.0.8+ with ULFM.

**Reference spec:** `docs/superpowers/specs/2026-04-15-hsdp-ulfm-integration-design.md`.

**File structure:**

New files:
- `mpi_ulfm_extension/main_hsdp.py` — forked from `main.py`; adds `--backend`, `--hsdp_shard_size`, FSDP wrapping, group construction, hook registration, variant branches.
- `mpi_ulfm_extension/ulfm_collectives/hsdp_training_manager.py` — `HSDPULFMTrainingManager(ULFMTrainingManager)` subclass.
- `mpi_ulfm_extension/ulfm_collectives/hsdp_groups.py` — pure helpers for the 2D rank layout (unit-testable without MPI/distributed).
- `mpi_ulfm_extension/test/test_hsdp_groups.py` — pytest for the layout helpers.
- `mpi_ulfm_extension/launch_hsdp_nccl.sh` — launcher for the NCCL baseline.
- `mpi_ulfm_extension/launch_hsdp_ulfm.sh` — launcher for the ULFM variant.

Modified files:
- `mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py` — add `create_ulfm_hsdp_hook(ulfm_opts)` factory + an `HSDPHookState` dataclass that carries a unit-index counter.
- `mpi_ulfm_extension/ulfm_collectives/__init__.py` — export `HSDPULFMTrainingManager`, `create_ulfm_hsdp_hook`, `HSDPHookState`, and the layout helpers.

Unchanged: `orchestrator.py`, `policy.py`, `failure_simulator.py`, C++ extension (`src/`, `include/`).

---

## Task 1: Pure layout helpers (`hsdp_groups.py`)

**Rationale:** The 2D rank layout is pure arithmetic and fully unit-testable without MPI. Extracting it gives us one piece with tight tests before any distributed glue.

**Files:**
- Create: `mpi_ulfm_extension/ulfm_collectives/hsdp_groups.py`
- Create: `mpi_ulfm_extension/test/test_hsdp_groups.py`

- [ ] **Step 1: Write the failing test**

Create `mpi_ulfm_extension/test/test_hsdp_groups.py`:

```python
import pytest
from ulfm_collectives.hsdp_groups import (
    compute_hsdp_layout,
    replica_ranks,
    replicate_peer_ranks,
    replica0_ranks,
)


def test_layout_divisibility():
    with pytest.raises(ValueError, match="divisible"):
        compute_hsdp_layout(world_size=5, shard_size=2)


def test_layout_values():
    layout = compute_hsdp_layout(world_size=8, shard_size=2)
    assert layout.num_replicas == 4
    assert layout.shard_size == 2


def test_replica_ranks_contiguous():
    # shard_size=2, world=8: replica 0=[0,1], 1=[2,3], 2=[4,5], 3=[6,7]
    assert replica_ranks(replica_id=0, shard_size=2) == [0, 1]
    assert replica_ranks(replica_id=2, shard_size=2) == [4, 5]


def test_replicate_peer_ranks():
    # shard_size=2, num_replicas=4: offset 0 -> [0,2,4,6], offset 1 -> [1,3,5,7]
    assert replicate_peer_ranks(offset=0, shard_size=2, num_replicas=4) == [0, 2, 4, 6]
    assert replicate_peer_ranks(offset=1, shard_size=2, num_replicas=4) == [1, 3, 5, 7]


def test_replica0_ranks():
    assert replica0_ranks(shard_size=2) == [0, 1]
    assert replica0_ranks(shard_size=4) == [0, 1, 2, 3]


def test_rank_decomposition_roundtrip():
    layout = compute_hsdp_layout(world_size=8, shard_size=2)
    for rank in range(8):
        rid = layout.replica_id_of(rank)
        sr = layout.shard_rank_of(rank)
        assert 0 <= rid < 4
        assert 0 <= sr < 2
        assert rank in replica_ranks(rid, 2)
        assert rank in replicate_peer_ranks(sr, 2, 4)
```

- [ ] **Step 2: Run test to verify it fails**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
python -m pytest test/test_hsdp_groups.py -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'ulfm_collectives.hsdp_groups'`.

- [ ] **Step 3: Write minimal implementation**

Create `mpi_ulfm_extension/ulfm_collectives/hsdp_groups.py`:

```python
"""Pure helpers for the HSDP 2D rank layout. No distributed imports."""

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class HSDPLayout:
    world_size: int
    shard_size: int

    @property
    def num_replicas(self) -> int:
        return self.world_size // self.shard_size

    def replica_id_of(self, rank: int) -> int:
        return rank // self.shard_size

    def shard_rank_of(self, rank: int) -> int:
        return rank % self.shard_size


def compute_hsdp_layout(world_size: int, shard_size: int) -> HSDPLayout:
    if shard_size <= 0 or world_size <= 0:
        raise ValueError("world_size and shard_size must be positive")
    if world_size % shard_size != 0:
        raise ValueError(
            f"world_size ({world_size}) must be divisible by shard_size ({shard_size})"
        )
    return HSDPLayout(world_size=world_size, shard_size=shard_size)


def replica_ranks(replica_id: int, shard_size: int) -> List[int]:
    """Ranks belonging to `replica_id` (contiguous block)."""
    start = replica_id * shard_size
    return list(range(start, start + shard_size))


def replicate_peer_ranks(offset: int, shard_size: int, num_replicas: int) -> List[int]:
    """Ranks that share the same intra-replica offset across all replicas."""
    return [offset + r * shard_size for r in range(num_replicas)]


def replica0_ranks(shard_size: int) -> List[int]:
    return list(range(shard_size))
```

- [ ] **Step 4: Run test to verify it passes**

```bash
python -m pytest test/test_hsdp_groups.py -v
```

Expected: all 6 tests PASS.

- [ ] **Step 5: Commit**

```bash
git add mpi_ulfm_extension/ulfm_collectives/hsdp_groups.py \
        mpi_ulfm_extension/test/test_hsdp_groups.py
git commit -m "Add pure helpers for HSDP 2D rank layout"
```

---

## Task 2: HSDP comm hook factory

**Rationale:** New factory in `ulfm_hook.py` that matches FSDP1's `register_comm_hook` signature `(state, grad_tensor) -> Future` and routes through the existing orchestrator. Mirrors `create_ulfm_recovery_hook` with the bucket replaced by an FSDP shard grad.

**Files:**
- Modify: `mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py` (append new dataclass + factory at end of file)

- [ ] **Step 1: Add `HSDPHookState` dataclass**

Append to `mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py` (after existing factories, before the final blank line):

```python
@dataclass
class HSDPHookState:
    """State for the FSDP1 HYBRID_SHARD ULFM hook.

    Unlike DDP, FSDP fires the hook per FSDP unit (not per bucket). We assign
    a stable unit index in registration order — FSDP unit order is
    deterministic across ranks, so every rank sees the same indices.
    """

    pg: "Union[ULFM.ProcessGroupULFM, dist.ProcessGroup]"
    orchestrator: StepTxnOrchestrator
    _unit_counter: int = 0

    def next_unit_index(self) -> int:
        idx = self._unit_counter
        self._unit_counter += 1
        return idx

    def reset_unit_counter(self) -> None:
        """Call between training steps so unit indices restart at 0."""
        self._unit_counter = 0
```

- [ ] **Step 2: Add `create_ulfm_hsdp_hook` factory**

Append to `mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py`:

```python
def create_ulfm_hsdp_hook(ulfm_opts: ULFM.ULFMOptions = None):
    """
    Create an FSDP1 HYBRID_SHARD ULFM comm hook.

    FSDP fires this after the intra-replica reduce-scatter. The hook's job
    is to perform the cross-replica allreduce on the shard grad. On failure,
    the orchestrator snapshots and restores the bf16 shard grad in place.

    Hook signature per FSDP1: (state, grad_shard: torch.Tensor) -> Future[Tensor].

    Up/down-casting to fp32 for MPI happens inside the C++ ulfm_allreduce;
    Python-side stays in bf16.

    Returns:
        Callable hook compatible with FullyShardedDataParallel.register_comm_hook()
    """
    opts = torch.distributed.AllreduceOptions()
    opts.reduceOp = torch.distributed.ReduceOp.SUM
    ulfm_opts = ulfm_opts if ulfm_opts is not None else ULFM.ULFMOptions()

    def hook(state: HSDPHookState, grad_shard: torch.Tensor):
        pg = state.pg
        orch = state.orchestrator
        unit_index = state.next_unit_index()

        logger.debug(
            f"[Rank {orch._rank}] HSDP hook entered for unit {unit_index}, "
            f"numel={grad_shard.numel()}, dtype={grad_shard.dtype}"
        )

        # 1) Quiesced? NOOP.
        if getattr(pg, "is_quiesced", lambda: False)():
            logger.warning(
                f"[Rank {orch._rank}] replicate_pg quiesced — skipping unit {unit_index}."
            )
            fut = torch.futures.Future()
            fut.set_result(grad_shard)
            return fut

        # 2) Sync GPU work before entering MPI (same rationale as DDP hook).
        torch.cuda.synchronize()

        # 3) Snapshot the bf16 shard grad for restore.
        orch.on_bucket_snapshot(grad_shard, unit_index, pg)

        # 4) Submit ulfm_allreduce on replicate_pg. up/down-cast lives in C++.
        work = pg.ulfm_allreduce([grad_shard], opts, ulfm_opts)

        def on_done(fut):
            _sim = get_failure_simulator()
            ctx = _sim.may_fail_here("post-allreduce") if _sim is not None else contextlib.nullcontext()
            with ctx:
                orch.handle_work_completion(work=work, bucket_index=unit_index)
            orch.increment_hook_counter()
            return fut.value()[0]

        return work.get_future().then(on_done)

    return hook
```

- [ ] **Step 3: Import check**

Run:

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
python -c "from ulfm_collectives.ulfm_hook import create_ulfm_hsdp_hook, HSDPHookState; \
           s = HSDPHookState(pg=None, orchestrator=None); \
           assert s.next_unit_index() == 0; \
           assert s.next_unit_index() == 1; \
           s.reset_unit_counter(); \
           assert s.next_unit_index() == 0; \
           print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py
git commit -m "Add create_ulfm_hsdp_hook factory and HSDPHookState"
```

---

## Task 3: `HSDPULFMTrainingManager` subclass

**Rationale:** Minimal subclass of `ULFMTrainingManager` that (a) accepts an already-FSDP-wrapped model instead of wrapping DDP internally, (b) uses `replicate_pg` as the orchestrator's PG, and (c) registers the new FSDP hook.

**Files:**
- Create: `mpi_ulfm_extension/ulfm_collectives/hsdp_training_manager.py`

- [ ] **Step 1: Write the training manager**

Create `mpi_ulfm_extension/ulfm_collectives/hsdp_training_manager.py`:

```python
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
```

- [ ] **Step 2: Import smoke test**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
python -c "from ulfm_collectives.hsdp_training_manager import HSDPULFMTrainingManager; print('ok')"
```

Expected: prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add mpi_ulfm_extension/ulfm_collectives/hsdp_training_manager.py
git commit -m "Add HSDPULFMTrainingManager subclass for FSDP1 HYBRID_SHARD"
```

---

## Task 4: Export new symbols

**Files:**
- Modify: `mpi_ulfm_extension/ulfm_collectives/__init__.py`

- [ ] **Step 1: Read existing `__init__.py`**

```bash
cat mpi_ulfm_extension/ulfm_collectives/__init__.py
```

- [ ] **Step 2: Append new exports**

Append to `mpi_ulfm_extension/ulfm_collectives/__init__.py` (keep existing content intact):

```python
# HSDP integration
from .hsdp_groups import (
    HSDPLayout,
    compute_hsdp_layout,
    replica_ranks,
    replicate_peer_ranks,
    replica0_ranks,
)
from .hsdp_training_manager import HSDPULFMTrainingManager
from .ulfm_hook import HSDPHookState, create_ulfm_hsdp_hook
```

- [ ] **Step 3: Verify exports resolve**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
python -c "
import ulfm_collectives as U
for name in ('HSDPLayout','compute_hsdp_layout','replica_ranks',
            'replicate_peer_ranks','replica0_ranks',
            'HSDPULFMTrainingManager','HSDPHookState','create_ulfm_hsdp_hook'):
    assert hasattr(U, name), name
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 4: Commit**

```bash
git add mpi_ulfm_extension/ulfm_collectives/__init__.py
git commit -m "Export HSDP integration symbols from ulfm_collectives"
```

---

## Task 5: `main_hsdp.py` — skeleton forked from `main.py`

**Rationale:** Copy `main.py` and lay down the new flags. No FSDP yet — this task just establishes the file, confirms it still runs in `--single_gpu` mode, and adds the new argparse options as no-ops so later tasks have a stable base.

**Files:**
- Create: `mpi_ulfm_extension/main_hsdp.py`

- [ ] **Step 1: Copy `main.py` to `main_hsdp.py`**

```bash
cp mpi_ulfm_extension/main.py mpi_ulfm_extension/main_hsdp.py
```

- [ ] **Step 2: Add new flags**

In `mpi_ulfm_extension/main_hsdp.py`, inside `parse_args`, add two new arguments next to the existing `--single_gpu`:

```python
parser.add_argument(
    "--backend",
    type=str,
    default="ulfm",
    choices=["nccl", "ulfm"],
    help="Cross-replica backend: 'nccl' (baseline) or 'ulfm' (fault-tolerant).",
)
parser.add_argument(
    "--hsdp_shard_size",
    type=int,
    default=None,
    help="Ranks per FSDP replica. Default: torch.cuda.device_count(). "
         "world_size must be divisible by this value.",
)
```

- [ ] **Step 3: Import smoke test**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('main_hsdp', 'main_hsdp.py')
mod = importlib.util.module_from_spec(spec)
# Do not execute __main__. Just load the module namespace.
sys.modules['main_hsdp'] = mod
spec.loader.exec_module(mod)
args = mod.parse_args(['--model_config','configs/llama_60m.json',
                       '--batch_size','1','--save_dir','/tmp/x',
                       '--total_batch_size','1','--single_gpu',
                       '--backend','nccl','--hsdp_shard_size','2'])
assert args.backend == 'nccl'
assert args.hsdp_shard_size == 2
print('ok')
"
```

Expected: prints `ok` (may print some config warnings; those are fine).

- [ ] **Step 4: Commit**

```bash
git add mpi_ulfm_extension/main_hsdp.py
git commit -m "Fork main.py to main_hsdp.py and add --backend, --hsdp_shard_size"
```

---

## Task 6: Process-group construction in `main_hsdp.py`

**Rationale:** Build `shard_pg` and `replicate_pg` after `dist.init_process_group(...)`. The world PG backend is `nccl` when `--backend nccl`, `ulfm` when `--backend ulfm`. Shard PG is always NCCL. Replicate PG matches the world backend.

**Files:**
- Modify: `mpi_ulfm_extension/main_hsdp.py`

- [ ] **Step 1: Add a group-building helper**

In `mpi_ulfm_extension/main_hsdp.py`, add this helper above `main(args)`:

```python
from ulfm_collectives.hsdp_groups import (
    compute_hsdp_layout,
    replica_ranks,
    replicate_peer_ranks,
)


def build_hsdp_groups(world_size: int, shard_size: int, backend: str):
    """Build (shard_pg, replicate_pg, layout) for the current rank.

    shard_pg is always NCCL. replicate_pg matches `backend`
    ('nccl' or 'ulfm'). Every group is created on every rank (required by
    PyTorch dist.new_group), but each rank only belongs to one of each kind.
    """
    layout = compute_hsdp_layout(world_size=world_size, shard_size=shard_size)
    my_rank = dist.get_rank()

    # --- Shard groups: one per replica ---
    shard_pg = None
    for rid in range(layout.num_replicas):
        ranks = replica_ranks(rid, shard_size)
        pg = dist.new_group(ranks=ranks, backend="nccl")
        if my_rank in ranks:
            shard_pg = pg

    # --- Replicate groups: one per intra-replica offset ---
    replicate_pg = None
    for offset in range(shard_size):
        ranks = replicate_peer_ranks(offset, shard_size, layout.num_replicas)
        # backend=None on the ULFM side inherits the world backend (ulfm);
        # backend='nccl' on the baseline side forces NCCL explicitly.
        if backend == "ulfm":
            pg = dist.new_group(ranks=ranks, backend=None)
        else:
            pg = dist.new_group(ranks=ranks, backend="nccl")
        if my_rank in ranks:
            replicate_pg = pg

    assert shard_pg is not None and replicate_pg is not None
    return shard_pg, replicate_pg, layout
```

- [ ] **Step 2: Swap world-PG init to respect `--backend`**

In `main(args)`, replace:

```python
    dist.init_process_group(backend="ulfm")
```

with:

```python
    dist.init_process_group(backend=args.backend)
```

- [ ] **Step 3: Build the subgroups after world init**

In `main(args)`, immediately after the block that sets `global_rank`, `world_size`, `local_rank`, and calls `torch.cuda.set_device(local_rank)`, and before the FailureSimulator block, add:

```python
    if not args.single_gpu:
        if args.hsdp_shard_size is None:
            shard_size = max(torch.cuda.device_count(), 1)
        else:
            shard_size = args.hsdp_shard_size
        shard_pg, replicate_pg, hsdp_layout = build_hsdp_groups(
            world_size=world_size, shard_size=shard_size, backend=args.backend
        )
        logger.info(
            f"HSDP layout: num_replicas={hsdp_layout.num_replicas}, "
            f"shard_size={shard_size}, my replica={hsdp_layout.replica_id_of(global_rank)}, "
            f"my shard_rank={hsdp_layout.shard_rank_of(global_rank)}"
        )
    else:
        shard_pg = replicate_pg = hsdp_layout = None
```

- [ ] **Step 4: Commit**

```bash
git add mpi_ulfm_extension/main_hsdp.py
git commit -m "main_hsdp: build shard_pg and replicate_pg after world init"
```

---

## Task 7: FSDP wrapping in `main_hsdp.py`

**Rationale:** Replace the DDP-implicit model with an explicit FSDP1 HYBRID_SHARD wrap that uses `(shard_pg, replicate_pg)` as the 2D process group. Applies regardless of `--backend`; the baseline variant just gets an all-NCCL HSDP.

**Files:**
- Modify: `mpi_ulfm_extension/main_hsdp.py`

- [ ] **Step 1: Add FSDP imports at the top of `main_hsdp.py`**

After the existing `import torch.distributed as dist` line, add:

```python
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
import functools
```

- [ ] **Step 2: Wrap the model with FSDP after it's created**

In `main(args)`, immediately after the block that creates `model` (the `if args.continue_from is not None:` / `else:` branch that ends with `.to(device=...)`), and before `if args.activation_checkpointing:`, add:

```python
    if not args.single_gpu:
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        model = FullyShardedDataParallel(
            model,
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            process_group=(shard_pg, replicate_pg),
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
            mixed_precision=None,
        )
        logger.info(f"[Rank {global_rank}] Model wrapped with FSDP HYBRID_SHARD")
```

- [ ] **Step 3: Commit**

```bash
git add mpi_ulfm_extension/main_hsdp.py
git commit -m "main_hsdp: wrap model with FSDP HYBRID_SHARD using (shard_pg, replicate_pg)"
```

---

## Task 8: Wire training manager + training-loop branches

**Rationale:** Replace the DDP-based `ULFMTrainingManager` with `HSDPULFMTrainingManager` for the ULFM variant. The NCCL baseline variant uses FSDP directly — no manager, no hook, no failure simulator. The per-microbatch loop body is unified via a simple branch.

**Files:**
- Modify: `mpi_ulfm_extension/main_hsdp.py`

- [ ] **Step 1: Swap the training-manager construction**

In `main(args)`, replace the existing block:

```python
    if not args.single_gpu:
        if not _ULFM_AVAILABLE:
            raise RuntimeError("ulfm_collectives not available; cannot run distributed training without ULFM.")
        training_manager = ULFMTrainingManager(
            LMWrapper(model),
            grad_accum_steps=args.gradient_accumulation,
            failure_strategy="continue",
            enable_auto_repair=True,
            policy_type="static",
            initial_world_size=world_size,
        )
```

with:

```python
    training_manager = None
    if not args.single_gpu and args.backend == "ulfm":
        if not _ULFM_AVAILABLE:
            raise RuntimeError(
                "ulfm_collectives not available; cannot run --backend ulfm without it."
            )
        from ulfm_collectives.hsdp_training_manager import HSDPULFMTrainingManager
        training_manager = HSDPULFMTrainingManager(
            fsdp_model=model,
            replicate_pg=replicate_pg,
            grad_accum_steps=args.gradient_accumulation,
            policy_type="static",
            initial_world_size=hsdp_layout.num_replicas,
        )
```

Also update the `sim` guard to skip the failure simulator on the baseline path. Find:

```python
    if _ULFM_AVAILABLE and not args.single_gpu:
        sim = FailureSimulator(
            ...
        )
        set_failure_simulator(sim)
        sim.initialize(rank=global_rank, world_size=world_size)
    else:
        sim = None
```

Change the condition and the exclusion list to:

```python
    if _ULFM_AVAILABLE and not args.single_gpu and args.backend == "ulfm":
        from ulfm_collectives.hsdp_groups import replica0_ranks as _r0
        sim = FailureSimulator(
            seed=42,
            desired_failures=0,
            total_minibatches=100 * args.gradient_accumulation,
            target_ranks={},
            config_path=None,
            start_minibatch=args.failure_start_step,
        )
        sim.excluded_ranks = set(_r0(shard_size))
        set_failure_simulator(sim)
        sim.initialize(rank=global_rank, world_size=world_size)
    else:
        sim = None
```

- [ ] **Step 2: Update the per-microbatch loop body**

Find the block in the training loop that currently reads:

```python
        if args.single_gpu:
            loss = model(**batch).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            if global_step % args.gradient_accumulation != 0:
                continue
            stepped = True
            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            grad_norm = sum(
                [
                    torch.norm(p.grad.clone().detach().cpu())
                    for p in model.parameters()
                    if p.grad is not None
                ]
            )
            if not layer_wise_flag:
                optimizer.step()
                optimizer.zero_grad()
        else:
            sim.begin_minibatch(batch_idx)
            with sim.may_fail_here("pre-forward"):
                loss, stepped = training_manager.train_step(
                    batch_idx, batch, None, lm_criterion, optimizer
                )
            if not stepped:
                continue
            grad_norm = 0.0
```

Replace with:

```python
        if args.single_gpu:
            loss = model(**batch).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            if global_step % args.gradient_accumulation != 0:
                continue
            stepped = True
            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(trainable_params, args.grad_clipping)
            grad_norm = 0.0
            if not layer_wise_flag:
                optimizer.step()
                optimizer.zero_grad()
        elif args.backend == "nccl":
            # FSDP baseline: no manager, no hook, no failure simulator.
            # FSDP itself handles no_sync across microbatches if requested;
            # here we sync every microbatch for simplicity in stage 1.
            loss = model(batch=batch).loss if hasattr(model, "module") else model(**batch).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            if global_step % args.gradient_accumulation != 0:
                continue
            stepped = True
            grad_norm = 0.0
            if not layer_wise_flag:
                optimizer.step()
                optimizer.zero_grad()
        else:
            sim.begin_minibatch(batch_idx)
            with sim.may_fail_here("pre-forward"):
                loss, stepped = training_manager.train_step(
                    batch_idx, batch, None, lm_criterion, optimizer
                )
            if not stepped:
                continue
            grad_norm = 0.0
```

**Note:** `training_manager.train_step` expects its model argument to accept a single batch dict. The existing `LMWrapper` handles that; `HSDPULFMTrainingManager` receives the already-FSDP-wrapped HF model, which expects kwargs. In Task 3 we stored the FSDP model as `self.ddp_model`; the parent's `train_step` calls `self.ddp_model(data)` where `data` is the microbatch tensor dict. HF models called as `model(batch_dict)` treat the dict as `input_ids` — wrong. So wrap with `LMWrapper` *before* FSDP:

- [ ] **Step 3: Wrap with `LMWrapper` before FSDP**

In `main_hsdp.py`, in the FSDP wrapping block added in Task 7, change:

```python
        model = FullyShardedDataParallel(
            model,
```

to:

```python
        model = FullyShardedDataParallel(
            LMWrapper(model),
```

And update the NCCL-baseline loop body (Step 2 above) from `model(**batch)` to `model(batch)` to match `LMWrapper.forward`. The corrected NCCL branch is:

```python
        elif args.backend == "nccl":
            loss = model(batch).loss
            scaled_loss = loss / args.gradient_accumulation
            scaled_loss.backward()
            if global_step % args.gradient_accumulation != 0:
                continue
            stepped = True
            grad_norm = 0.0
            if not layer_wise_flag:
                optimizer.step()
                optimizer.zero_grad()
```

Apply this correction (both the `LMWrapper` wrap and the NCCL branch call-site).

- [ ] **Step 4: Import smoke test**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
python -c "
import importlib.util, sys
spec = importlib.util.spec_from_file_location('main_hsdp', 'main_hsdp.py')
mod = importlib.util.module_from_spec(spec)
sys.modules['main_hsdp'] = mod
spec.loader.exec_module(mod)
print('ok')
"
```

Expected: prints `ok`.

- [ ] **Step 5: Commit**

```bash
git add mpi_ulfm_extension/main_hsdp.py
git commit -m "main_hsdp: wire HSDPULFMTrainingManager and NCCL-baseline loop branches"
```

---

## Task 9: Launch scripts

**Rationale:** Repeatable multi-rank launch commands for each variant, mirroring existing `launch_ulfm.sh` / `launch_original.sh` conventions.

**Files:**
- Create: `mpi_ulfm_extension/launch_hsdp_nccl.sh`
- Create: `mpi_ulfm_extension/launch_hsdp_ulfm.sh`

- [ ] **Step 1: Read existing launchers to copy conventions**

```bash
cat mpi_ulfm_extension/launch_ulfm.sh
cat mpi_ulfm_extension/launch_original.sh
```

- [ ] **Step 2: Create `launch_hsdp_nccl.sh`**

```bash
cat > mpi_ulfm_extension/launch_hsdp_nccl.sh <<'EOF'
#!/usr/bin/env bash
# HSDP NCCL baseline: 2 replicas × 2 shards = 4 ranks.
# Use torchrun so NCCL world init is straightforward.
set -euo pipefail

cd "$(dirname "$0")"

torchrun \
  --nproc_per_node=4 \
  --nnodes=1 \
  --master_addr=127.0.0.1 \
  --master_port=29500 \
  main_hsdp.py \
  --backend nccl \
  --hsdp_shard_size 2 \
  --model_config configs/llama_60m.json \
  --batch_size 4 \
  --total_batch_size 16 \
  --save_dir /tmp/hsdp_nccl_ckpt \
  --num_training_steps 20 \
  --dtype bfloat16 \
  "$@"
EOF
chmod +x mpi_ulfm_extension/launch_hsdp_nccl.sh
```

- [ ] **Step 3: Create `launch_hsdp_ulfm.sh`**

```bash
cat > mpi_ulfm_extension/launch_hsdp_ulfm.sh <<'EOF'
#!/usr/bin/env bash
# HSDP ULFM: 2 replicas × 2 shards = 4 ranks, launched via mpirun for ULFM.
set -euo pipefail

cd "$(dirname "$0")"

MPI_HOME=${MPI_HOME:-/home/ziyueliu/openmpi-5.0.8-install}
export LD_LIBRARY_PATH="${MPI_HOME}/lib:/home/ziyueliu/project/pytorch/torch/lib:${LD_LIBRARY_PATH:-}"

"${MPI_HOME}/bin/mpirun" -np 4 \
  --with-ft ulfm \
  -x LD_LIBRARY_PATH \
  -x MASTER_ADDR=127.0.0.1 \
  -x MASTER_PORT=29500 \
  python main_hsdp.py \
    --backend ulfm \
    --hsdp_shard_size 2 \
    --model_config configs/llama_60m.json \
    --batch_size 4 \
    --total_batch_size 16 \
    --save_dir /tmp/hsdp_ulfm_ckpt \
    --num_training_steps 20 \
    --dtype bfloat16 \
    "$@"
EOF
chmod +x mpi_ulfm_extension/launch_hsdp_ulfm.sh
```

- [ ] **Step 4: Commit**

```bash
git add mpi_ulfm_extension/launch_hsdp_nccl.sh mpi_ulfm_extension/launch_hsdp_ulfm.sh
git commit -m "Add launch scripts for HSDP NCCL baseline and ULFM variants"
```

---

## Task 10: Smoke test — NCCL baseline

**Rationale:** Confirm FSDP HYBRID_SHARD + NCCL cross-replica is functional end-to-end before introducing ULFM. This is a manual integration test (can't unit-test distributed code cleanly).

**Files:** none (runtime test only)

- [ ] **Step 1: Run 20-step NCCL baseline**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
./launch_hsdp_nccl.sh 2>&1 | tee /tmp/hsdp_nccl_smoke.log
```

Expected:
- No MPI / ULFM errors in stderr.
- `HSDP layout: num_replicas=2, shard_size=2, my replica=..., my shard_rank=...` on each rank.
- `Model wrapped with FSDP HYBRID_SHARD` on each rank.
- 20 `Update step N/20, global step M, loss: ...` lines on rank 0.
- Exits `Finished successfully`.

- [ ] **Step 2: Fix any failures before proceeding**

If the run fails, debug at the point of failure (group construction, FSDP wrap, dataloader, or loss computation) before moving to Task 11.

- [ ] **Step 3: Record baseline final loss**

```bash
grep "Update step 20" /tmp/hsdp_nccl_smoke.log | tail -1
```

Keep this number for comparison in Task 11.

- [ ] **Step 4: Commit log (optional)**

```bash
# Only if you want the log in the repo for later reference
git add mpi_ulfm_extension/ -A  # will be no-op if nothing changed
# (This task creates no code; no commit is required.)
```

---

## Task 11: Smoke test — ULFM, no failures

**Rationale:** Confirm the ULFM hook fires, MPI allreduce on `replicate_pg` works, and loss matches the NCCL baseline within bf16 noise.

**Files:** none (runtime test only)

- [ ] **Step 1: Run 20-step ULFM smoke**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
./launch_hsdp_ulfm.sh 2>&1 | tee /tmp/hsdp_ulfm_smoke.log
```

Expected:
- `ULFM collectives extension ... initialized` or similar ULFM import success.
- `HSDPULFMTrainingManager initialized: policy=static, grad_accum=...` on each rank.
- `HSDP ULFM hook registered on replicate_pg` on each rank.
- Per-step `HSDP hook entered for unit N` debug lines (if DEBUG logging on).
- 20 update steps; loss trajectory numerically close to the NCCL baseline from Task 10 (bf16 reduction tolerance; expect a few ULP of difference per step, cumulative drift acceptable).

- [ ] **Step 2: Compare loss trajectories**

```bash
paste <(grep "Update step" /tmp/hsdp_nccl_smoke.log | awk '{print $NF}') \
      <(grep "Update step" /tmp/hsdp_ulfm_smoke.log | awk '{print $NF}')
```

Expected: adjacent columns within ~1e-2 relative difference per step, trending together.

- [ ] **Step 3: Fix any divergence before Task 12**

If the ULFM variant diverges significantly from NCCL or crashes, suspect (in order): hook registration not firing, unit-index mismatch across ranks, bf16/fp32 round-trip in `ulfm_allreduce` configured differently from what the baseline does, grad normalization factor mismatch.

---

## Task 12: Failure-injection test

**Rationale:** Validate the full fault-tolerance loop end-to-end. Kill one rank mid-training, confirm the sibling in the same replica dies via NCCL watchdog, the surviving replica continues, and loss does not blow up.

**Files:** none (runtime test only)

- [ ] **Step 1: Enable failure injection for one rank**

Edit `mpi_ulfm_extension/failure_config.yaml` (or add a new one, e.g. `failure_config_hsdp.yaml`) to inject a single failure at step 10 on rank 2 (a rank in replica 1, *not* replica 0 which is excluded). Verify the file before running:

```bash
cat mpi_ulfm_extension/failure_config.yaml
```

In the ULFM-variant wiring (Task 8 Step 1), pass this config to `FailureSimulator(..., config_path=...)`. If you prefer not to edit config files, set `desired_failures=1` and `target_ranks={2: [10]}` directly in `main_hsdp.py` when `--failure_start_step` is used. Use whichever of the two paths matches project convention — inspect `example_ulfm_ddp.py` for the established idiom.

- [ ] **Step 2: Run with injection**

```bash
cd /home/ziyueliu/project/pytorch/mpi_ulfm_extension
./launch_hsdp_ulfm.sh --num_training_steps 30 --failure_start_step 5 2>&1 \
  | tee /tmp/hsdp_ulfm_fail.log
```

Expected:
- At the injected step: `Failure noticed via MPIX_Comm_agree` on surviving ranks.
- NCCL watchdog kills rank 3 (the shard sibling of rank 2) within its configured timeout.
- Policy/orchestrator log lines: restore mode, possible extra microbatch, commit continues.
- Training reaches step 30 on ranks 0 and 1 (surviving replica).
- Loss continues decreasing (or at least does not NaN/Inf).

- [ ] **Step 3: Both restore branches**

Run twice more: once with the failure aligned on the policy boundary (mid-window = non-blocking restore), once off-boundary (blocking restore). The exact `failure_start_step` values depend on `gradient_accumulation`; choose values that place the injected failure in each regime.

Expected: both branches complete successfully, producing different log signatures (`NON_BLOCKING` vs `BLOCKING`).

- [ ] **Step 4: No commit**

This task is runtime-only; nothing to commit unless you created a new `failure_config_hsdp.yaml`, in which case:

```bash
git add mpi_ulfm_extension/failure_config_hsdp.yaml
git commit -m "Add HSDP-specific failure injection config"
```

---

## Verification summary

After all 12 tasks:

- `pytest mpi_ulfm_extension/test/test_hsdp_groups.py -v` → 6 passing tests.
- `./launch_hsdp_nccl.sh` → 20 steps, clean exit.
- `./launch_hsdp_ulfm.sh` → 20 steps, clean exit, loss matches NCCL baseline.
- `./launch_hsdp_ulfm.sh` with failure injection → surviving replica continues training through injected failure, both restore branches exercised.

All done. Stage 2 (fp32 accumulator for both variants) is a separate plan.
