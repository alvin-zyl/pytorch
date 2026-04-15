# HSDP + ULFM Integration — Design (Stage 1: bf16 Standalone)

**Date:** 2026-04-15
**Status:** Approved — pending implementation plan
**Scope:** Stage 1. Standalone `main.py`-style binary integrating PyTorch FSDP1
`HYBRID_SHARD` with the existing ULFM fault-tolerance stack. Pure bf16, no fp32
gradient accumulator. Two variants in one binary: NCCL baseline and ULFM.

## Context

The existing ULFM integration (`mpi_ulfm_extension/main.py` and `nanotron/`
trainer) plugs fault tolerance into `torch.nn.parallel.DistributedDataParallel`
via `register_comm_hook`. We now extend the same pattern to HSDP so multiple
full-model replicas can train with fault-tolerant cross-replica gradient sync.

Decisions reached during brainstorming:

- **Failure model:** cross-replica only. Intra-replica rank death → NCCL
  watchdog kills the whole replica → ULFM on the cross-replica group detects
  one fewer replica and continues with survivors. No intra-replica re-sharding.
- **FSDP API:** FSDP1 `HYBRID_SHARD`. It exposes `register_comm_hook` on the
  cross-replica allreduce, so the ULFM hook pattern drops in. FSDP2 is
  rejected because it has no comm-hook escape hatch.
- **Framework:** standalone `main_hsdp.py`, not nanotron. Nanotron's fp32
  accumulator, 4D parallelism, and checkpoint glue would all need rewriting
  before any HSDP+ULFM correctness can be observed; standalone first gets the
  integration question answered in isolation.
- **Precision:** pure bf16 now. MPI cannot allreduce bf16 directly, so the
  C++ `ulfm_allreduce` upcasts/downcasts on the wire. No Python-side fp32
  buffer in stage 1. Stage 2 (separate design) will add an fp32 accumulator
  that works for both the NCCL baseline and the ULFM variant.
- **Shard layout:** contiguous ranks per replica. Rank `i` belongs to replica
  `i // shard_size`, intra-replica rank `i % shard_size`. Keeps intra-replica
  collectives on-node over NVLink.

## Architecture

### Binary layout

One file `mpi_ulfm_extension/main_hsdp.py`, forked from `main.py`, with
flags:

- `--backend {nccl,ulfm}` — selects process-group backend and hook.
- `--hsdp_shard_size N` — ranks per replica (default: `torch.cuda.device_count()`).
  Replica count = `world_size / N`.

Training loop, dataloader, optimizer, wandb, eval, and checkpoint code are
shared across variants. Only three points diverge: process-group init, FSDP
construction, and hook registration. A single binary keeps the two variants
from drifting apart silently.

### Process groups

- Global init: `dist.init_process_group(backend="ulfm")` if `--backend ulfm`,
  else `"nccl"`. World PG type matches existing `main.py`.
- Two subgroups built via `dist.new_group(...)` (not `init_device_mesh` —
  ULFM backend is not wired for device-mesh routing):
  - **`shard_pg`** (intra-replica): NCCL, ranks `[r*S .. (r+1)*S - 1]` for
    each replica `r`. FSDP uses this for all-gather (fwd) and reduce-scatter
    (bwd). No fault tolerance on this group.
  - **`replicate_pg`** (cross-replica): NCCL in baseline, ULFM in ULFM
    variant. Ranks `[s, s+S, s+2S, ...]` for each intra-replica offset `s`.
    This is where the ULFM hook intercepts the cross-replica allreduce.

### FSDP wrapping

```python
model = FullyShardedDataParallel(
    hf_model,
    sharding_strategy=ShardingStrategy.HYBRID_SHARD,
    process_group=(shard_pg, replicate_pg),   # tuple API, mixed backends OK
    auto_wrap_policy=transformer_auto_wrap_policy(
        transformer_layer_cls={LlamaDecoderLayer},
    ),
    device_id=local_rank,
    use_orig_params=True,
    mixed_precision=None,   # model already cast to bf16
)
```

- `use_orig_params=True` preserves `named_parameters` structure. Matters for
  Stage 2 (fp32 accumulator needs per-param handles).
- `auto_wrap_policy`: wrap each transformer block as one FSDP unit — standard
  Llama/HF pattern. More units = more comm/compute overlap, at the cost of
  more hook firings per step.
- `mixed_precision=None` + bf16-cast model → params/grads stay bf16 throughout
  FSDP. Intra-replica reduce-scatter is bf16 over NCCL (supported). The
  cross-replica allreduce in the ULFM variant upcasts to fp32 inside the C++
  `ulfm_allreduce` because MPI does not support bf16.
- Activation checkpointing remains controlled by the existing flag and
  composes with FSDP via `checkpoint_wrapper`.

### Comm hook (ULFM variant)

FSDP1 HYBRID_SHARD fires `register_comm_hook` after the intra-replica
reduce-scatter. The hook input is a 1-D bf16 shard-grad tensor; the hook's
responsibility is the cross-replica allreduce.

New factory in `mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py`:

```python
def create_ulfm_hsdp_hook(ulfm_opts):
    def hook(state, grad_shard):
        pg = state.pg                 # replicate_pg (ULFM)
        orch = state.orchestrator
        unit_index = state.next_unit_index()

        if getattr(pg, "is_quiesced", lambda: False)():
            fut = torch.futures.Future()
            fut.set_result(grad_shard)
            return fut

        torch.cuda.synchronize()      # same rationale as DDP hook
        orch.on_bucket_snapshot(grad_shard, unit_index, pg)

        opts = torch.distributed.AllreduceOptions()
        opts.reduceOp = torch.distributed.ReduceOp.SUM
        work = pg.ulfm_allreduce([grad_shard], opts, ulfm_opts)

        def on_done(fut):
            _sim = get_failure_simulator()
            ctx = _sim.may_fail_here("post-allreduce") if _sim else contextlib.nullcontext()
            with ctx:
                orch.handle_work_completion(work=work, bucket_index=unit_index)
            orch.increment_hook_counter()
            return fut.value()[0]

        return work.get_future().then(on_done)
    return hook
```

- Immediate (not deferred): no pipeline to coordinate with.
- Unit indices assigned in registration order via `state.next_unit_index()`.
  FSDP unit order is deterministic across ranks.
- Up/down-casting happens inside C++ `ulfm_allreduce`; Python-side is pure bf16.
- Restore path overwrites bf16 shard grad in place with bf16 snapshot — no
  fp32 scratch.

### Training manager

New class `HSDPULFMTrainingManager(ULFMTrainingManager)` in
`mpi_ulfm_extension/ulfm_collectives/hsdp_training_manager.py`.

Overrides:

- Constructor takes `fsdp_model` (already wrapped) and `replicate_pg`. Skips
  the internal `DistributedDataParallel(model)` wrap. Orchestrator's PG is
  `replicate_pg`, not world.
- `_register_ulfm_hook(...)` registers `create_ulfm_hsdp_hook` on the FSDP
  model (FSDP hook signature: `(state, grad_tensor)`, not DDP's
  `(state, GradBucket)`).

Reused verbatim from the parent:

- `train_step(...)` microbatch state machine, `no_sync()` context, restore
  modes, optimizer commit, policy bookkeeping.
- Orchestrator, policy, failure simulator, C++ extension — no changes.

FSDP's `no_sync()` skips the *cross-replica* allreduce while still running
intra-replica reduce-scatter per microbatch. This is the behavior we need.

`p.grad.div_(...)` in the normalize step works on FSDP shard grads because
`use_orig_params=True` exposes them via `.parameters()`.

### Baseline (NCCL variant)

No training manager, no hook, no failure simulator. FSDP's default HYBRID_SHARD
cross-replica allreduce runs, standard optimizer step. Serves as the reference
for correctness comparison.

### Failure simulator

Two adjustments vs existing `main.py`:

- Exclude **all ranks in replica 0** (the log/wandb replica), not just rank 0.
  Killing any rank in replica 0 cascades via NCCL watchdog and loses the
  wandb writer.
- Failure granularity is implicitly per-replica because the orchestrator's
  rank tracking lives on `replicate_pg`. No explicit changes — falls out of
  the PG choice.

## Testing

1. 1 replica × 2 shards, NCCL — FSDP smoke.
2. 2 replicas × 2 shards, NCCL — cross-replica correctness; loss matches DDP
   at matched GBS.
3. 2 replicas × 2 shards, ULFM, no failures — hook fires, loss matches NCCL.
4. 2 replicas × 2 shards, ULFM, kill one rank mid-training — watchdog
   cascades, one replica lost, training continues with survivors; loss does
   not diverge.
5. Failure at policy boundary vs mid-window — exercises both `BLOCKING` and
   `NON_BLOCKING` restore branches.
6. NCCL baseline vs ULFM-no-failure at matched GBS — loss curves overlap
   within noise.

## Files

New:

- `mpi_ulfm_extension/main_hsdp.py` — standalone training binary with
  `--backend`, `--hsdp_shard_size`.
- `mpi_ulfm_extension/ulfm_collectives/hsdp_training_manager.py` —
  `HSDPULFMTrainingManager(ULFMTrainingManager)`.

Modified:

- `mpi_ulfm_extension/ulfm_collectives/ulfm_hook.py` — add
  `create_ulfm_hsdp_hook(ulfm_opts)` factory.
- `mpi_ulfm_extension/ulfm_collectives/__init__.py` — export new symbols.

Unchanged: orchestrator, policy, failure simulator, C++ extension.

## Out of Scope (Stage 2+)

- FP32 gradient accumulator for HSDP. Must work for both NCCL baseline and
  ULFM variant. Deferred to its own design cycle.
- Porting HSDP+ULFM into nanotron (requires 4D parallelism context, fp32
  accumulator rewrite for sharded grads, checkpoint glue).
- Intra-replica fault tolerance (re-sharding across surviving shard ranks).
