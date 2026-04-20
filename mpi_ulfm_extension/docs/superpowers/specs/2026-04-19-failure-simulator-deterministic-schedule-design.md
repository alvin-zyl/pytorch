# Deterministic Replica-Aware Failure Schedule — Design

**Date:** 2026-04-19
**Scope:** `mpi_ulfm_extension/ulfm_collectives/failure_simulator.py` and new sibling modules under `mpi_ulfm_extension/ulfm_collectives/failure/`.
**Status:** Draft for review.

## Motivation

The current `FailureSimulator` is a per-rank stochastic injector. Each rank independently rolls dice every minibatch to decide whether to kill itself at a randomly-selected code location. This is hard to use in practice for 3D parallelism and HSDP, because:

- Failures land on individual ranks, not replicas, so a scheduled kill can hit any rank regardless of which replica it belongs to.
- Each rank's decision is independent, so the "plan" for the run is only visible in aggregate after the fact.
- There is no way to know in advance what the run will look like; reproducibility relies on multi-rank seeds lining up.

This design replaces the stochastic per-rank simulator with a **deterministic, replica-aware failure schedule** that is either generated ahead of time by a standalone tool or generated once at training init (identically on every rank).

## Goals

1. **Schedule-first.** Every kill is a scheduled entry `(step, replica_id, local_rank, location)` known before training begins.
2. **Replica-aware.** Generation operates at replica granularity: pick which replicas fail, then randomly pick one rank within each chosen replica. At most one kill per replica across the whole run.
3. **Identical across the world.** Every rank's simulator sees the same schedule — whether generated at init or loaded from file.
4. **Standalone generation.** A CLI tool can produce a schedule from a parallelism spec alone, without launching training, and save it to YAML. The standalone output is bit-identical to what the in-process generator produces given the same inputs.
5. **Layout-aware.** Generator encodes the same rank-to-replica mapping used by `nanotron/src/nanotron/parallel/context_ulfm.py` (3D) and `main_hsdp.py` (HSDP), so callers only pass parallelism dims.

## Non-Goals

- No changes to ULFM recovery semantics, `ULFMReducer`, or `ProcessGroupULFM`.
- No migration of the old `failure_config.yaml` weighted-locations format.
- No support for heterogeneous replicas (replicas of differing sizes).
- No changes to the `may_fail` / `may_fail_here` decorator/context-manager public surface. (`begin_minibatch` loses its internal `minibatch == 0` skip and its stochastic decision path — that is a simulator-internal change, not a change to the decorator API.)

## Architecture

### Module layout

```
mpi_ulfm_extension/
  ulfm_collectives/
    failure/
      __init__.py
      schedule.py        # FailureSchedule, FailureEntry, ParallelismSpec, GeneratorConfig; YAML load/save
      topology.py        # replicas_3d(tp,pp,dp,ep), replicas_hsdp(world,shard_size) -> List[List[int]]
      generator.py       # generate(spec, seed, num_failures, step_range, location_weights, sampling) -> FailureSchedule
    failure_simulator.py # Existing file; internals refactored to consume FailureSchedule. Public API preserved.
  scripts/
    generate_failure_schedule.py  # CLI wrapper over generator. No MPI, no torch, no CUDA.
```

### Data flow

Three entry paths, one runtime contract:

```
[offline]   CLI args ─────────────> generator.generate() ──> FailureSchedule ──> yaml.dump() ──> schedule.yaml
[training/load]   --failure-schedule path.yaml ─> schedule.load() ─> FailureSchedule ─┐
[training/inline] --failure-num N --failure-seed S ... ─> generator.generate() ───────┤
                                                                                      └──> FailureSimulator
```

The simulator has no knowledge of which path produced the schedule.

### Cross-rank determinism

Every rank independently calls either `FailureSchedule.load()` or `generator.generate(seed=...)` with identical inputs. Same inputs → same output. No broadcast.

The generator is a **pure function of its inputs** (`seed`, `parallelism_spec`, `num_failures`, `step_range`, `location_weights`, `sampling`). It does not read rank, world size, environment variables, wall clock, or filesystem state. It does not import torch or MPI. Running it with `-np 1` as a standalone script produces byte-identical output to calling it from rank 42 of 64 during training init.

### Topology assertion

When a schedule is loaded from file, its embedded `parallelism` block is compared against the runtime's actual topology. Mismatch → fail fast at `FailureSimulator.initialize()` with a message naming both the expected and actual world size. Prevents running a 4-replica schedule on an 8-replica topology by accident.

## Data Model

```python
# ulfm_collectives/failure/schedule.py

@dataclass(frozen=True)
class FailureEntry:
    step: int              # minibatch index (micro-step)
    replica_id: int        # index into replicas list
    local_rank: int        # index within that replica's rank list
    location: str          # e.g. "post-allreduce", "backward"

@dataclass(frozen=True)
class ParallelismSpec:
    kind: Literal["3d", "hsdp"]
    # 3d fields (None for hsdp)
    tp: Optional[int] = None
    pp: Optional[int] = None
    dp: Optional[int] = None
    ep: Optional[int] = None
    # hsdp fields (None for 3d)
    world_size: Optional[int] = None
    shard_size: Optional[int] = None

@dataclass(frozen=True)
class GeneratorConfig:
    seed: int
    num_failures: int
    step_range: Tuple[int, int]               # [start, end) — inclusive start, exclusive end
    sampling: Literal["iid", "stratified"]    # default: "stratified"
    location_weights: Dict[str, float]        # weighted set; values must be > 0

@dataclass(frozen=True)
class FailureSchedule:
    parallelism: ParallelismSpec
    generator_config: Optional[GeneratorConfig]  # None for hand-written schedules
    entries: Tuple[FailureEntry, ...]            # sorted by (step, replica_id)

    @classmethod
    def load(cls, path: str) -> "FailureSchedule": ...
    def save(self, path: str) -> None: ...
    def assert_matches_topology(self, spec: ParallelismSpec) -> None: ...
    def entries_for_rank(self, global_rank: int) -> List[FailureEntry]: ...
```

### On-disk YAML

```yaml
parallelism:
  kind: 3d
  tp: 2
  pp: 2
  dp: 4
  ep: 1
generator_config:
  seed: 42
  num_failures: 3
  step_range: [10, 200]
  sampling: stratified
  location_weights:
    post-allreduce: 0.6
    backward: 0.4
entries:
  - {step: 32,  replica_id: 1, local_rank: 3, location: post-allreduce}
  - {step: 80,  replica_id: 0, local_rank: 0, location: backward}
  - {step: 155, replica_id: 3, local_rank: 7, location: post-allreduce}
```

Hand-written schedules may omit the `generator_config` block. `FailureSchedule.load` accepts either.

### Invariants enforced on load

- `entries` sorted by `(step, replica_id)` (sort is applied, not just validated).
- At most one entry per `replica_id` across the list.
- `replica_id` within `[0, num_replicas)` and `local_rank` within `[0, replica_size)`, derived from `parallelism`.
- `location` is a non-empty string.

## Topology Functions

Pure functions in `ulfm_collectives/failure/topology.py`. No torch, no MPI — numpy + stdlib only.

```python
def replicas_3d(tp: int, pp: int, dp: int, ep: int) -> List[List[int]]:
    """
    Mirrors nanotron/src/nanotron/parallel/context_ulfm.py:
        ranks = arange(world).reshape((ep, pp, dp, tp))
    A replica = one DP slice. Inner list is ordered by (ep, pp, tp) so local_rank
    is stable across runs.
    """
    world = tp * pp * dp * ep
    ranks = np.arange(world).reshape((ep, pp, dp, tp))
    return [ranks[:, :, dp_idx, :].reshape(-1).tolist() for dp_idx in range(dp)]

def replicas_hsdp(world_size: int, shard_size: int) -> List[List[int]]:
    """
    Mirrors main_hsdp.py: contiguous chunks of shard_size ranks form one shard
    (one replica in the sense of failure-unit; shard_pg).
    """
    assert world_size % shard_size == 0
    num_replicas = world_size // shard_size
    return [list(range(r * shard_size, (r + 1) * shard_size)) for r in range(num_replicas)]
```

Unit tests assert the output layouts match the mapping actually used by `context_ulfm.py` and `main_hsdp.py`.

## Generator

```python
# ulfm_collectives/failure/generator.py

def generate(
    parallelism: ParallelismSpec,
    seed: int,
    num_failures: int,
    step_range: Tuple[int, int],
    location_weights: Dict[str, float],
    sampling: Literal["iid", "stratified"] = "stratified",
) -> FailureSchedule:
```

### Algorithm

1. Compute `replicas = replicas_3d(...)` or `replicas_hsdp(...)` from `parallelism`.
2. Validate (`num_failures ≤ len(replicas)`, `num_failures == 0` short-circuits to an empty schedule, non-empty `location_weights`, all weights > 0, `step_range[1] > step_range[0]`; in stratified mode additionally require `(end - start) ≥ num_failures`).
3. Construct a single `random.Random(seed)` instance. All random decisions come from this RNG in a canonical call order.
4. Sample `num_failures` distinct replica ids via `rng.sample(range(len(replicas)), num_failures)`. Keep the sample order (do not re-sort) so that the mapping from bucket index to replica is shuffled, not monotonic in `replica_id`. `random.sample` on a plain `range` is stable across supported Python versions (3.9+), so no extra sort is required for determinism.
5. For each chosen replica, in sample order, draw:
   - **step**:
     - `sampling == "iid"`: `rng.randrange(start, end)`. Collisions allowed (different replicas can share the same step).
     - `sampling == "stratified"`: divide `[start, end)` into `num_failures` buckets of equal width (integer division; remainder distributed to the last bucket). Draw one step per bucket with `rng.randrange(bucket_start, bucket_end)`.
   - **local_rank**: `rng.randrange(0, len(replica))`.
   - **location**: weighted choice over the *sorted* list of `(name, weight)` pairs from `location_weights`, using `rng.choices` with normalized weights. Sorting the keys prevents dict iteration order from influencing the result.
6. Sort final entries by `(step, replica_id)` and return the `FailureSchedule` with a populated `GeneratorConfig`.

### Determinism guarantees

- The RNG is seeded exactly once and consumed in a single, fixed order.
- All iteration over `location_weights` and the chosen replica ids is over sorted keys / sorted indices.
- Output YAML is sorted by `(step, replica_id)` so serialized content depends only on the `FailureSchedule` contents.

## Simulator Refactor

`ulfm_collectives/failure_simulator.py` keeps its public API. Internal state is gutted and replaced with a schedule consumer.

### Constructor

```python
FailureSimulator(schedule: FailureSchedule, enabled: bool = True)
```

Removed: `seed`, `desired_failures`, `total_minibatches`, `target_ranks`, `excluded_ranks`, `start_minibatch`, `config_path`. All of these are either inputs to the generator or properties of the schedule.

### `initialize(rank, world_size)`

1. Derive the expected world size from `schedule.parallelism`; raise `RuntimeError` if it does not match `world_size`.
2. Compute the replicas list via the appropriate topology function.
3. Resolve this rank's entries: for each entry, convert `(replica_id, local_rank)` → `global_rank = replicas[replica_id][local_rank]`; keep entries where `global_rank == rank`.
4. Cache `self._my_entries: Dict[int, FailureEntry]` keyed by `step`.
5. Rank 0 logs the full schedule at INFO. Every rank logs its own assigned entries (or "no failures assigned").
6. Scan `schedule` for `location` values that are not in `_registered_locations` at the end of `initialize`, and log a `WARNING` listing the unreached entries. Decorators register at import/definition time so their locations are already present at init; `may_fail_here` context managers register on first `__enter__`, which typically happens inside the training loop, so this scan is best-effort and may over-warn for locations that only exist inside the step loop. The run continues either way.

### `begin_minibatch(step)`

1. If `step` not in `self._my_entries`, return.
2. Otherwise arm the entry: set `self._state.target_location = entry.location`.

No stochastic sampling. No minibatch-0 skip.

### `check(location)`

Unchanged behavior: if the armed entry's location matches, SIGKILL; otherwise return False.

### `may_fail` / `may_fail_here`

Unchanged — still register locations, still delegate to `check`.

### `reset()`

Re-runs the rank resolution step (same schedule, same assignments).

### `get_stats()`

Returns the schedule summary + this rank's assigned entries + history of injected failures. The `failure_probability` field and the `compute_probability` static method are removed.

## CLI Surfaces

### Standalone generator

```
mpi_ulfm_extension/scripts/generate_failure_schedule.py
    --kind {3d,hsdp}
    (3d) --tp N --pp N --dp N --ep N
    (hsdp) --world-size N --shard-size N
    --num-failures N
    --step-range START END      # [START, END)
    --seed N                    # default 0
    --sampling {iid,stratified} # default stratified
    --locations name:weight [name:weight ...]
    [--output PATH | --print | --dry-run]
```

`--print` dumps YAML to stdout. `--dry-run` prints a human-readable table (step / replica / rank / location) without writing. Single process. No MPI. No CUDA. No torch.

Example:

```bash
python scripts/generate_failure_schedule.py \
    --kind 3d --tp 2 --pp 2 --dp 4 --ep 1 \
    --num-failures 3 --step-range 10 200 \
    --seed 42 \
    --locations post-allreduce:0.6 backward:0.4 \
    --output schedules/3d_demo.yaml
```

### Training script integration

Both paths exposed, mutually exclusive. If neither is given → no failures (equivalent to today's `desired_failures=0`).

```
# Load a pre-generated schedule
--failure-schedule PATH

# Inline generation
--failure-num N
--failure-seed N
--failure-step-range START END
--failure-sampling {iid,stratified}
--failure-locations name:weight [name:weight ...]
```

Parallelism is derived from the training script's own argument space (`--hsdp_shard_size` in `main_hsdp.py`, nanotron's config for 3D), not repeated on the failure flags. The training script constructs a `ParallelismSpec` from its known topology, then:

- **Load path:** `sched = FailureSchedule.load(path); sched.assert_matches_topology(spec)`.
- **Inline path:** `sched = generator.generate(spec, seed=..., num_failures=..., step_range=..., location_weights=..., sampling=...)`.

Either way: `sim = FailureSimulator(schedule=sched); sim.initialize(rank=global_rank, world_size=world_size)`.

### Backwards compatibility

- Today's `failure_config.yaml` (location-weights YAML for the stochastic simulator) is no longer consumed by code. The file remains in the repo as an example of location names only.
- The `--failure_start_step` / `start_minibatch` arg becomes the low end of `--failure-step-range`.
- The `FailureSimulator` public class, its decorator, its context manager, its singleton accessors, and the `begin_minibatch`/`check` methods keep their names and signatures. Call sites in `main.py`, `main_hsdp.py`, and `nanotron/trainer_ulfm.py` need updates only for the constructor and the new CLI flags.

## Error Handling

All failure modes are checked explicitly and produce a specific `ValueError` or `RuntimeError` with the offending input named.

**Generator:**
- `num_failures > len(replicas)` → `ValueError`.
- `num_failures == 0` → empty schedule, no error.
- Stratified with `(end - start) < num_failures` → `ValueError`.
- Empty / zero / negative `location_weights` → `ValueError`.
- Invalid `ParallelismSpec` (`kind=3d` missing a dim; `kind=hsdp` with `world_size % shard_size != 0`) → `ValueError`.

**Schedule load:**
- YAML parse errors propagate.
- Missing required top-level keys → `ValueError` naming the missing key.
- `replica_id` or `local_rank` out of bounds → `ValueError`.
- More than one entry per `replica_id` → `ValueError` (violates one-kill-per-replica rule).

**Simulator init:**
- World-size mismatch → `RuntimeError` naming both the schedule's expected world size and the runtime's actual world size.
- Unregistered scheduled locations → `WARNING` log at end of `initialize`, not a failure; best-effort (over-warns for `may_fail_here` locations that register only on first loop iteration).

**Runtime (`begin_minibatch`, `check`):** pure lookups; no new error paths.

## Testing

New test directory `mpi_ulfm_extension/test/failure/`:

1. **`test_topology.py`** — `replicas_3d` and `replicas_hsdp` with small known inputs. Includes a consistency test that constructs `ranks[ep,pp,dp,tp]` the way `context_ulfm.py` does and asserts the replica list matches the topology function's output.

2. **`test_generator.py`** — pure-function tests:
   - Determinism: same inputs → byte-identical YAML across N independent calls.
   - Invariants: exactly N entries, distinct `replica_id`s, `step` in `[start, end)`.
   - Stratified: bucket coverage assertion (one entry per bucket).
   - i.i.d.: no bucket guarantee; still N entries, `step` in range.
   - All documented error cases raise.
   - Location weights are honored statistically over 1000 regenerations with different seeds.

3. **`test_schedule.py`** — round-trip `save → load` preserves all fields; hand-written YAML without `generator_config` loads fine; `assert_matches_topology` fails on mismatched spec; invariant violations are caught.

4. **`test_simulator.py`** — unit tests against a fake schedule, `os.kill` patched:
   - Rank with no assigned entries: `begin_minibatch` is a no-op, `check` always returns False.
   - Rank with an entry at step 5, location `X`: before step 5 → no-op; at step 5, `check("Y")` → False, `check("X")` → would SIGKILL (intercepted).
   - Topology mismatch raises in `initialize`.
   - Unregistered-location warning fires.

5. **`test_standalone_vs_inline.py`** — drive the generator in a subprocess (mimicking the standalone script) with a fixed arg set, drive the in-process `generator.generate` with the identical arg set, assert byte-equal YAML. Anchors the "standalone equals runtime" requirement.

6. **Manual integration** (not automated, requires ULFM-enabled `mpirun`): a small 4-rank HSDP run and an 8-rank 3D run consume a known schedule and confirm the expected rank dies at the expected step.
