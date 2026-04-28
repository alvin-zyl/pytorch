import os
import sys
import time
import json
import random
import argparse
import functools
import numpy as np

import torch
import torch.nn as nn
import torch.utils.data
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel, MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
import torch.distributed.checkpoint as dcp
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

import transformers
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM
from transformers import LlamaForCausalLM
from transformers import default_data_collator
import datasets
import datasets.distributed
import wandb
from loguru import logger
from pretraining_utils import training_utils, args_utils
from pretraining_utils.dataloader import PreprocessedIterableDataset
import datetime, pdb, pickle
from torch.profiler import profile, ProfilerActivity

try:
    import ulfm_collectives as ULFM
    from ulfm_collectives.training_manager import ULFMTrainingManager
    from ulfm_collectives.failure_simulator import FailureSimulator, set_failure_simulator
    from ulfm_collectives.failure import (
        FailureSchedule,
        ParallelismSpec,
        generate as generate_failure_schedule,
    )
    from ulfm_collectives.hsdp_groups import (
        compute_hsdp_layout,
        replica_ranks,
        replica0_ranks,
        replicate_peer_ranks,
    )
    from ulfm_collectives.hsdp_training_manager import HSDPULFMTrainingManager
    from ulfm_collectives.policy import GradRestoreMode
    _ULFM_AVAILABLE = True
except ImportError:
    _ULFM_AVAILABLE = False

transformers.logging.set_verbosity_error()


torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_flash_sdp(False)


class LMWrapper(nn.Module):
    """Wraps a causal LM to accept a single batch dict, as required by ULFMTrainingManager."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, batch):
        return self.model(**batch)


def get_rank():
    if not dist.is_available():
        return 0
    if not dist.is_initialized():
        return 0
    return dist.get_rank()


def is_main_process():
    return get_rank() == 0


def parse_args(args):
    parser = argparse.ArgumentParser()

    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--wandb_project", type=str, default="ulfm")
    parser.add_argument("--model_config", type=str, required=True)
    parser.add_argument("--offline_mode", default=False, action="store_true")
    parser.add_argument("--continue_from", type=str, default=None)
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--gradient_accumulation", type=int, default=1)
    parser.add_argument("--total_batch_size", type=int, default=None)
    parser.add_argument("--max_length", type=int, default=256)
    parser.add_argument("--optimizer", default="adamw")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["linear", "cosine", "cosine_restarts"],
    )
    parser.add_argument("--min_lr_ratio", type=float, default=0.1)
    parser.add_argument("--activation_checkpointing", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--warmup_steps", type=int, default=1_000)
    parser.add_argument("--eval_every", type=int, default=5_000)
    parser.add_argument(
        "--num_training_steps",
        type=int,
        default=10_000,
        help="Number of **update steps** to train for. "
        "Notice that gradient accumulation is taken into account.",
    )
    parser.add_argument(
        "--max_train_tokens",
        type=training_utils.max_train_tokens_to_number,
        default=None,
        help="Number of tokens to train on. Overwrites num_training_steps. "
        "You can use M and B suffixes, e.g. 100M or 1B.",
    )
    parser.add_argument("--save_every", type=int, default=10_000)
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--tags", type=str, default=None)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16" if torch.cuda.is_bf16_supported() else "float32",
    )
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="test")
    parser.add_argument("--grad_clipping", type=float, default=0.0)
    # beta1 for adafactor
    parser.add_argument("--beta1", type=float, default=0.0)
    # disable ddp, single_gpu
    parser.add_argument("--single_gpu", default=False, action="store_true")
    parser.add_argument(
        "--backend",
        type=str,
        default="ulfm",
        choices=["nccl", "ulfm"],
        help="Cross-replica backend: 'nccl' (baseline) or 'ulfm' (fault-tolerant).",
    )
    parser.add_argument(
        "--mixed_precision",
        default=False,
        action="store_true",
        help="Enable FSDP native mixed precision: bf16 compute, fp32 reduce, "
             "fp32 master weights. Requires the model to remain in fp32; "
             "when set, the --dtype bf16 cast below is skipped.",
    )
    parser.add_argument(
        "--hsdp_shard_size",
        type=int,
        default=None,
        help="Ranks per FSDP replica. Default: torch.cuda.device_count(). "
             "world_size must be divisible by this value.",
    )
    # --------------------------------------------------------------
    # Failure schedule (deterministic replica-aware).
    # Either load a pre-generated YAML via --failure_schedule, or generate
    # inline with --failure_count + --failure_step_range + --failure_locations.
    # The two are mutually exclusive. If neither is given, no failures are injected.
    # --------------------------------------------------------------
    parser.add_argument(
        "--failure_schedule",
        type=str,
        default=None,
        help="Path to a YAML failure schedule (see scripts/generate_failure_schedule.py).",
    )
    parser.add_argument(
        "--failure_count",
        type=int,
        default=0,
        help="Inline generator: number of replicas to kill. 0 disables injection.",
    )
    parser.add_argument(
        "--failure_seed",
        type=int,
        default=0,
        help="Inline generator seed. All ranks must pass the same value.",
    )
    parser.add_argument(
        "--failure_step_range",
        type=int,
        nargs=2,
        default=None,
        metavar=("START", "END"),
        help="Inline generator: half-open [START, END) minibatch index range for kills.",
    )
    parser.add_argument(
        "--failure_sampling",
        choices=("iid", "stratified"),
        default="stratified",
        help="Inline generator: step-sampling mode.",
    )
    parser.add_argument(
        "--failure_locations",
        nargs="+",
        default=None,
        metavar="NAME:WEIGHT",
        help="Inline generator: location:weight pairs (e.g. post-allreduce:0.6 backward:0.4).",
    )
    parser.add_argument(
        "--ulfm_verbose",
        default=False,
        action="store_true",
        help="Enable verbose ULFM logging: turns on ProcessGroupULFM C++ "
             "verbose mode (set_ulfm_verbose_logging(True)) and bumps the "
             "Python 'ulfm_collectives' logger to DEBUG. Mirrors nanotron's "
             "--ulfm-verbose flag.",
    )
    parser.add_argument(
        "--failure_exclude_replicas",
        type=str,
        default="0",
        help="Comma-separated replica ids to spare from selection "
             "(default '0' spares the replica that contains global rank 0, "
             "which hosts wandb). Pass empty string to allow all replicas.",
    )

    args = parser.parse_args(args)

    args = args_utils.check_args_torchrun_main(args)
    return args


@torch.no_grad()
def evaluate_model(
    model,
    preprocess_batched,
    pad_idx,
    global_rank,
    world_size,
    device,
    batch_size,
    dataloader=None,
):
    _time = time.time()
    if dataloader is None:
        val_data = datasets.load_dataset(
            "allenai/c4", "en", split="validation", streaming=True
        )  # DGX
        val_data = val_data.shuffle(seed=42)
        if is_main_process():
            logger.info(
                f"Loaded validation dataset in {time.time() - _time:.2f} seconds"
            )

        if not args.single_gpu:
            val_data = datasets.distributed.split_dataset_by_node(
                val_data, rank=global_rank, world_size=world_size
            )

        val_data_mapped = val_data.map(
            preprocess_batched,
            batched=True,
            remove_columns=["text", "timestamp", "url"],
        )
        val_data_mapped.batch = lambda batch_size: training_utils.batch_fn(
            val_data_mapped, batch_size
        )

    target_eval_tokens = 10_000_000
    evaluated_on_tokens = 0
    total_loss = torch.tensor(0.0).to(device)
    total_batches = 1
    if is_main_process():
        logger.info(f"Eval set prepared in {time.time() - _time:.2f} seconds")

    for batch in (
        val_data_mapped.batch(batch_size=batch_size)
        if dataloader is None
        else dataloader
    ):
        if evaluated_on_tokens > target_eval_tokens:
            break
        total_batches += 1

        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100

        loss = model(**batch).loss
        total_loss += loss.detach()

        evaluated_on_tokens += (batch["input_ids"] != pad_idx).sum().item() * world_size

    total_loss = total_loss / total_batches

    # Gather losses across all GPUs
    gathered_losses = [torch.zeros_like(total_loss) for _ in range(world_size)]
    dist.all_gather(gathered_losses, total_loss)
    total_loss = sum([t.item() for t in gathered_losses]) / world_size

    return total_loss, evaluated_on_tokens

def build_hsdp_groups(world_size: int, shard_size: int, backend: str):
    """Build (shard_pg, replicate_pg, layout) for the current rank.

    shard_pg is always NCCL. replicate_pg matches `backend`
    ('nccl' or 'ulfm'). Every group is created on every rank (required by
    PyTorch dist.new_group), but each rank only belongs to one of each kind.
    """
    layout = compute_hsdp_layout(world_size=world_size, shard_size=shard_size)
    my_rank = dist.get_rank()

    _timeout_sec = os.getenv("NCCL_PG_TIMEOUT_SECONDS")
    _nccl_timeout = (
        datetime.timedelta(seconds=float(_timeout_sec))
        if _timeout_sec is not None
        else None
    )

    # --- Shard groups: one per replica ---
    shard_pg = None
    for rid in range(layout.num_replicas):
        ranks = replica_ranks(rid, shard_size)
        pg = dist.new_group(ranks=ranks, backend="nccl", timeout=_nccl_timeout)
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
            pg = dist.new_group(ranks=ranks, backend="nccl", timeout=_nccl_timeout)
        if my_rank in ranks:
            replicate_pg = pg

    assert shard_pg is not None and replicate_pg is not None
    return shard_pg, replicate_pg, layout


def _build_failure_simulator(args, world_size: int, shard_size: int):
    """Return a configured FailureSimulator.

    Resolves the schedule from either --failure_schedule (YAML file) or the
    inline --failure_count / --failure_step_range / --failure_locations
    flags. The two sources are mutually exclusive. When no failures are
    requested, returns a simulator with an empty schedule so downstream
    ``begin_minibatch`` / ``may_fail_here`` calls become no-ops.
    """
    spec = ParallelismSpec(kind="hsdp", world_size=world_size, shard_size=shard_size)

    if args.failure_schedule and args.failure_count > 0:
        raise ValueError(
            "--failure_schedule and --failure_count are mutually exclusive; "
            "pick one source for the failure schedule."
        )

    if args.failure_schedule:
        schedule = FailureSchedule.load(args.failure_schedule)
        schedule.assert_matches_topology(spec)
        return FailureSimulator(schedule=schedule)

    if args.failure_count <= 0:
        empty = FailureSchedule(parallelism=spec, generator_config=None, entries=())
        return FailureSimulator(schedule=empty, enabled=False)

    if args.failure_step_range is None:
        raise ValueError(
            "--failure_count > 0 requires --failure_step_range START END."
        )
    if not args.failure_locations:
        raise ValueError(
            "--failure_count > 0 requires --failure_locations NAME:WEIGHT ..."
        )

    weights = {}
    for item in args.failure_locations:
        if ":" not in item:
            raise ValueError(
                f"--failure_locations entry {item!r} must be NAME:WEIGHT."
            )
        name, weight = item.rsplit(":", 1)
        weights[name] = float(weight)

    exclude_str = (args.failure_exclude_replicas or "").strip()
    exclude_ids = [int(x) for x in exclude_str.split(",") if x.strip()] if exclude_str else []

    start, end = args.failure_step_range
    schedule = generate_failure_schedule(
        parallelism=spec,
        seed=args.failure_seed,
        num_failures=args.failure_count,
        step_range=(start, end),
        location_weights=weights,
        sampling=args.failure_sampling,
        exclude_replica_ids=exclude_ids,
    )
    return FailureSimulator(schedule=schedule)


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    if args.backend == "ulfm":
        # PyTorch hardcodes a Store() placeholder for the ULFM backend and ignores
        # any store= argument. Work around: init first, then build a real TCPStore
        # using MPI barriers for coordination, then monkey-patch _world.pg_map so
        # subsequent new_group(backend="nccl") can exchange ncclUniqueIds.
        import datetime
        dist.init_process_group(backend="ulfm")
        _rank = dist.get_rank()
        _world_size = dist.get_world_size()
        master_addr = os.environ.get("MASTER_ADDR", "localhost")
        master_port = int(os.environ.get("MASTER_PORT", "29500"))
        if _rank == 0:
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=_world_size,
                is_master=True,
                timeout=dist.default_pg_timeout,
                wait_for_workers=False,
            )
        dist.barrier()
        if _rank != 0:
            store = dist.TCPStore(
                host_name=master_addr,
                port=master_port,
                world_size=_world_size,
                is_master=False,
                timeout=dist.default_pg_timeout,
            )
        dist.barrier()
        import torch.distributed.distributed_c10d as _c10d
        _default_pg = _c10d._get_default_group()
        _backend_str, _ = _c10d._world.pg_map[_default_pg]
        _c10d._world.pg_map[_default_pg] = (_backend_str, store)
    else:
        dist.init_process_group(backend=args.backend)

    # assert "LOCAL_RANK" in os.environ, "torchrun should set LOCAL_RANK"
    # global_rank = int(os.environ.get("RANK", os.environ.get("SLURM_PROCID")))
    # local_rank = int(os.environ.get("LOCAL_RANK", os.environ.get("SLURM_PROCID")))
    # world_size = int(os.environ["WORLD_SIZE"])

    global_rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("OMPI_COMM_WORLD_LOCAL_RANK",
                                    str(global_rank % max(torch.cuda.device_count(), 1))))
    torch.cuda.set_device(local_rank)

    if not args.single_gpu:
        if args.hsdp_shard_size is None:
            shard_size = max(torch.cuda.device_count(), 1)
        else:
            shard_size = args.hsdp_shard_size
        # Mirror nanotron's ParallelContext: build a derived world_pg via
        # dist.new_group(range(W), backend=...) — sibling of the default
        # group at MPI_COMM_WORLD, dedicated to world-level operations
        # (e.g. ULFM consensus). shard_pg / replicate_pg are also
        # MPI_COMM_WORLD-rooted siblings (PyTorch's new_group always
        # validates ranks against the default group).
        world_pg = dist.new_group(
            ranks=list(range(world_size)),
            backend=dist.get_backend(),
        )
        shard_pg, replicate_pg, hsdp_layout = build_hsdp_groups(
            world_size=world_size, shard_size=shard_size, backend=args.backend
        )
        logger.info(
            f"HSDP layout: num_replicas={hsdp_layout.num_replicas}, "
            f"shard_size={shard_size}, my replica={hsdp_layout.replica_id_of(global_rank)}, "
            f"my shard_rank={hsdp_layout.shard_rank_of(global_rank)}"
        )
    else:
        world_pg = None
        shard_pg = replicate_pg = hsdp_layout = None

    logger.info(
        f"Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
    )

    logger.info("Process group initialized")
    device = f"cuda:{local_rank}"

    if _ULFM_AVAILABLE and not args.single_gpu and args.backend == "ulfm":
        if args.ulfm_verbose:
            import logging as _logging
            ULFM.set_ulfm_verbose_logging(True)
            _logging.getLogger("ulfm_collectives").setLevel(_logging.DEBUG)
            if global_rank == 0:
                logger.info("ULFM verbose logging enabled (C++ + Python DEBUG)")
        sim = _build_failure_simulator(args, world_size=world_size, shard_size=shard_size)
        if sim is not None:
            if global_rank == 0:
                for _line in sim.describe().splitlines():
                    logger.info(_line)
            set_failure_simulator(sim)
            sim.initialize(rank=global_rank, world_size=world_size)
    else:
        sim = None

    if args.total_batch_size is not None:
        assert (
            args.total_batch_size % world_size == 0
        ), "total_batch_size must be divisible by world_size"
        args.gradient_accumulation = args.total_batch_size // (
            args.batch_size * world_size
        )
        if is_main_process():
            logger.info(
                f"{args.gradient_accumulation}-{world_size}-{args.total_batch_size}-{args.batch_size}"
            )
        assert (
            args.gradient_accumulation > 0
        ), "gradient_accumulation must be greater than 0"

    assert (
        args.gradient_accumulation * args.batch_size * world_size
        == args.total_batch_size
    ), "gradient_accumulation * batch_size * world_size must be equal to total_batch_size"

    # Loguru is used for script-level messages (loss / update step). Silence
    # it on non-rank-0 so the "Update step / loss" line only prints once.
    if global_rank != 0:
        logger.remove()

    # Stdlib logging is used by ulfm_collectives (orchestrator,
    # training_manager, hooks). Configure the root logger on EVERY rank with
    # an INFO-level StreamHandler so messages like "Restore completed",
    # "Communicator repaired", etc. surface from every rank that emits them.
    # Without this, stdlib's lastResort handler suppresses everything below
    # WARNING. --ulfm_verbose later may further bump ulfm_collectives to DEBUG.
    import logging as _logging
    _logging.basicConfig(
        level=_logging.INFO,
        format=f"%(asctime)s [%(levelname)s|rank{global_rank}] %(name)s: %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
    )

    if global_rank == 0:
        model_name = args.model_config.split("/")[1]
        model_name = model_name.split(".")[0]
        run_name = (
            f"{model_name}"
            if args.run_name is None
            else args.run_name
        )
        wandb.init(project=args.wandb_project, name=run_name)

    logger.info(f"Using dist with rank {global_rank} (only rank 0 will log)")
    logger.info("*" * 40)
    logger.info(f"Starting training with the arguments")
    for k, v in vars(args).items():
        logger.info(f"{k:30} {v}")
    logger.info("*" * 40)

    if args.offline_mode:
        logger.info("Loading tokenized data from disk")
        data = datasets.load_from_disk("/data/ziyueliu/datasets/.cache/huggingface/datasets/c4/tokenized/seq_len_4096")
        logger.info("Finished loading from disk")
    else:
        data = datasets.load_dataset("allenai/c4", "en", split="train", streaming=True)

        seed_for_shuffle = 42
        logger.info(f"Shuffling data with seed {seed_for_shuffle}")
        data: datasets.Dataset = data.shuffle(seed=seed_for_shuffle)

    if not args.single_gpu:
        if args.offline_mode:
            train_data: datasets.Dataset = data["train"]
            train_data = datasets.distributed.split_dataset_by_node(
                train_data,
                rank=global_rank,
                world_size=world_size,
            )
            eval_data = data["validation"]
            eval_data = datasets.distributed.split_dataset_by_node(
                eval_data,
                rank=global_rank,
                world_size=world_size,
            )
        else:
            data = datasets.distributed.split_dataset_by_node(
                data,
                rank=global_rank,
                world_size=world_size,
            )

    tokenizer = AutoTokenizer.from_pretrained(
        "t5-base", model_max_length=args.max_length
    )

    def preprocess_batched(batch):
        batch = tokenizer(
            batch["text"],
            max_length=args.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        return batch

    if args.offline_mode:
        train_dataloader = torch.utils.data.DataLoader(
            train_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=default_data_collator,
            shuffle=True,
        )
        eval_dataloader = torch.utils.data.DataLoader(
            eval_data,
            batch_size=args.batch_size,
            num_workers=args.workers,
            collate_fn=default_data_collator,
            shuffle=True,
        )
    else:
        # it doesn't matter which tokenizer we use, because we train from scratch
        # T5 tokenizer was trained on C4 and we are also training on C4, so it's a good choice

        dataset = PreprocessedIterableDataset(
            data, tokenizer, batch_size=args.batch_size, max_length=args.max_length
        )
        train_dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=None, num_workers=args.workers
        )
        eval_dataloader = None

    # Always build the architecture from --model_config. When resuming, we
    # load weights AFTER FSDP wrapping via dcp.load (sharded distributed
    # checkpoint), since save_every now writes sharded checkpoints — they
    # are not directly loadable via HF from_pretrained.
    if args.continue_from is not None:
        logger.info("*" * 40)
        logger.info(
            f"Will resume from sharded checkpoint at {args.continue_from} "
            "(architecture rebuilt from --model_config; weights loaded post-FSDP wrap)"
        )
    else:
        logger.info("*" * 40)
        logger.info("Building model from scratch")

    model_config = AutoConfig.from_pretrained(args.model_config)
    model = AutoModelForCausalLM.from_config(model_config)

    if args.dtype in ["bf16", "bfloat16"] and not args.mixed_precision:
        model = model.to(device=device, dtype=torch.bfloat16)
    else:
        model = model.to(device=device)

    if args.activation_checkpointing:
        # Non-reentrant: FSDP post-backward hook fires once per flat_param per
        # backward. Reentrant mode can fire it multiple times, which breaks the
        # sync-step fold in _reduce_grad (would double-count on the 2nd fire).
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    if not args.single_gpu:
        wrap_policy = functools.partial(
            transformer_auto_wrap_policy,
            transformer_layer_cls={LlamaDecoderLayer},
        )
        mp_policy = None
        if args.mixed_precision:
            mp_policy = MixedPrecision(
                param_dtype=torch.bfloat16,
                reduce_dtype=torch.float32,
                buffer_dtype=torch.bfloat16,
                keep_low_precision_grads=False,
            )
        model = FullyShardedDataParallel(
            LMWrapper(model),
            sharding_strategy=ShardingStrategy.HYBRID_SHARD,
            process_group=(shard_pg, replicate_pg),
            auto_wrap_policy=wrap_policy,
            device_id=local_rank,
            use_orig_params=True,
            mixed_precision=mp_policy,
        )

    # Resume model weights from a sharded checkpoint. dcp.load is a
    # collective — every rank must enter the state_dict_type and dcp.load
    # block. The pattern: build a template state_dict by reading the current
    # (freshly initialized) sharded params, hand it to dcp.load to overwrite
    # in-place from disk, then load_state_dict to reapply.
    if args.continue_from is not None:
        if args.single_gpu:
            sd = torch.load(
                os.path.join(args.continue_from, "pytorch_model.bin"),
                map_location="cpu",
            )
            model.load_state_dict(sd)
        else:
            with FullyShardedDataParallel.state_dict_type(
                model, StateDictType.SHARDED_STATE_DICT
            ):
                sharded_sd = {"model": model.state_dict()}
                dcp.load(
                    state_dict=sharded_sd,
                    storage_reader=dcp.FileSystemReader(args.continue_from),
                )
                model.load_state_dict(sharded_sd["model"])
        logger.info(f"Loaded sharded model weights from {args.continue_from}")

    global_step = 0
    update_step = 0
    tokens_seen = 0
    tokens_seen_before = 0

    # ====== starting config ======= #

    n_total_params = sum(p.numel() for p in model.parameters())
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    layer_wise_flag = True if "per_layer" in args.optimizer.lower() else False

    optimizer = training_utils.build_optimizer(model, trainable_params, args)
    if layer_wise_flag:
        if not isinstance(optimizer, dict):
            raise ValueError("Layer-wise optimizer is not properly constructed.")

    if not layer_wise_flag:
        scheduler = training_utils.get_scheculer(
            optimizer=optimizer,
            scheduler_type=args.scheduler,
            num_training_steps=args.num_training_steps,
            warmup_steps=args.warmup_steps,
            min_lr_ratio=args.min_lr_ratio,
        )

    if args.continue_from is not None:
        optimizer_checkpoint = torch.load(
            os.path.join(args.continue_from, "optimizer.pt"), map_location="cpu"
        )
        scheduler.load_state_dict(optimizer_checkpoint["scheduler"])
        # Load sharded optimizer state via FSDP's distributed API. Each rank
        # reads its own shard from the dcp checkpoint dir; the result is
        # rekeyed to flat-param ids and applied via optimizer.load_state_dict.
        if args.single_gpu and "optimizer" in optimizer_checkpoint:
            optimizer.load_state_dict(optimizer_checkpoint["optimizer"])
        elif not args.single_gpu:
            with FullyShardedDataParallel.state_dict_type(
                model, StateDictType.SHARDED_STATE_DICT
            ):
                optim_template = FullyShardedDataParallel.optim_state_dict(
                    model, optimizer
                )
                state_dict_to_load = {"optim": optim_template}
                dcp.load(
                    state_dict=state_dict_to_load,
                    storage_reader=dcp.FileSystemReader(args.continue_from),
                )
                flattened_osd = FullyShardedDataParallel.optim_state_dict_to_load(
                    model, optimizer, state_dict_to_load["optim"]
                )
                optimizer.load_state_dict(flattened_osd)
        logger.info(f"Optimizer and scheduler restored from {args.continue_from}")

        if os.path.exists(os.path.join(args.continue_from, "training_state.json")):
            logger.info(
                f"Loading training state like global_step, update_step, and tokens_seen from {args.continue_from}"
            )
            with open(os.path.join(args.continue_from, "training_state.json")) as f:
                _old_state = json.load(f)
            global_step = _old_state["global_step"]
            update_step = _old_state["update_step"]
            tokens_seen = _old_state["tokens_seen"]
            tokens_seen_before = _old_state["tokens_seen_before"]
            logger.info(f"global_step       : {global_step}")
            logger.info(f"update_step       : {update_step}")
            logger.info(f"tokens_seen       : {tokens_seen}")
            logger.info(f"tokens_seen_before: {tokens_seen_before}")
            logger.info(
                f"Will train for {args.num_training_steps - update_step} update steps"
            )

    scheduler_start_step = update_step

    # print params and trainable params
    logger.info(f"\n{model}\n")
    logger.info(
        f"All params: \n{[n for n,p in model.named_parameters() if p.requires_grad]}\n"
    )
    logger.info(
        f"Total params: {sum(p.numel() for p in model.parameters()) / 1_000_000:.2f}M"
    )
    logger.info(
        f"Trainable params: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1_000_000:.2f}M"
    )
    logger.info(f"Saving model to {args.save_dir} every {args.save_every} update steps")

    # Initialize wandb
    run_config = dict(vars(args))
    run_config.update(
        {
            "max_lr": run_config.pop(
                "lr"
            ),  # rename lr to max_lr to avoid conflicts with scheduler
            "total_params_M": n_total_params / 1_000_000,
            "dataset": "allenai/c4",
            "model": model_config.to_dict(),
            "world_size": world_size,
            "device": str(device),
        }
    )

    if global_rank == 0:
        wandb.config.update(run_config, allow_val_change=True)
        wandb.save(os.path.abspath(__file__), policy="now")  # save current script

    lm_criterion = lambda output, _: output.loss

    training_manager = None
    if not args.single_gpu and args.backend == "ulfm":
        if not _ULFM_AVAILABLE:
            raise RuntimeError(
                "ulfm_collectives not available; cannot run --backend ulfm without it."
            )
        training_manager = HSDPULFMTrainingManager(
            fsdp_model=model,
            replicate_pg=replicate_pg,
            world_pg=world_pg,
            grad_accum_steps=args.gradient_accumulation,
            policy_type="static",
            initial_world_size=hsdp_layout.num_replicas,
        )

    # global steps and others are defined above
    pad_idx = tokenizer.pad_token_id
    update_time = time.time()
    local_step = 0  # when continue_from is used, local_step != global_step

    # ##############################
    # TRAINING LOOP
    # ##############################

    max_memory = torch.cuda.max_memory_allocated()
    if global_rank == 0:
        logger.info(f"Maximum memory allocated before training: {max_memory} bytes\n")
    torch.cuda.reset_peak_memory_stats()

    if not args.single_gpu:
        dist.barrier()

    if global_rank == 0:
        print(f"Rank {global_rank} starting training loop.")

    # Helper: bring a microbatch to device, set labels, count tokens.
    def _prepare_batch(batch):
        batch = {k: v.to(device) for k, v in batch.items()}
        batch["labels"] = (
            batch["input_ids"].clone() if "labels" not in batch else batch["labels"]
        )
        batch["labels"][batch["labels"] == pad_idx] = -100
        return batch

    # Outer loop is per UPDATE STEP; inner loop is gradient accumulation.
    # For ULFM, the inner microbatch loop is wrapped by a restore-mode loop
    # that may run extra microbatches at a policy boundary (mirrors nanotron).
    data_iter = iter(train_dataloader)
    batch_idx = -1  # cumulative microbatch counter (matches old `batch_idx`)

    # Resume: skip microbatches already seen in earlier update_steps.
    if update_step > 0:
        batches_to_skip = update_step * args.gradient_accumulation
        for _ in range(batches_to_skip):
            try:
                next(data_iter)
                batch_idx += 1
            except StopIteration:
                break

    while update_step < args.num_training_steps:
        # ============================================================
        # Single-GPU / NCCL backend: simple inner microbatch loop
        # ============================================================
        if args.single_gpu or args.backend == "nccl":
            loss = None
            data_exhausted = False
            for micro_idx in range(args.gradient_accumulation):
                try:
                    raw_batch = next(data_iter)
                except StopIteration:
                    data_exhausted = True
                    break
                batch_idx += 1
                global_step += 1
                local_step += 1
                batch = _prepare_batch(raw_batch)
                tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

                if args.single_gpu:
                    loss = model(**batch).loss
                else:
                    loss = model(batch).loss
                scaled_loss = loss / args.gradient_accumulation
                scaled_loss.backward()

            if data_exhausted:
                break

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

        # ============================================================
        # ULFM backend: nanotron-style restore-mode loop around the
        # microbatch loop. On failure, query orchestrator for extra
        # microbatches and run another pass before optimizer step.
        # ============================================================
        else:
            is_first_pass = True
            loss = None
            data_exhausted = False

            while True:
                if is_first_pass:
                    n_micro = training_manager.get_effective_n_microbatches()
                else:
                    n_micro = training_manager.get_n_extra_microbatches()

                training_manager.prepare_iteration(is_first_pass=is_first_pass)
                training_manager.on_world_consensus()

                # Inner microbatch loop (forward + backward only)
                for micro_idx in range(n_micro):
                    try:
                        raw_batch = next(data_iter)
                    except StopIteration:
                        data_exhausted = True
                        break
                    batch_idx += 1
                    global_step += 1
                    local_step += 1
                    batch = _prepare_batch(raw_batch)
                    tokens_seen += (batch["input_ids"] != pad_idx).sum().item() * world_size

                    # Extended pass: wait for non-blocking restore before the
                    # first microbatch's backward.
                    if not is_first_pass and micro_idx == 0:
                        training_manager.wait_restore_before_backward()

                    sim.begin_minibatch(batch_idx)
                    with sim.may_fail_here("pre-forward"):
                        loss = training_manager.microbatch_step(
                            batch_idx, micro_idx, n_micro,
                            batch, None, lm_criterion,
                        )

                if data_exhausted:
                    break

                is_first_pass = False

                # Cross-replica allreduce on each FSDP unit's grad shard.
                training_manager.fire_cross_replica_allreduces()
                # Intra-replica barrier: ensure all shard-mates within a
                # replica observe the same outcome of the cross-replica
                # reduce before consensus / restore-mode dispatch. NCCL
                # collective on shard_pg, no MPI traffic.
                if shard_pg is not None:
                    dist.barrier(group=shard_pg)
                training_manager.on_consensus_step()

                mode = training_manager.get_restore_mode()
                if mode == GradRestoreMode.SKIP:
                    break  # success, proceed to optimizer step
                if mode == GradRestoreMode.NON_BLOCKING:
                    # Policy boundary: async restore + extra microbatches.
                    training_manager.start_nonblocking_restore()
                    continue

                # BLOCKING: blocking restore (re-reduce) loop
                crossed_boundary = False
                blocking_attempts = 0
                while training_manager.get_restore_mode() == GradRestoreMode.BLOCKING:
                    training_manager.start_blocking_restore()
                    blocking_attempts += 1
                    if training_manager.is_at_policy_boundary():
                        training_manager.start_nonblocking_restore()
                        crossed_boundary = True
                        break
                    if blocking_attempts > 3:
                        raise RuntimeError(
                            f"[Rank {global_rank}] Blocking restore retry limit exceeded"
                        )
                if crossed_boundary:
                    continue  # outer: run extra microbatches
                break  # restored without crossing boundary, proceed

            if data_exhausted:
                break

            # Post-loop: normalize, clip, optimizer step
            training_manager.normalize_gradients()
            if args.grad_clipping != 0.0:
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), args.grad_clipping
                )
            grad_norm = training_manager.compute_grad_norm()
            training_manager.optimizer_step(optimizer)

        if global_rank == 0:
            logger.info(
                f"Update step {update_step}/{args.num_training_steps}, global step {global_step}, loss: {loss.item() if isinstance(loss, torch.Tensor) else loss:.4f}"
            )

        if not layer_wise_flag:
            scheduler.step()

        update_step += 1
        # Match nanotron: sync GPU before timing so iter_time includes all
        # outstanding kernels (FSDP reshard, optimizer step, etc.).
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        update_time = time.time() - update_time

        # save checkpoint by save_every
        # Sharded distributed checkpoint: each rank writes its own shard
        # to the same directory. No single rank holds the full model in
        # memory. Resume via dcp.load with the same FSDP wrapping.
        if (
            local_step > args.gradient_accumulation
            and update_step % args.save_every == 0
        ):
            current_model_directory = f"{args.save_dir}/model_{update_step}"
            if global_rank == 0:
                logger.info(
                    f"Saving sharded model checkpoint to {current_model_directory}, update step {update_step}"
                )
                os.makedirs(current_model_directory, exist_ok=True)
            if not args.single_gpu:
                dist.barrier()  # ensure dir exists before all ranks write

            if args.single_gpu:
                # No FSDP — single rank, save HF-style.
                model.save_pretrained(
                    current_model_directory, max_shard_size="100GB"
                )
            else:
                # All ranks must enter state_dict_type, optim_state_dict, and
                # dcp.save (collective). Sharded model + sharded optimizer.
                with FullyShardedDataParallel.state_dict_type(
                    model, StateDictType.SHARDED_STATE_DICT
                ):
                    sharded_sd = {
                        "model": model.state_dict(),
                        "optim": FullyShardedDataParallel.optim_state_dict(
                            model, optimizer
                        ),
                    }
                dcp.save(
                    state_dict=sharded_sd,
                    storage_writer=dcp.FileSystemWriter(current_model_directory),
                )

            # Scheduler + training metadata are small — keep rank-0-only.
            # Optimizer state lives in the sharded dcp checkpoint above.
            if global_rank == 0:
                optimizer_checkpoint = {
                    "scheduler": scheduler.state_dict(),
                    "update_step": update_step,
                    "global_step": global_step,
                    "config": run_config,
                    "wandb": wandb.run.dir,
                    "dtype": args.dtype,
                }
                torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

                training_state_checkpoint = {
                    "global_step": global_step,
                    "update_step": update_step,
                    "tokens_seen": tokens_seen,
                    "tokens_seen_before": tokens_seen_before,
                    "update_time": update_time,
                }
                with open(f"{current_model_directory}/training_state.json", "w") as f:
                    json.dump(training_state_checkpoint, f, indent=4)

                # save wandb related info
                wandb_info = {
                    "wandb_id": wandb.run.id,
                }
                with open(f"{args.save_dir}/wandb.json", "w") as f:
                    json.dump(wandb_info, f, indent=4)

        # evaluation
        if update_step % args.eval_every == 0:
            logger.info(f"Performing evaluation at step {update_step}")
            model.eval()
            total_loss, evaluated_on_tokens = evaluate_model(
                model,
                preprocess_batched,
                pad_idx,
                global_rank,
                world_size,
                device,
                args.batch_size,
                eval_dataloader,
            )
            if global_rank == 0:
                wandb.log(
                    {
                        "final_eval_loss": total_loss,
                        "final_eval_perplexity": np.exp(total_loss),
                        "final_eval_tokens": evaluated_on_tokens,
                    },
                    step=global_step,
                )
            logger.info(
                f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
            )
            model.train()

        if not layer_wise_flag:
            lr = optimizer.param_groups[0]["lr"]
        else:
            lr = list(optimizer.values())[0].param_groups[0]["lr"]
        tokens_in_update = tokens_seen - tokens_seen_before
        tokens_seen_before = tokens_seen
        batches_in_update = args.gradient_accumulation * world_size

        max_memory = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

        # Nanotron-style throughput. Uses global_batch_size × seq_len (no pad
        # adjustment) so values are directly comparable with nanotron runs.
        elapsed_time_per_iteration_ms = update_time * 1000.0
        tokens_per_iter = args.total_batch_size * args.max_length
        tokens_per_sec = tokens_per_iter / update_time if update_time > 0 else 0.0
        consumed_tokens = update_step * tokens_per_iter
        # Live total GPUs: for ULFM, replicate_pg shrinks under failures while
        # shard_pg stays stable (failures are per-replica). For NCCL it's static.
        if args.backend == "ulfm" and training_manager is not None:
            total_gpus = training_manager.txn.curr_world_size * shard_size
        else:
            total_gpus = world_size
        tokens_per_sec_per_gpu = tokens_per_sec / total_gpus if total_gpus > 0 else 0.0

        # ULFM-specific metrics (mirrors nanotron_ulfm trainer_ulfm.py:498-513).
        ulfm_metrics = {}
        if args.backend == "ulfm" and training_manager is not None:
            txn = training_manager.txn
            curr_grad_accum = txn.curr_grad_accum_steps
            curr_dp_size = txn.curr_world_size
            gbs = txn.effective_batch_size
            total_workload = curr_dp_size * curr_grad_accum
            redundancy = (
                1.0 - (gbs / total_workload) if total_workload > 0 else 0.0
            )
            ulfm_metrics = {
                "ulfm/num_majors": txn.num_major_procs,
                "ulfm/num_minors": txn.num_minor_procs,
                "ulfm/num_major_spares": txn.num_major_spare_procs,
                "ulfm/num_minor_spares": txn.num_minor_spare_procs,
                "ulfm/curr_grad_accum": curr_grad_accum,
                "ulfm/curr_minor_grad_accum": txn.minor_proc_grad_accum_steps,
                "ulfm/redundancy": redundancy,
            }

        if global_rank == 0:
            wandb.log(
                {
                    "loss": loss.item() if isinstance(loss, torch.Tensor) else loss,
                    "lr": lr,
                    "update_step": update_step,
                    "consumed_tokens": consumed_tokens,
                    "elapsed_time_per_iteration_ms": elapsed_time_per_iteration_ms,
                    "tokens_per_sec": tokens_per_sec,
                    "tokens_per_sec_per_gpu": tokens_per_sec_per_gpu,
                    "total_gpus": total_gpus,
                    "global_batch_size": args.total_batch_size,
                    "tokens_seen": tokens_seen,
                    "throughput_tokens": tokens_in_update / update_time,
                    "throughput_examples": args.total_batch_size / update_time,
                    "throughput_batches": batches_in_update / update_time,
                    "gradnorm": grad_norm,
                    "max_memory": max_memory,
                    **ulfm_metrics,
                },
                step=global_step,
            )

        update_time = time.time()

    # ##############################
    # END of training loop
    # ##############################
    logger.info("Training finished")

    current_model_directory = f"{args.save_dir}/model_{update_step}"
    # Final sharded save: all ranks must enter the FSDP state_dict_type and
    # dcp.save (collective). The directory + log are rank-0-only.
    save_final = not os.path.exists(current_model_directory)
    if save_final and global_rank == 0:
        logger.info(
            f"Saving final sharded model checkpoint to {current_model_directory}, update step {update_step}"
        )
        os.makedirs(current_model_directory, exist_ok=True)
    if save_final and not args.single_gpu:
        dist.barrier()
        with FullyShardedDataParallel.state_dict_type(
            model, StateDictType.SHARDED_STATE_DICT
        ):
            sharded_sd = {
                "model": model.state_dict(),
                "optim": FullyShardedDataParallel.optim_state_dict(
                    model, optimizer
                ),
            }
        dcp.save(
            state_dict=sharded_sd,
            storage_writer=dcp.FileSystemWriter(current_model_directory),
        )
    elif save_final and args.single_gpu and global_rank == 0:
        model.save_pretrained(current_model_directory)

    if save_final and global_rank == 0:

        # Optimizer state lives in the sharded dcp checkpoint above; only
        # save scheduler + metadata here.
        optimizer_checkpoint = {
            "scheduler": scheduler.state_dict(),
            "update_step": update_step,
            "global_step": global_step,
            "config": run_config,
            "wandb": wandb.run.dir,
            "dtype": args.dtype,
        }
        torch.save(optimizer_checkpoint, f"{current_model_directory}/optimizer.pt")

        training_state_checkpoint = {
            "global_step": global_step,
            "update_step": update_step,
            "tokens_seen": tokens_seen,
            "tokens_seen_before": tokens_seen_before,
            "update_time": update_time,
        }
        with open(f"{current_model_directory}/training_state.json", "w") as f:
            json.dump(training_state_checkpoint, f, indent=4)

    # Final evaluation
    logger.info("Running final evaluation")
    model.eval()
    del loss, optimizer, scheduler
    import gc

    gc.collect()
    torch.cuda.empty_cache()

    total_loss, evaluated_on_tokens = evaluate_model(
        model,
        preprocess_batched,
        pad_idx,
        global_rank,
        world_size,
        device,
        args.batch_size,
        eval_dataloader,
    )

    if global_rank == 0:
        wandb.log(
            {
                "final_eval_loss": total_loss,
                "final_eval_perplexity": np.exp(total_loss),
                "final_eval_tokens": evaluated_on_tokens,
            },
            step=global_step,
        )
        logger.info(
            f"Eval loss and perplexity at step {update_step}: {total_loss}, {np.exp(total_loss)}"
        )

    logger.info("Script finished successfully")
    if global_rank == 0:
        print("Finished successfully")


if __name__ == "__main__":
    print("Starting script")
    args = parse_args(None)
    main(args)
