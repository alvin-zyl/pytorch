#!/usr/bin/env python3
"""
Advanced ULFM DDP Example with Generalized Failure Recovery

This example demonstrates the new generalized failure recovery system:
1. WorkULFM for per-operation failure detection
2. ProcessGroup-level recovery methods
3. Flexible failure handling strategies
4. Integration with PyTorch DDP training loops
"""

import os, time
import signal
import logging
import argparse
import torch
import torch.nn as nn
import torch.distributed as dist

try:
    import ulfm_collectives as ULFM
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )
from training_manager import ULFMTrainingManager
from failure_simulator import FailureSimulator, set_failure_simulator

# Logger will be configured based on CLI arguments
logger = logging.getLogger(__name__)


def log_rank0(message, level=logging.INFO):
    """Log basic training progress from rank 0 only."""
    if dist.is_initialized() and dist.get_rank() == 0:
        logger.log(level, message)


def create_simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 1000),
        nn.ReLU(),
        nn.Linear(1000, 4000),
        nn.ReLU(),
        nn.Linear(4000, 1000),
        nn.ReLU(),
        nn.Linear(1000, 1),
    )
    return model


def create_synthetic_data(batch_size=32, num_batches=50, input_size=10):
    """Create synthetic training data."""
    data_batches = []
    for _ in range(num_batches):
        x = torch.randn(batch_size, input_size)
        y = torch.randn(batch_size, 1)
        data_batches.append((x, y))
    return data_batches


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Advanced ULFM DDP Training with Generalized Recovery"
    )

    parser.add_argument(
        "--auto-repair",
        action="store_true",
        default=True,
        help="Enable automatic communicator repair (default: True)",
    )
    parser.add_argument(
        "--no-auto-repair",
        dest="auto_repair",
        action="store_false",
        help="Disable automatic communicator repair",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--failure-strategy",
        choices=["continue", "restart", "abort"],
        default="continue",
        help="Failure handling strategy (default: continue)",
    )
    parser.add_argument(
        "--epochs", type=int, default=1, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=30,
        help="Number of batches per epoch (default: 50)",
    )
    parser.add_argument(
        "--target-world-size",
        type=int,
        default=4,
        help="Target world size (default: 4)",
    )
    parser.add_argument(
        "--grad-accum-steps",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    parser.add_argument(
        "--simulate-failure",
        "-s",
        action="store_true",
        default=False,
        help="Simulate process failure for testing (only affects rank 1)",
    )
    parser.add_argument(
        "--policy",
        choices=["adaptive", "static"],
        default="static",
        help="Fault tolerance policy (default: static)",
    )

    return parser.parse_args()


def main():
    """Main training function with generalized recovery."""
    # Parse command line arguments
    args = parse_args()

    # Initialize distributed training with ULFM backend
    dist.init_process_group(backend="ulfm")

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    sim = FailureSimulator(
        seed=42,
        desired_failures=1,
        total_minibatches=30,
        target_ranks={1, 2}
    )
    set_failure_simulator(sim)
    sim.initialize(rank=dist.get_rank(), world_size=dist.get_world_size())                                                                                                                                                                       


    # Configure logging with rank info, module name, and verbosity level
    log_level = logging.DEBUG if args.verbose else logging.INFO

    # Enhanced format with file location (filename:line) for verbose mode
    if args.verbose:
        log_format = f"[Rank {rank}/{world_size}] [%(name)s:%(lineno)d] %(levelname)s: %(message)s"
    else:
        log_format = f"[Rank {rank}/{world_size}] [%(name)s] %(levelname)s: %(message)s"

    logging.basicConfig(
        level=log_level,
        format=log_format,
    )

    logger.info(f"Starting advanced ULFM training on rank {rank}/{world_size}")

    # Configure ULFM C++ verbose logging based on CLI argument
    ULFM.set_ulfm_verbose_logging(args.verbose)

    if args.verbose:
        log_rank0(f"CLI arguments: {vars(args)}", logging.DEBUG)
        log_rank0(
            f"ULFM verbose logging enabled: {ULFM.is_ulfm_verbose_logging()}",
            logging.DEBUG,
        )

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_simple_model().to(device)
    log_rank0(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create training manager with policy-driven recovery
    training_manager = ULFMTrainingManager(
        model,
        grad_accum_steps=args.grad_accum_steps,
        failure_strategy=args.failure_strategy,
        enable_auto_repair=args.auto_repair,
        policy_type=args.policy,
        # target_world_size=args.target_world_size,
        initial_world_size=world_size,
    )

    # Create synthetic data
    train_data = create_synthetic_data(
        batch_size=args.batch_size, num_batches=args.num_batches
    )
    log_rank0(f"Created synthetic dataset: {len(train_data)} batches")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(training_manager.ddp_model.parameters(), lr=0.001)

    # Training loop with failure simulation
    num_epochs = args.epochs
    log_rank0(f"Starting resilient training for {num_epochs} epochs...")

    if args.verbose:
        log_rank0(
            f"Training configuration: auto_repair={args.auto_repair}, strategy={args.failure_strategy}",
            logging.DEBUG,
        )
        log_rank0(f"Failure simulation: {args.simulate_failure}", logging.DEBUG)

    dist.barrier()
    update_steps = 0
    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        successful_batches = 0

        log_rank0(f"Starting epoch {epoch}")

        for batch_idx, (data, target) in enumerate(train_data):
            sim.begin_minibatch(batch_idx)

            data, target = data.to(device), target.to(device)

            with sim.may_fail_here("pre-forward"):
                loss, stepped = training_manager.train_step(
                    batch_idx, data, target, criterion, optimizer
                )

            if loss is not None:
                epoch_loss += loss
                successful_batches += 1
                logger.debug(f"Minibatch {batch_idx} completed")

                if stepped:
                    update_steps += 1
                    log_rank0(
                        f"Epoch {epoch}, Batch {batch_idx}, Update step {update_steps}, Loss: {loss:.6f}"
                    )

    # Report final statistics
    stats = training_manager.get_recovery_stats()
    log_rank0("=== Training Completed ===")
    log_rank0(f"Recovery Statistics: {stats}")
    logger.debug(f"[Rank {rank}] Sleeping briefly to ensure clean exit...")
    time.sleep(1)
    logger.debug(f"[Rank {rank}] Exiting now.")


if __name__ == "__main__":
    main()
