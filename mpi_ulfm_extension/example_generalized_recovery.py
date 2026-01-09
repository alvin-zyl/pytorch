#!/usr/bin/env python3
"""
Advanced ULFM DDP Example with Generalized Failure Recovery

This example demonstrates the new generalized failure recovery system:
1. WorkULFM for per-operation failure detection
2. ProcessGroup-level recovery methods
3. Flexible failure handling strategies
4. Integration with PyTorch DDP training loops
"""

import os
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

# Logger will be configured based on CLI arguments
logger = logging.getLogger(__name__)


def create_simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 10000),
        nn.ReLU(),
        nn.Linear(10000, 40000),
        nn.ReLU(),
        nn.Linear(40000, 10000),
        nn.ReLU(),
        nn.Linear(10000, 10000),
        nn.ReLU(),
        nn.Linear(10000, 1),
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
        default=5,
        help="Number of batches per epoch (default: 50)",
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
        choices=["adaptive", "fixed"],
        default="adaptive",
        help="Fault tolerance policy (default: adaptive)",
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
        logger.debug(f"CLI arguments: {vars(args)}")
        logger.debug(f"ULFM verbose logging enabled: {ULFM.is_ulfm_verbose_logging()}")

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_simple_model().to(device)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create training manager with policy-driven recovery
    training_manager = ULFMTrainingManager(
        model,
        failure_strategy=args.failure_strategy,
        enable_auto_repair=args.auto_repair,
        policy_type=args.policy,
    )

    # Create synthetic data
    train_data = create_synthetic_data(
        batch_size=args.batch_size, num_batches=args.num_batches
    )
    logger.info(f"Created synthetic dataset: {len(train_data)} batches")

    # Setup training
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(training_manager.ddp_model.parameters(), lr=0.001)

    # Training loop with failure simulation
    num_epochs = args.epochs
    logger.info(f"Starting resilient training for {num_epochs} epochs...")

    if args.verbose:
        logger.debug(
            f"Training configuration: auto_repair={args.auto_repair}, strategy={args.failure_strategy}"
        )
        logger.debug(f"Failure simulation: {args.simulate_failure}")

    for epoch in range(1, num_epochs + 1):
        epoch_loss = 0.0
        successful_batches = 0

        logger.info(f"Starting epoch {epoch}")

        for batch_idx, (data, target) in enumerate(train_data):
            # Simulate failure for testing (rank 1, epoch 3, batch 20)
            if args.simulate_failure and rank == 2 and epoch == 1 and batch_idx == 0:
                logger.warning(f"[Rank {rank}] Simulating process failure")
                os.kill(os.getpid(), signal.SIGKILL)

            data, target = data.to(device), target.to(device)

            loss, stepped = training_manager.train_step(
                batch_idx, data, target, criterion, optimizer
            )

            if loss is not None:
                epoch_loss += loss
                successful_batches += 1

                if stepped:
                    logger.debug(f"Batch {batch_idx} completed")

                if successful_batches % 10 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}")
                elif args.verbose and successful_batches % 5 == 0:
                    logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}")

        # Report epoch results
        if successful_batches > 0:
            avg_loss = epoch_loss / successful_batches
            logger.info(
                f"Epoch {epoch} completed: {successful_batches}/{len(train_data)} batches, Avg Loss: {avg_loss:.6f}"
            )
        else:
            logger.error(f"Epoch {epoch} failed completely")

    # Report final statistics
    stats = training_manager.get_recovery_stats()
    logger.info("=== Training Completed ===")
    logger.info(f"Recovery Statistics: {stats}")


if __name__ == "__main__":
    main()
