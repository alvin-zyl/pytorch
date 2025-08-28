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
from torch.nn.parallel import DistributedDataParallel

try:
    import ulfm_collectives as ULFM
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )

# Logger will be configured based on CLI arguments
logger = logging.getLogger(__name__)


class ULFMTrainingManager:
    """
    Training manager that handles ULFM failure recovery in a structured way.
    """

    def __init__(
        self, model, failure_strategy="continue", enable_auto_repair=True, verbose=False
    ):
        self.failure_strategy = failure_strategy
        self.enable_auto_repair = enable_auto_repair
        self.verbose = verbose
        self.process_group = dist.group.WORLD

        # Verify we have ULFM process group
        if not isinstance(self.process_group, ULFM.ProcessGroupULFM):
            raise ValueError("Must use ULFM backend: dist.init_process_group('ulfm')")

        # Create DDP model with ULFM hook
        self.ddp_model = DistributedDataParallel(model)
        self._register_ulfm_hook()

        # Recovery statistics
        self.failure_count = 0
        self.recovery_count = 0

        if self.verbose:
            logger.info(
                f"ULFMTrainingManager initialized with auto_repair={enable_auto_repair}, strategy={failure_strategy}"
            )

    def _register_ulfm_hook(self):
        """Register ULFM communication hook with recovery logic."""
        hook = self._create_recovery_hook()
        self.ddp_model.register_comm_hook(state=self.process_group, hook=hook)
        logger.info(f"ULFM hook registered with strategy: {self.failure_strategy}")

    def _create_recovery_hook(self):
        """Create ULFM hook with comprehensive recovery logic."""
        strategy_map = {
            "continue": ULFM.ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS,
            "restart": ULFM.ULFMFailureHandlingStrategy.RESTART_FAILED_PROCESSES,
            "abort": ULFM.ULFMFailureHandlingStrategy.ABORT_ON_FAILURE,
        }

        def recovery_hook(state: ULFM.ProcessGroupULFM, bucket):
            """Advanced ULFM hook with failure recovery."""
            try:
                tensor = bucket.buffer()
                tensor.div_(state.size())

                # Configure ULFM options
                ulfm_opts = ULFM.ULFMOptions()
                ulfm_opts.auto_repair = self.enable_auto_repair
                ulfm_opts.failure_strategy = strategy_map[self.failure_strategy]
                ulfm_opts.max_retries = 3
                ulfm_opts.retry_delay_ms = 100

                # Configure allreduce options
                allreduce_opts = torch.distributed.AllreduceOptions()
                allreduce_opts.reduceOp = torch.distributed.ReduceOp.SUM

                # Perform ULFM allreduce
                work = state.ulfm_allreduce([tensor], allreduce_opts, ulfm_opts)

                def handle_completion(fut):
                    """Handle work completion and check for failures."""
                    try:
                        result_tensor = fut.value()[0]

                        # Check for failures using WorkULFM
                        if hasattr(work, "has_failures") and work.has_failures():
                            failed_ranks = work.get_failed_ranks()
                            self.failure_count += 1

                            logger.warning(
                                f"Communication failures detected in ranks: {failed_ranks}"
                            )

                            if self.verbose:
                                logger.info(
                                    f"Total failures so far: {self.failure_count}"
                                )

                            if not self.enable_auto_repair:
                                if state.repair_communicator():
                                    self.recovery_count += 1
                                    logger.info(
                                        "Process group recovery (manual) successful"
                                    )
                                else:
                                    logger.error(
                                        "Process group recovery (manual) failed"
                                    )
                                    raise RuntimeError(
                                        "Failed to recover (manual) from process failures"
                                    )
                            else:
                                if not state.check_for_failures():
                                    self.recovery_count += 1
                                    logger.info(
                                        "Auto-repair handled failures successfully"
                                    )
                                else:
                                    logger.error(
                                        "Process group recovery (auto-repair) failed"
                                    )
                                    raise RuntimeError(
                                        "Failed to recover (auto-repair) from process failures"
                                    )

                        return result_tensor

                    except Exception as e:
                        logger.error(f"Error handling work completion: {e}")
                        raise

                return work.get_future().then(handle_completion)

            except Exception as e:
                logger.error(f"ULFM recovery hook failed: {e}")
                # Create fallback future
                future = torch.futures.Future()
                future.set_result(bucket.buffer())
                return future

        return recovery_hook

    def train_step(self, data, target, criterion, optimizer):
        """
        Single training step with comprehensive error handling.

        Returns:
            loss: Training loss (or None if step failed)
            recovered: Whether recovery was performed
        """
        recovered = False

        try:
            optimizer.zero_grad()
            output = self.ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            return loss.item(), recovered

        except RuntimeError as e:
            if "process failures" in str(e).lower():
                logger.warning(f"Training step failed due to process failures: {e}")

                # Attempt recovery at training level
                if self._attempt_training_recovery():
                    recovered = True
                    logger.info("Training-level recovery successful, retrying step")
                    return self.train_step(data, target, criterion, optimizer)
                else:
                    logger.error("Training-level recovery failed")
                    raise
            else:
                # Re-raise non-ULFM errors
                raise

    def _attempt_training_recovery(self):
        """Attempt recovery at the training level."""
        try:
            # Check process group status
            if self.process_group.check_for_failures():
                logger.info("Attempting training-level communicator repair")
                if self.process_group.repair_communicator():
                    self.recovery_count += 1
                    return True
                else:
                    return False
            else:
                # No failures detected
                return True

        except Exception as e:
            logger.error(f"Training-level recovery attempt failed: {e}")
            return False

    def get_recovery_stats(self):
        """Get failure and recovery statistics."""
        return {
            "failure_count": self.failure_count,
            "recovery_count": self.recovery_count,
            "success_rate": self.recovery_count / max(1, self.failure_count),
        }


def create_simple_model():
    """Create a simple model for testing."""
    model = nn.Sequential(
        nn.Linear(10, 50), nn.ReLU(), nn.Linear(50, 20), nn.ReLU(), nn.Linear(20, 1)
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
        "--epochs", type=int, default=5, help="Number of training epochs (default: 5)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Training batch size (default: 32)"
    )
    parser.add_argument(
        "--num-batches",
        type=int,
        default=50,
        help="Number of batches per epoch (default: 50)",
    )
    parser.add_argument(
        "--simulate-failure",
        "-s",
        action="store_true",
        default=False,
        help="Simulate process failure for testing (only affects rank 1)",
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

    # Configure logging with rank info and verbosity level
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format=f"[Rank {rank}/{world_size}] %(levelname)s: %(message)s",
    )

    logger.info(f"Starting advanced ULFM training on rank {rank}/{world_size}")

    # Configure ULFM C++ verbose logging based on CLI argument
    ULFM.set_ulfm_verbose_logging(args.verbose)

    if args.verbose:
        logger.debug(f"CLI arguments: {vars(args)}")
        logger.debug(
            f"ULFM verbose logging enabled: {ULFM.is_ulfm_verbose_logging()}"
        )

    # Set device
    device = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create model
    model = create_simple_model().to(device)
    logger.info(
        f"Model created with {sum(p.numel() for p in model.parameters())} parameters"
    )

    # Create training manager with generalized recovery
    training_manager = ULFMTrainingManager(
        model,
        failure_strategy=args.failure_strategy,
        enable_auto_repair=args.auto_repair,
        verbose=args.verbose,
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
            if args.simulate_failure and rank == 1 and epoch == 3 and batch_idx == 20:
                logger.warning(f"[Rank {rank}] Simulating process failure")
                os.kill(os.getpid(), signal.SIGKILL)

            data, target = data.to(device), target.to(device)

            try:
                loss, recovered = training_manager.train_step(
                    data, target, criterion, optimizer
                )

                if loss is not None:
                    epoch_loss += loss
                    successful_batches += 1

                    if recovered:
                        logger.info(f"Batch {batch_idx} completed after recovery")

                    if batch_idx % 10 == 0:
                        logger.info(
                            f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}"
                        )
                    elif args.verbose and batch_idx % 5 == 0:
                        logger.info(
                            f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss:.6f}"
                        )

            except Exception as e:
                logger.error(f"Batch {batch_idx} failed permanently: {e}")
                # In production, you might want to implement more sophisticated
                # failure handling here (e.g., checkpoint restoration)
                break

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
