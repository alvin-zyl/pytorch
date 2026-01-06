#!/usr/bin/env python3
"""
Advanced ULFM DDP Example with Generalized Failure Recovery

This example demonstrates the new generalized failure recovery system:
1. WorkULFM for per-operation failure detection
2. ProcessGroup-level recovery methods
3. Flexible failure handling strategies
4. Integration with PyTorch DDP training loops
"""

import contextlib
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
from orchestrator import StepTxnOrchestrator
from ulfm_hook import create_ulfm_recovery_hook, HookState
from policy import create_policy, GradRestoreMode

# Logger will be configured based on CLI arguments
logger = logging.getLogger(__name__)


class ULFMTrainingManager:
    """
    Training manager that handles ULFM failure recovery in a structured way.

    Now policy-driven: delegates fault tolerance decisions to a FaultTolerancePolicy.
    """

    def __init__(
        self,
        model,
        grad_accum_steps=1,
        failure_strategy="continue",
        enable_auto_repair=True,
        policy_type="adaptive",  # "adaptive" or "fixed"
    ):
        self.failure_strategy = failure_strategy
        self.process_group = dist.group.WORLD

        # Verify we have ULFM process group
        if not isinstance(self.process_group, ULFM.ProcessGroupULFM):
            raise ValueError("Must use ULFM backend: dist.init_process_group('ulfm')")

        # Create fault tolerance policy
        policy = create_policy(
            policy_type=policy_type,
            initial_grad_accum_steps=grad_accum_steps,
            enable_auto_repair=enable_auto_repair,
        )

        # Create orchestrator for gradient management (with policy)
        rank = dist.get_rank()
        self.txn = StepTxnOrchestrator(rank=rank, pg=self.process_group, policy=policy)

        # Create DDP model with ULFM hook
        self.ddp_model = DistributedDataParallel(model)
        self._register_ulfm_hook()

        self._micro_in_window = 0  # counts [0..K-1] within an accumulation window

        logger.debug(
            f"[Rank {rank}] ULFMTrainingManager initialized with policy={policy_type}, "
            f"auto_repair={enable_auto_repair}, strategy={failure_strategy}"
        )

    def _register_ulfm_hook(self):
        """Register ULFM communication hook with recovery logic."""
        hstate = HookState(
            pg=self.process_group, orchestrator=self.txn
        )
        hook = create_ulfm_recovery_hook(
            failure_strategy=self.failure_strategy
        )
        self.ddp_model.register_comm_hook(state=hstate, hook=hook)
        logger.debug(
            f"[Rank {self.txn._rank}] ULFM hook registered with policy type: {type(self.policy).__name__}"
        )

        # Store hook for statistics access
        self._hook = hook

    @property
    def policy(self):
        """Get the fault tolerance policy from orchestrator."""
        return self.txn.policy

    def _get_grad_accum_steps(self):
        """Get current gradient accumulation steps from policy."""
        return self.txn.policy.grad_accum_steps

    def _notify_window_start(self):
        """Notify policy that a new accumulation window is starting."""
        self.policy.on_window_start()
        self.txn.reset_hook_counter()

    def _on_microbatch_complete(self, microbatch_idx):
        """Notify policy that a microbatch completed and get decision."""
        return self.policy.on_microbatch_complete(microbatch_idx)

    def _in_last_micro(self):
        return (self._micro_in_window + 1) == self._get_grad_accum_steps()
    
    def _get_restore_mode(self):
        return self.txn.get_restore_plan()
    
    def _start_restore_gradients_non_blocking(self):
        self.txn.restore_gradients_non_blocking()
    
    def _wait_restore_before_backward(self):
        self.txn.wait_restore_before_backward()

    def _should_skip_step(self):
        return self.txn.should_skip_step()
    
    def _on_step_skipped(self):
        self.txn.on_step_skipped()

    def _start_restore_gradients_blocking(self):
        self.txn.restore_gradients_blocking()
    
    def _on_step_committed(self):
        self.txn.after_successful_commit()

    def train_step(self, batch_idx, data, target, criterion, optimizer, scaler=None):
        """
        One microbatch step; handles grad-accum windows and ULFM healing.
        Now policy-driven: consults policy for all fault tolerance decisions.

        Restoration logic (after failure detected by hook):
        (a) At POLICY boundary (policy.on_failure returns at_iteration_boundary=True):
            - Skip current optimizer step (need_extra_microbatch=True)
            - Add one more grad acc step as "extended step"
            - During forward of extended step: async restore corrupted gradients
            - Wait for restoration, do backward, then optimizer step

        (b) NOT at policy boundary (at_iteration_boundary=False):
            - Continue current iteration normally
            - Before optimizer step: restore gradients blockingly if needed
            - Then do optimizer step

        Returns: (loss_value, stepped_bool)
        """
        dp_world = dist.get_world_size(self.process_group)
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

        # === Forward ===
        output = self.ddp_model(data)

        # === Wait for async restoration if it was started ===
        if restore_mode == GradRestoreMode.NON_BLOCKING:
            self._wait_restore_before_backward()
            logger.debug(
                f"[Rank {self.txn._rank}] Non-blocking restoration completed before backward"
            )

        # === Backward (use no_sync on non-last microbatches) ===
        ctx = (
            self.ddp_model.no_sync()
            if not self._in_last_micro()
            else contextlib.nullcontext()
        )
        with ctx:
            loss = criterion(output, target)
            if scaler is None:
                loss.backward()
            else:
                scaler.scale(loss).backward()

        # === After backward: check with policy if we should commit ===
        state = self._on_microbatch_complete(self._micro_in_window)
        restore_mode = self._get_restore_mode()

        stepped = False
        pending_skip = self._should_skip_step()

        # === Decide whether to commit optimizer step ===
        if state.at_iteration_boundary:
            if pending_skip:
                logger.info(
                    f"[Rank {self.txn._rank}] Skipping optimizer step per recovery plan"
                )
                self._on_step_skipped()
                return float(loss.detach()), False

            # At window boundary: time to commit

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
                        p.grad.div_(dp_world)
                optimizer.step()
            else:
                if hasattr(scaler, "unscale_"):
                    scaler.unscale_(optimizer)
                for p in self.ddp_model.parameters():
                    if p.grad is not None:
                        p.grad.div_(dp_world)
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

        return float(loss.detach()), stepped

    def get_recovery_stats(self):
        """Get failure and recovery statistics from policy."""
        # Statistics are now tracked centrally in the policy object
        return self.policy.get_stats()


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
            if args.simulate_failure and rank == 1 and epoch == 1 and batch_idx == 2:
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
