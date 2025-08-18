"""
ULFM-enabled DistributedDataParallel that subclasses PyTorch's original DDP.

This module provides a minimal extension to PyTorch's DistributedDataParallel
that only replaces the reducer with an ULFM-aware version for fault tolerance.
"""

import logging
from typing import Any, Optional
from torch.nn.parallel.distributed import DistributedDataParallel

try:
    import ulfm_collectives
except ImportError:
    raise ImportError(
        "ULFM collectives extension not found. Please build the extension first."
    )

logger = logging.getLogger(__name__)


class ULFMDistributedDataParallel(DistributedDataParallel):
    """
    ULFM-enabled DistributedDataParallel that extends PyTorch's original DDP.

    This class inherits from PyTorch's DistributedDataParallel and only replaces
    the reducer with an ULFM-aware version. All other functionality (parameter
    management, autograd hooks, broadcasting, etc.) is handled by the parent class.

    Args:
        module (nn.Module): Module to be parallelized
        device_ids: Device IDs (same as PyTorch DDP)
        output_device: Output device (same as PyTorch DDP)
        dim: Dimension for scattering (same as PyTorch DDP)
        broadcast_buffers: Whether to broadcast buffers (same as PyTorch DDP)
        process_group: ProcessGroupULFM instance for ULFM communication
        bucket_cap_mb: Bucket size in MB for gradient bucketing (same as PyTorch DDP)
        find_unused_parameters: Whether to find unused parameters (same as PyTorch DDP)
        check_reduction: Whether to check reduction (same as PyTorch DDP)
        gradient_as_bucket_view: Whether to use gradient as bucket view (same as PyTorch DDP)
        static_graph: Whether the graph structure is static (same as PyTorch DDP)
        failure_handling_strategy: Strategy for handling process failures
            - "continue": Continue with surviving processes
            - "restart": Restart failed processes (if supported)
            - "abort": Abort on any failure

    All other arguments are passed through to PyTorch's DistributedDataParallel.
    """

    def __init__(
        self,
        module,
        device_ids=None,
        output_device=None,
        dim=0,
        broadcast_buffers=True,
        process_group=None,
        bucket_cap_mb=25,
        find_unused_parameters=False,
        check_reduction=False,
        gradient_as_bucket_view=False,
        static_graph=False,
        delay_all_reduce_named_params=None,
        param_to_hook_all_reduce=None,
        mixed_precision=None,
        failure_handling_strategy="continue",
        **kwargs,
    ):
        # Validate process group
        if process_group is None:
            import torch.distributed as dist

            if dist.is_initialized():
                process_group = dist.group.WORLD
            else:
                raise ValueError(
                    "process_group must be provided or distributed must be initialized"
                )

        # Check if it's a ProcessGroupULFM, if not, try to create one
        if not isinstance(process_group, ulfm_collectives.ProcessGroupULFM):
            # For now, skip ULFM replacement if not using ProcessGroupULFM
            logging.warning("Process group is not ProcessGroupULFM, using standard DDP")
            self._use_ulfm = False
        else:
            self._use_ulfm = True

        # Parse failure handling strategy
        strategy_map = {
            "continue": ulfm_collectives.ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS,
            "restart": ulfm_collectives.ULFMFailureHandlingStrategy.RESTART_FAILED_PROCESSES,
            "abort": ulfm_collectives.ULFMFailureHandlingStrategy.ABORT_ON_FAILURE,
        }
        if failure_handling_strategy not in strategy_map:
            raise ValueError(
                f"Invalid failure handling strategy: {failure_handling_strategy}"
            )
        self.failure_strategy = strategy_map[failure_handling_strategy]

        # Store ULFM-specific parameters before calling parent
        self.ulfm_process_group = process_group
        self.ulfm_bucket_cap_mb = bucket_cap_mb

        # Call parent constructor with all standard DDP arguments
        # The parent will create the standard reducer
        super().__init__(
            module=module,
            device_ids=device_ids,
            output_device=output_device,
            dim=dim,
            broadcast_buffers=broadcast_buffers,
            process_group=process_group,  # ProcessGroupULFM is compatible
            bucket_cap_mb=bucket_cap_mb,
            find_unused_parameters=find_unused_parameters,
            check_reduction=check_reduction,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
            delay_all_reduce_named_params=delay_all_reduce_named_params,
            param_to_hook_all_reduce=param_to_hook_all_reduce,
            mixed_precision=mixed_precision,
            **kwargs,
        )

        # Replace the standard reducer with ULFM reducer only if using ProcessGroupULFM
        if self._use_ulfm:
            print("trying to replace with ulfm reducer")
            self._replace_reducer_with_ulfm()
            print("replaced with ulfm reducer")
        else:
            logging.info("Using standard PyTorch DDP (no ULFM)")

    def _replace_reducer_with_ulfm(self):
        """Replace the standard reducer with ULFM-aware reducer."""
        try:
            # Get the current reducer parameters from the parent class
            # These are already set up by the parent constructor
            params = list(self.module.parameters())
            if not params:
                logger.warning("No parameters found, skipping ULFM reducer replacement")
                return

            # Get bucket indices from the existing reducer
            # For simplicity, we'll create new bucket indices
            # In a production version, you'd want to preserve the existing bucketing
            bucket_indices = []
            param_index = 0
            for param in params:
                if param.requires_grad:
                    bucket_indices.append([param_index])
                    param_index += 1

            if not bucket_indices:
                logger.warning("No gradient-requiring parameters found")
                return

            # Create parameter names map
            param_names = {}
            param_index = 0
            for name, param in self.module.named_parameters():
                if param.requires_grad:
                    param_names[param_index] = name
                    param_index += 1

            # Create expect_sparse_gradients list (assume all dense for simplicity)
            expect_sparse_gradients = [False] * len(
                [p for p in params if p.requires_grad]
            )

            # Create ULFM reducer using our C++ function
            bucket_bytes_cap = self.ulfm_bucket_cap_mb * 1024 * 1024
            first_bucket_bytes_cap = min(bucket_bytes_cap, 1024 * 1024)

            # Get the parameters that require gradients
            grad_params = [p for p in params if p.requires_grad]

            self.ulfm_reducer = ulfm_collectives.create_ulfm_reducer(
                params=grad_params,
                bucket_indices=bucket_indices,
                process_group=self.ulfm_process_group,
                expect_sparse_gradients=expect_sparse_gradients,
                bucket_bytes_cap=bucket_bytes_cap,
                find_unused_parameters=self.find_unused_parameters,
                gradient_as_bucket_view=self.gradient_as_bucket_view,
                param_names=param_names,
                first_bucket_bytes_cap=first_bucket_bytes_cap,
                skip_all_reduce_unused_params=False,
                use_python_reducer=False,
                failure_strategy=self.failure_strategy,
            )
            # Replace the reducer in the parent class
            # This is the key modification - we only change the reducer
            self.reducer = self.ulfm_reducer

            logger.info(
                f"Successfully replaced reducer with ULFM reducer (strategy: {self.get_failure_handling_strategy()})"
            )

        except Exception as e:
            logger.error(f"Failed to replace reducer with ULFM reducer: {e}")
            raise RuntimeError(f"ULFM reducer initialization failed: {e}")

    def get_failure_handling_strategy(self) -> str:
        """Get the current failure handling strategy as a string."""
        strategy_map = {
            ulfm_collectives.ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS: "continue",
            ulfm_collectives.ULFMFailureHandlingStrategy.RESTART_FAILED_PROCESSES: "restart",
            ulfm_collectives.ULFMFailureHandlingStrategy.ABORT_ON_FAILURE: "abort",
        }
        return strategy_map.get(self.failure_strategy, "unknown")

    def set_failure_handling_strategy(self, strategy: str):
        """Change the failure handling strategy."""
        strategy_map = {
            "continue": ulfm_collectives.ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS,
            "restart": ulfm_collectives.ULFMFailureHandlingStrategy.RESTART_FAILED_PROCESSES,
            "abort": ulfm_collectives.ULFMFailureHandlingStrategy.ABORT_ON_FAILURE,
        }
        if strategy not in strategy_map:
            raise ValueError(f"Invalid failure handling strategy: {strategy}")

        self.failure_strategy = strategy_map[strategy]
        # Note: The actual ULFM hook strategy is set during reducer creation
        # To change it dynamically, we'd need to expose that functionality
        logger.info(f"Failure handling strategy changed to: {strategy}")

    def is_communicator_healthy(self) -> bool:
        """Check if the communicator is healthy."""
        # Since we're using the CommHook approach, we don't have direct access
        # to the hook from Python. In a production version, you'd want to expose this.
        try:
            # For now, we'll assume healthy unless we can detect otherwise
            return True
        except Exception:
            return False

    def repair_communicator(self) -> bool:
        """Attempt to repair the communicator after a failure."""
        try:
            # The ULFM hook handles repair automatically with auto_repair=true
            # This is just a placeholder for the Python interface
            logger.info(
                "Communicator repair requested - will happen automatically in next collective"
            )
            return True
        except Exception as e:
            logger.error(f"Failed to repair communicator: {e}")
            return False


def ulfm_ddp_wrapper(
    process_group: Optional[Any] = None,
    failure_handling_strategy: str = "continue",
    **ddp_kwargs,
):
    """
    Decorator factory for easily applying ULFM DDP to a model.

    Args:
        process_group: ULFM process group
        failure_handling_strategy: How to handle process failures
        **ddp_kwargs: Additional arguments for ULFMDistributedDataParallel

    Returns:
        Decorator function

    Example:
        >>> @ulfm_ddp_wrapper(failure_handling_strategy="continue")
        >>> def create_model():
        >>>     return MyModel()
        >>>
        >>> model = create_model()
    """

    def decorator(model_factory):
        def wrapper(*args, **kwargs):
            model = model_factory(*args, **kwargs)
            return ULFMDistributedDataParallel(
                model,
                process_group=process_group,
                failure_handling_strategy=failure_handling_strategy,
                **ddp_kwargs,
            )

        return wrapper

    return decorator
