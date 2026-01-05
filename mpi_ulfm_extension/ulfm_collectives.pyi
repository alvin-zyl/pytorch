# mypy: allow-untyped-defs
# mypy: disable-error-code="type-arg"
from datetime import timedelta
from enum import Enum
from typing import Any, Optional, Union, Tuple

import torch
from torch import Tensor
from torch._C._distributed_c10d import ProcessGroup, Work, AllreduceOptions, Reducer

# This module is defined in mpi_ulfm_extension/src/bindings.cpp

def createProcessGroupULFM(ranks: list[int]) -> ProcessGroupULFM: ...

class ULFMFailureHandlingStrategy(Enum):
    CONTINUE_WITH_SURVIVORS = ...
    RESTART_FAILED_PROCESSES = ...
    ABORT_ON_FAILURE = ...

class ULFMOptions:
    auto_repair: bool
    failure_strategy: ULFMFailureHandlingStrategy
    max_retries: int
    retry_delay_ms: int
    
    def __init__(self) -> None: ...

class WorkULFM(Work):
    """ULFM-aware Work class for failure detection (no recovery logic)."""
    def has_failures(self) -> bool: 
        """Check if this work detected any failures."""
        ...
    
    def get_failed_ranks(self) -> list[int]: 
        """Get list of ranks that failed during this work."""
        ...
    
    def was_noop(self) -> bool:
        """Check if this work was marked as a no-op due to failures."""
        ...
    
    def markNoop(self) -> None:
        """Mark this work as a no-op (used internally for failure handling)."""
        ...

class ProcessGroupULFM(ProcessGroup):
    def ulfm_allreduce(
        self,
        tensors: list[Tensor],
        opts: AllreduceOptions = ...,
        ulfm_opts: ULFMOptions = ...
    ) -> WorkULFM: ...
    
    # Legacy recovery methods (backward compatibility)
    def repair_communicator(self) -> bool:
        """Legacy method: repair the MPI communicator after failures."""
        ...
    
    def notify_all_ranks_of_failure(self) -> None:
        """Legacy method: notify all ranks of detected failures."""
        ...
    
    def check_for_failures(self) -> bool:
        """Legacy method: check if there are any detected failures."""
        ...
    
    # New modular failure recovery system
    def detect_and_recover_failures(
        self, 
        auto_repair: bool = True
    ) -> Tuple[bool, list[int]]:
        """
        Comprehensive 5-step failure detection and recovery workflow.
        
        Follows ULFM best practices:
        1. Notice failure (comm_agree first)
        2. Get failed ranks (while communicator is revoked)
        3. Ack failures (MUST come before collective operations)
        4. Agree on failed ranks (now safe after ack)
        5. Repair communicator if needed and requested
        
        Args:
            auto_repair: Whether to automatically repair the communicator
        
        Returns:
            Tuple of (success: bool, failed_ranks: list[int])
        """
        ...
    
    # State management methods
    def set_quiesce(self, v: bool) -> None:
        """Set the quiesce state of the process group."""
        ...
    
    def is_quiesced(self) -> bool:
        """Check if the process group is currently quiesced."""
        ...
    
    def worldEpoch(self) -> int:
        """Get the current world epoch (increments after communicator repairs)."""
        ...


# ULFM logging control
def set_ulfm_verbose_logging(verbose: bool) -> None:
    """Enable or disable verbose ULFM logging."""
    ...

def is_ulfm_verbose_logging() -> bool:
    """Check if verbose ULFM logging is enabled."""
    ...
