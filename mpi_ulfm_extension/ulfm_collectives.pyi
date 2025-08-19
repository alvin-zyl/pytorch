# mypy: allow-untyped-defs
# mypy: disable-error-code="type-arg"
from datetime import timedelta
from enum import Enum
from typing import Any, Optional, Union

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

class ProcessGroupULFM(ProcessGroup):
    def ulfm_allreduce(
        self,
        tensors: list[Tensor],
        opts: AllreduceOptions = ...,
        ulfm_opts: ULFMOptions = ...
    ) -> Work: ...

class ULFMCommHook:
    def __init__(
        self,
        process_group: ProcessGroupULFM,
        failure_strategy: ULFMFailureHandlingStrategy = ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS
    ) -> None: ...
    
    def set_failure_handling_strategy(self, strategy: ULFMFailureHandlingStrategy) -> None: ...
    def get_failure_handling_strategy(self) -> ULFMFailureHandlingStrategy: ...
    def is_communicator_healthy(self) -> bool: ...
    def repair_communicator(self) -> bool: ...

def create_ulfm_hook(
    process_group: ProcessGroupULFM,
    failure_strategy: ULFMFailureHandlingStrategy = ULFMFailureHandlingStrategy.CONTINUE_WITH_SURVIVORS
) -> ULFMCommHook: ...