import os
import signal
import torch
import ulfm_collectives
import torch.distributed as dist
from mpi4py import MPI


size = MPI.COMM_WORLD.Get_size()
global_rank = MPI.COMM_WORLD.Get_rank()
local_rank = global_rank % 4

torch.cuda.set_device(local_rank)
print(
    f"[Rank {global_rank}] Global rank {global_rank}, local rank {local_rank}, device: {torch.cuda.current_device()}"
)

dist.init_process_group("ulfm")
rank = dist.get_rank()
print(
    f"[Rank {global_rank}] Rank obtained from process group: {rank}"
)

if global_rank == 2:
    print(f"[Rank {global_rank}] simulating failure")
    os.kill(os.getpid(), signal.SIGKILL)

# this goes through gloo
x = torch.ones(6)
dist.all_reduce(x)
print(f"[Rank {rank}] cpu allreduce: {x}")

# this goes through dummy
if torch.cuda.is_available():
    y = x.cuda()
    dist.all_reduce(y)
    print(f"[Rank {rank}] cuda allreduce: {y}")

    # try:
    #     dist.broadcast(y, 0)
    # except RuntimeError:
    #     print("got RuntimeError when calling broadcast")
