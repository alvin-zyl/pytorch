#!/bin/bash -l
#PBS -N ulfm_pytorch
#PBS -A TensorCompress
#PBS -q debug-scaling
#PBS -l select=8:ncpus=64:ngpus=4:mpiprocs=4
#PBS -l walltime=00:10:00
#PBS -l filesystems=home:eagle
#PBS -j oe

# set -euo pipefail

# --- Environment ---
source ~/.bash_profile
ompi2-pytorch

cd /eagle/TensorCompress/$USER/project/pytorch/mpi_ulfm_extension

# --- Run CoLA Nanotron ---
LOGDIR="/eagle/TensorCompress/$USER/project/pytorch/mpi_ulfm_extension/.logging/$(date +%Y%m%d)"
mkdir -p "$LOGDIR"
LOGFILE="${LOGDIR}/hsdp_$(date +%Y%m%d_%H%M%S)_${PBS_JOBID}.log"
echo "LOGFILE=$LOGFILE"

export MASTER_ADDR=$(head -1 "$PBS_NODEFILE")
export MASTER_PORT=$(( RANDOM + 1000 ))

export TMPDIR=/tmp

TORCH_NCCL_DUMP_ON_TIMEOUT=0 NCCL_PG_TIMEOUT_SECONDS=20 \
    mpiexec --with-ft ulfm --map-by ppr:4:node:PE=8 --bind-to core --mca pml ob1 --mca btl tcp,self,sm \
    python main_hsdp.py --backend ulfm --hsdp_shard_size 16 --model_config configs/llama7b.json \
    --batch_size 1 --total_batch_size 32 --num_training_steps 10000 --warmup_steps 1000 \
    --weight_decay 0.1 --grad_clipping 1.0 --dtype bfloat16 --offline_mode --mixed_precision --max_length 4096 \
    --activation_checkpointing \
    --failure_count 1 --failure_step_range 10 20 --failure_locations post-allreduce:1.0 \
    > "$LOGFILE" 2>&1