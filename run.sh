#!bin/bash

WORK_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
# . "$WORK_DIR/venv/Scripts/activate"
NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=4
NRANKS=$((NNODES * NRANKS_PER_NODE))
echo WORK_DIR: $WORK_DIR
# cd $WORK_DIR

# echo SCRIPT_PTH: $SCRIPT_PTH
# echo WORK_DIR: $WORK_DIR


## Torchrun
torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR/test_window_parallelism.py"
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR/send_recv_bench.py"
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR/test.py"

## MPIEXEC
# export MASTER_ADDR=$(hostname)
# export MASTER_PORT=27777
# mpiexec -n $NRANKS -ppn $NRANKS_PER_NODE \
#     python "/eagle/datascience/eku/swin-parallelism/send_recv_bench.py"