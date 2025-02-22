#!bin/bash
WORK_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=4
NRANKS=$((NNODES * NRANKS_PER_NODE))
cd $WORK_DIR

## MPIEXEC
export MASTER_ADDR=$(hostname)
export MASTER_PORT=27777
mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} \
    python "${WORK_DIR}/send_recv_bench.py"

## TORCHRUN
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\test_window_parallelism.py"
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\send_recv_bench.py"