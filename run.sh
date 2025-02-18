#!bin/bash

WORK_DIR="/c/Users/eugen/Codes/SWIN_parallelism"
. "$WORK_DIR/venv/Scripts/activate"
NRANKS_PER_NODE=4
cd $WORK_DIR

echo SCRIPT_PTH: $SCRIPT_PTH
echo WORK_DIR: $WORK_DIR


torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\test_window_parallelism.py"
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\send_recv_bench.py"
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\train.py"