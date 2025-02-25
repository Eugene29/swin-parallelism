#!bin/bash -x
#PBS -l select=16
#PBS -l place=scatter
#PBS -l walltime=00:30:00
#PBS -q lustre_scaling
#PBS -A datascience
#PBS -l filesystems=home:flare
#PBS -k doe
#PBS -j oe
#PBS -N send_recv_bench
#PBS -o /flare/Aurora_deployment/eku/swin-parallelism/o/
#PBS -e /flare/Aurora_deployment/eku/swin-parallelism/e/


## MPIEXEC
WORK_DIR=$(dirname ${BASH_SOURCE[0]} | xargs realpath)
WORK_DIR="/flare/Aurora_deployment/eku/swin-parallelism"
. ${WORK_DIR}/venv/bin/activate
NNODES=$(wc -l < $PBS_NODEFILE)
NRANKS_PER_NODE=12
NRANKS=$((NNODES * NRANKS_PER_NODE))
cd $WORK_DIR

export MASTER_ADDR=$(hostname)
export MASTER_PORT=$RANDOM
export CCL_TOPO_FABRIC_VERTEX_CONNECTION_CHECK=0
mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} \
    python "${WORK_DIR}/test_window_parallelism.py" |& tee "${WORK_DIR}/run.log"

# mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} \
#     python "${WORK_DIR}/vaino_pingpong.py"
    # python "${WORK_DIR}/send_recv_bench.py"


## TORCHRUN
# PYTHON_FILE="/c/Users/eugen/Codes/swin_parallelism/test_window_parallelism.py"
# torchrun --nproc-per-node 12 ${PYTHON_FILE}
# torchrun --nproc-per-node $NRANKS_PER_NODE "$WORK_DIR\send_recv_bench.py"