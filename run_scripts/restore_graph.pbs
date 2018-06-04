#!/bin/bash
##PBS -l nodes=16:ppn=32:xe+64:ppn=16:xk
##PBS -l nodes=10:ppn=16:xk
#PBS -l walltime=00:40:00
#PBS -N restore_graph
#PBS -e logs/log.${PBS_JOBNAME}_${PBS_JOBID}.err
#PBS -o logs/log.${PBS_JOBNAME}_${PBS_JOBID}.out

PREVIOUS_JOBNAME=""
PREVIOUS_JOBID=""

echo "Starting"

if [ -z "${PREVIOUS_JOBNAME}" ] || [ -z "${PREVIOUS_JOBID}" ];
then
echo "Please set the enviromeent variables PREVIOUS_JOBNAME and PREVIOUS_JOBID above then rerun this script"
echo "Exiting"
exit 1
fi

cd $PBS_O_WORKDIR
mkdir -p logs

NUM_GPU=$(aprun -n ${PBS_NUM_NODES} -N 1 -- /sbin/lspci | grep NVIDIA | wc -l)
let NUM_PS=${PBS_NUM_NODES}-${NUM_GPU}
NUM_WORKER=${NUM_GPU}
if [ "${NUM_PS}" -eq '0' ];
then
echo "all nodes have a GPU, giving some of them to the PS"
NUM_PS=$((${NUM_GPU}/4))
NUM_WORKER=$((${PBS_NUM_NODES}-${NUM_PS}))
fi

echo "NUM_PS ${NUM_PS}, NUM_WORKER ${NUM_WORKER}"

module load bwpy
module load bwpy-mpi

DATA_DIR="${HOME}/scratch/ImageNet/tf_records"

APOUT_LOGS="${PBS_O_WORKDIR}/logs/apout.${PBS_JOBNAME}_${PBS_JOBID}"
echo "output at ${APOUT_LOGS}.*"
CHECKPT_DIR="checkpoint_dir_${PREVIOUS_JOBNAME}_${PREVIOUS_JOBID}.bw"
echo "restoring checkpoint ${CHECKPT_DIR}"

RUN_CMD="python ${PBS_O_WORKDIR}/../BWDistributedTrain/inception_imagenet_distributed_train.py \
--checkpoint_dir ${CHECKPT_DIR} \
--data_dir $DATA_DIR/train \
--num_train_examples 385455 \
--batch_size 1 \
--restore "
echo "Running Comand"
echo ${RUN_CMD}
aprun -b -cc none -n ${NUM_PS} -N 1 $RUN_CMD --ps_worker ps : -n ${NUM_WORKER} -N 1 $RUN_CMD --ps_worker worker \
1> ${PBS_O_WORKDIR}/logs/${APOUT_LOGS}.out \
2> ${PBS_O_WORKDIR}/logs/${APOUT_LOGS}.err

echo "Done, Thank you for flying."
