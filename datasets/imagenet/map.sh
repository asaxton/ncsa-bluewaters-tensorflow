#!/bin/bash

if [ -z "${1}" ]
then
echo "Usage <my_script.sh> <total num of nodes> [list of tokens]"
exit 1
fi

MY_SCRIPT=${1}
shift
NUM_DIV=${1}
shift

if [ ! -z "$*" ]
then
TOT_LIST=($*)
else
read tmp
TOT_LIST=($tmp)
fi

let NUM_LIST=${#TOT_LIST[@]}
let r=$NUM_DIV-1
let step_block=$NUM_LIST/$NUM_DIV
let step_remainder=$NUM_LIST%$NUM_DIV

b=0
a=0

index=$ALPS_APP_PE

if [ "$index" -ge "$step_remainder" ]
then
let exes=$index-$step_remainder
let a=$step_block*$step_remainder+$step_remainder+$step_block*exes
let b=$step_block
else
let a=$step_block*$index+$index
let b=$step_block+1
fi

LIST_SLICE="${TOT_LIST[@]:$a:$b}"

echo "Starting $MY_SCRIPT" >> $LOG_DIR/${PBS_JOBNAME}_map_index_${index}_of_${NUM_DIV}_jobid_${PBS_JOBID}.out
echo "Starting $MY_SCRIPT" >> $LOG_DIR/${PBS_JOBNAME}_map_index_${index}_of_${NUM_DIV}_jobid_${PBS_JOBID}.err

./${MY_SCRIPT} $LIST_SLICE 1>> $LOG_DIR/${PBS_JOBNAME}_map_index_${index}_of_${NUM_DIV}_jobid_${PBS_JOBID}.out \
    2>> $LOG_DIR/${PBS_JOBNAME}_map_index_${index}_of_${NUM_DIV}_jobid_${PBS_JOBID}.err

echo "Done $MY_SCRIPT" >> $LOG_DIR/${PBS_JOBNAME}_map_index_${index}_of_${NUM_DIV}_jobid_${PBS_JOBID}.out
echo "Done $MY_SCRIPT" >> $LOG_DIR/${PBS_JOBNAME}_map_index_${index}_of_${NUM_DIV}_jobid_${PBS_JOBID}.err
