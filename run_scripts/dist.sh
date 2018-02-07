#!/bin/bash
cd $PBS_O_WORKDIR

WORKER_HOSTS_TASKS=$1;
PS_HOSTS_TASKS=$2;
DATA_DIR=$3;
TRAIN_DIR=$4;

shift 4

CMD_ARGS=${*}
CMD_LOC="../benchmarks/scripts/tf_cnn_benchmarks/"
MY_HOST_NAME=$(hostname)
WORKER_HOSTS=""
PS_HOSTS=""
WHAT_AM_I=""
MY_TASK_NUMBER=""
WORKER_WAIT_TIME=20

for h in $(echo $WORKER_HOSTS_TASKS | sed "s/,/ /g")
do
    HOST_NAME=$(echo $h | cut -d ':' -f 1)
    WORKER_HOSTS="${HOST_NAME}:2222,${WORKER_HOSTS}"
    if [ ${HOST_NAME} == ${MY_HOST_NAME} ]
    then
	WHAT_AM_I='worker'
	echo "Im a worker, sleeping $WORKER_WAIT_TIME waiting for the parameter server to start up"
	sleep $WORKER_WAIT_TIME
	MY_TASK_NUMBER=$(echo $h | cut -d ':' -f 2)
    fi
done

for h in $(echo $PS_HOSTS_TASKS | sed "s/,/ /g")
do
    HOST_NAME=$(echo $h | cut -d ':' -f 1)
    PS_HOSTS="${HOST_NAME}:2222,${PS_HOSTS}"
    if [ ${HOST_NAME} == ${MY_HOST_NAME} ]
    then
	WHAT_AM_I='ps'
	MY_TASK_NUMBER=$(echo $h | cut -d ':' -f 2)
    fi
done

if [ -z "$WHAT_AM_I" ]; then
    echo "($MY_HOST_NAME) Im the throw away node, exiting gracefully"
    exit 0
fi 

WORKER_HOSTS=$(echo $WORKER_HOSTS | sed 's/,$//')
PS_HOSTS=$(echo $PS_HOSTS | sed 's/,$//')


PY_CMD="${CMD_LOC}/tf_cnn_benchmarks.py \
--ps_hosts ${PS_HOSTS} \
--worker_hosts ${WORKER_HOSTS} \
--job_name ${WHAT_AM_I} \
--task_index ${MY_TASK_NUMBER} \
${CMD_ARGS}"

echo "Hi! I am ${MY_HOST_NAME}. My job is: ${WHAT_AM_I} with task id: ${MY_TASK_NUMBER}. I'm about to run
python ${PY_CMD}"

python $PY_CMD
