#!/bin/bash

### set the number of processing elements (PEs) or cores
### set the number of PEs per node
#PBS -l nodes=16:ppn=1:xk
### set the wallclock time
#PBS -l walltime=02:00:00
### set the job name
#PBS -N BuildImageNetTFRecords
### set the job stdout and stderr
#PBS -e logs/log.${PBS_JOBNAME}_${PBS_JOBID}.err
#PBS -o logs/log.${PBS_JOBNAME}_${PBS_JOBID}.out
### set email notification
##PBS -m bea
##PBS -M nowhere@illinois.edu
### In case of multiple allocations, select which one to charge
##PBS -A xyz

# NOTE: lines that begin with "#PBS" are not interpreted by the shell but ARE 
# used by the batch system, wheras lines that begin with multiple # signs, 
# like "##PBS" are considered "commented out" by the batch system 
# and have no effect.  

# If you launched the job in a directory prepared for the job to run within, 
# you'll want to cd to that directory
# [uncomment the following line to enable this]
cd $PBS_O_WORKDIR

# Alternatively, the job script can create its own job-ID-unique directory 
# to run within.  In that case you'll need to create and populate that 
# directory with executables and perhaps inputs
# [uncomment and customize the following lines to enable this behavior] 
# mkdir -p /scratch/sciteam/$USER/$PBS_JOBID
# cd /scratch/sciteam/$USER/$PBS_JOBID
# cp /scratch/job/setup/directory/* .

# To add certain modules that you do not have added via ~/.modules 
#. /opt/modules/default/init/bash
#module load craype-hugepages2M  perftools

### launch the application
### redirecting stdin and stdout if needed
### NOTE: (the "in" file must exist for input)

module load bwpy/0.3.0
module load tensorflow

echo "Starting"
MODE="train"
#MODE="validation"


RAW_DATA_DIR="${HOME}/scratch/ImageNet/raw-imagenet/$MODE"
DATA_LIST_FILE="${MODE}_files.txt"
META_DATA_FILE="$PBS_O_WORKDIR/../../models/research/inception/inception/data/imagenet_metadata.txt"
OUTPUT_DIR="${HOME}/scratch/ImageNet/tf_records/${MODE}"
mkdir -p ${HOME}/scratch/ImageNet
mkdir -p ${HOME}/scratch/ImageNet/tf_records

LOG_DIR="logs/build_imagenet_data_${PBS_JOBID}"

# make sure you qsub this script from the same directoy
# that build_imagenet_data.pbs is in.
NUM_DIV=$(cat $PBS_NODEFILE | wc -l)

RUN_CMD="python build_imagenet_data.py"
RUN_ARGUMENTS="--name ${MODE} \
--data_dir $RAW_DATA_DIR \
--output_directory $OUTPUT_DIR \
--imagenet_metadata_file $META_DATA_FILE \
--data_list_file $DATA_LIST_FILE \
--shards 1024 \
--proc_tot $NUM_DIV \
--distributed"

let r=$NUM_DIV-1

TOT_LENGTH=$(wc -l $DATA_LIST_FILE | cut -d " " -f1)

mkdir -p $LOG_DIR

echo "Removing existing output directory and its contents: ${OUTPUT_DIR}"
rm -rf $OUTPUT_DIR
mkdir $OUTPUT_DIR

echo "Done, Removing existing output directory and its contents: ${OUTPUT_DIR}"

BB_RUN_SCRIPT=do_bb.sh
TF_RUN_SCRIPT=do_tf.sh

cat <<EOF>$BB_RUN_SCRIPT
#!/bin/bash

set -e

index=\$ALPS_APP_PE
echo "\$index: starting $RUN_SCRIPT"

SPACING_ARG="--proc_index \$index"

RUN_BBOX="$RUN_CMD $RUN_ARGUMENTS \$SPACING_ARG --gen_bbox_store"

echo "Starting BBox">  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.err
echo "Starting BBox">  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.out

\$RUN_BBOX 1>> $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.out \\
2>> $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.err

echo "done BBox" >>  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.err
echo "done BBox" >>  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.out

EOF

chmod u=rwx $BB_RUN_SCRIPT

echo "Doing aprun -b -n 16 -N 1 -- $BB_RUN_SCRIPT "
aprun -b -n 16 -N 1 -- $BB_RUN_SCRIPT

cat <<EOF>$TF_RUN_SCRIPT
#!/bin/bash

set -e

index=\$ALPS_APP_PE
echo "\$index: starting $RUN_SCRIPT"

SPACING_ARG="--proc_index \$index"

echo "Starting tf_record" >>  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.err
echo "Starting tf_record" >>  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.out

RUN_TF_RECORD="$RUN_CMD $RUN_ARGUMENTS \$SPACING_ARG"

\$RUN_TF_RECORD 1>> $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.out \\
2>> $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.err

echo "done tf_record" >>  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.err
echo "done tf_record" >>  $LOG_DIR/\${PBS_JOBNAME}_index_\${index}_of_${NUM_DIV}_jobid_\${PBS_JOBID}.out
EOF


chmod u=rwx $TF_RUN_SCRIPT

echo "Doing aprun -b -n 16 -N 1 -- $TF_RUN_SCRIPT "
aprun -b -n 16 -N 1 -- $TF_RUN_SCRIPT

echo "Done, thank you for flying."

### For more information see the man page for aprun
