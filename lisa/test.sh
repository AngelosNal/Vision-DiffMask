JOB_FILE=$(realpath "$0")
HPARAMS_FILE=$(pwd)/hparams.txt
# CHECKPOINTDIR=$(pwd)/array_job_${SLURM_ARRAY_JOB_ID}

echo $(head -$SLURM_ARRAY_TASK_ID $HPARAMS_FILE | tail -1)
