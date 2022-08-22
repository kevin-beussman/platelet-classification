#!/bin/bash
#SBATCH --job-name=platelet_tf
#SBATCH --account=stf

##SBATCH --partition=gpu-2080ti
#SBATCH --partition=ckpt
##SBATCH --partition=compute

#SBATCH --gres=gpu:1
##SBATCH --gres=gpu:2080ti:1
##SBATCH --gres=gpu:a40:1

#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=32GB

#SBATCH --time=6:00:00

#SBATCH --chdir='./output2022-08-07'

# You can get email notifications when your job starts and stops - useful for long running jobs
#SBATCH --mail-user=beussk@uw.edu
#SBATCH --mail-type=ALL

module load singularity

singularity exec --nv /gscratch/stf/beussk/tf_kb2.sif python ~/platelet_factin_tf/platelet_factin_tf.py # > ./output-${SLURM_JOBID}.log

exit 0
