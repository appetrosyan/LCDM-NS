#!/bin/bash

#SBATCH --job-name=CobayaTestRun
#SBATCH --output=Cosmology/test.out.%j
#SBATCH --time=0:00:01
#SBATCH --mem=1mb

module load python/3.7

source ~/.virtualenvs/py37/bin/activate

srun ~/.virtualenvs/py37/bin/cobaya-run ~/run.yaml -m ~/CosmologyRuns/

wait
