#!/bin/bash

# script to set the environment variables for the session
# run this script as source init_env.sh

echo "set environment variables"

export USER_PATH="/home/rr568/NPT/non-parametric-transformers"
export LOG_PATH="/share/kuleshov/richras/imputation/logs"
export DATA_IN="/home/rr568/deep_pop_gen/data_in"
export CLEANED_CHR20_SNPS="/share/kuleshov/ian/data_in/chr20_10k/cleaned_bak"
export DATA_OUT="/share/kuleshov/richras/imputation/data_out"
export SCRATCH_DATA="/scratch/rr568/datasets/"
export ENV_NAME='npt_sklearn'
export USERNAME='rr568'
CONDA_BASE=$(conda info --base)
echo $CONDA_BASE
source $CONDA_BASE/etc/profile.d/conda.sh
conda activate $ENV_NAME
echo "All done"

# define functions for ad hoc use on the terminal 
json()
{
    # call the function as json params.json
    cat $1 | jq | less
}

csv()
{
    # call the function as csv data.csv
    cat $1 | column -t -s, | less -S

}