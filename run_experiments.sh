#!/bin/bash

# Define the arguments for each experiment
declare -a experiments=(
    "--rank 0 --mlflow_run_name exp1 "
    "--enable_galore --rank 2 --mlflow_run_name exp2"
    "--enable_galore --rank 4 --mlflow_run_name exp3"
    "--enable_galore --rank 8 --mlflow_run_name exp4"
    "--enable_galore --rank 16 --mlflow_run_name exp5"
    "--enable_galore --rank 32 --mlflow_run_name exp6"
    "--enable_galore --rank 64 --mlflow_run_name exp7"
    "--enable_galore --rank 128 --mlflow_run_name exp8"

)

# Loop through each experiment and run the script
for exp_args in "${experiments[@]}"; do
    python train_cifar.py --mlflow_experiment_name galore_resnet --epochs 5 --seed 42 --precision medium $exp_args
done