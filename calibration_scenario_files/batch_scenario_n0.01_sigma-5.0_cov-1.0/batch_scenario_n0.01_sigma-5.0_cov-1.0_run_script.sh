#!/bin/sh
#PBS -N HS_UKI_Postrun_batch_scenario_n0.01_sigma-5.0_cov-1.0
#PBS -A your_account
#PBS -l walltime=00:15:00
#PBS -q main
#PBS -l select=1:ncpus=128
#PBS -M your_email@colorado.edu
#PBS -m abe

module load ncarenv/23.09
module load julia/1.9.2
export JULIA_NUM_THREADS=128
julia your_path/UF_UKI/calibration_scenario_files/batch_scenario_n0.01_sigma-5.0_cov-1.0/batch_scenario_n0.01_sigma-5.0_cov-1.0_postrun_julia_script.jl
