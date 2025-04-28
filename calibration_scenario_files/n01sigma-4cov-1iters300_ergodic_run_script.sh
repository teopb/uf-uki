#!/bin/sh
#PBS -N HS_UKI_Ergodic_Run_Script
#PBS -A YOUR_ACCOUNT
#PBS -l walltime=00:30:00
#PBS -q main
#PBS -l select=1:ncpus=32
#PBS -M YOUR_EMAIL
#PBS -m abe

module load ncarenv/23.09
module load julia/1.9.2
export JULIA_NUM_THREADS=32

# variables to pass to the julia script
prefix="ergodic_scenario"
batch_size=1
noise_level=0.01
sigma_exponent=1
covariance_exponent=1
iters=50

julia hs_cali_run_generic_ergodic.jl \
    $prefix \
    $batch_size \
    $noise_level \
    $sigma_exponent \
    $covariance_exponent \
    $iters
