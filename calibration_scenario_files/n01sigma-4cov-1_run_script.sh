#!/bin/sh
#PBS -N HS_UKI_New_Run_Script
#PBS -A YOUR_ACCOUNT
#PBS -l walltime=00:15:00
#PBS -q main
#PBS -l select=1:ncpus=128
#PBS -M YOUR_EMAIL
#PBS -m abe

# Sample run script for the HS calibration

module load ncarenv/23.09
module load julia/1.9.2
export JULIA_NUM_THREADS=128

# variables to pass to the julia script
prefix="batch_scenario"
batch_size=100
noise_level=0.01
sigma_exponent=-4
covariance_exponent=-1
iters=300

julia hs_cali_run_generic.jl \
    $prefix \
    $batch_size \
    $noise_level \
    $sigma_exponent \
    $covariance_exponent \
    $iters
