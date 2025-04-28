#!/bin/sh
#PBS -N uki_ergodic_param_test
#PBS -A YourAccount
#PBS -l walltime=12:00:00
#PBS -q main
#PBS -l select=1:ncpus=128
#PBS -M your_email
#PBS -m a

module load ncarenv/23.09
module load julia/1.9.2
export JULIA_NUM_THREADS=16

julia uki_ergodic_param_test.jl true