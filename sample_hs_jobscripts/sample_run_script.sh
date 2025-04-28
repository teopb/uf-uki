#!/bin/sh
#PBS -N HS_UKI_Postrun_scenario_name
#PBS -A YourAccount
#PBS -l walltime=00:15:00
#PBS -q main
#PBS -l select=1:ncpus=128
#PBS -M your_email@colorado.edu
#PBS -m a

module load ncarenv/23.09
module load julia/1.9.2
export JULIA_NUM_THREADS=128
julia /glade/work/teopb/hs_uq/uki_v4/calibration_scenario_files/scenario_name/scenario_name_postrun_julia_script.jl