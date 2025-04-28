#!/bin/sh
#PBS -N initial_bind_HS_UKI_Postrun_scenario_name
#PBS -A YourAccount
#PBS -l walltime=00:30:00
#PBS -q main
#PBS -l select=20:ncpus=128
#PBS -M your_email@colorado.edu
#PBS -m a

module load ncarenv/23.09
module load julia/1.9.2
export JULIA_NUM_THREADS=128

node_count=20
# print out contents of the nodefile
# cat $PBS_NODEFILE

nodes=( $(uniq $PBS_NODEFILE) )

# Create placeholder variable names to be replaced when creating a scenario
case_name="scenario_name"
case_directory="proj_case_dir"
sigma_n=9
batch_n=batch_size

# submit jobs in groups of 20 (there are sigma_n * batch_n total jobs)

total_jobs=$((sigma_n * batch_n))
submit_count=0

while [ $submit_count -lt $total_jobs ]
do
    for i in $(seq 1 $node_count)
    do
        # calculate the sigma and batch number for the current job
        # there should be sigma_n batches of batch_n jobs
        sigma=$((submit_count / batch_n + 1))
        batch=$((submit_count % batch_n + 1))

        # Check if sigma or batch is out of range continue to next job otherwise create a case name and submit the job
        if [ $sigma -gt $sigma_n ] || [ $batch -gt $batch_n ]; then
            continue
        else
            # increment submit_count
            submit_count=$((submit_count + 1))

            # Create a case name
            full_case_name="hs_uki_case_${sigma}_${batch}"
            # BIND_NODE=${nodes[$i]}
            # export BIND_NODE
            # echo ${nodes[$i]}
            (export BIND_NODE=${nodes[$((i-1))]} && cd ${case_directory}/${full_case_name} && ./case.submit --no-batch) &  # Added & to run in background
        fi
    done
    wait  # Wait for all background processes to complete before starting next batch    
done

# Submit the the run script for next iteration
qsub /glade/work/teopb/hs_uq/uki_v4/calibration_scenario_files/scenario_name/scenario_name_run_script_binded.sh

