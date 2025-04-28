# To be run before calibration to create case duplicates and parameter files.
# export JULIA_NUM_THREADS=128
# qcmd -l walltime=00:15:00 -v JULIA_NUM_THREADS -- julia hs_cali_run_par_sen5_n001.jl

include("../UKI.jl")
include("../held_suarez_uki_funcs.jl")
using LinearAlgebra

# parse command line arguments
prefix = ARGS[1]
batch_size = parse(Int64, ARGS[2])
noise_level = parse(Float64, ARGS[3])
sigma_exponent = parse(Float64, ARGS[4])
covariance_exponent = parse(Float64, ARGS[5])
iters = parse(Int64, ARGS[6])
case_prefix = "$(prefix)_n$(noise_level)_sigma$(sigma_exponent)_cov$(covariance_exponent)"

#make directory for script files with case prefix
script_parent_dir = "your_path/UF_UKI/calibration_scenario_files"
script_dir = script_parent_dir * "/" * case_prefix
script_sample_dir = "../sample_hs_jobscripts"

#make directory for script files with case prefix
mkpath(script_dir)

# held suarez parameters
CAM_dir = "your_path/CAM_hs_nl_bind"
base_case = "your_path/CESM_cases/HS_short_test"
ground_truth_output_dir = "your_path/HS_ground_truth"
ground_truth_name = "HS_ground_truth"
proj_case_dir = "your_path/$(case_prefix)_hs_uki_cases"
scratch_case_dir = "your_path/$(case_prefix)_hs_uki_cases"
# postrun_script = script_dir * "/$(case_prefix)_postrun_bash_script.sh"
postrun_script = ""
ground_truth_length_days = 401
ground_truth_spinup_days = 1
time_step_secs = 1200
short_run_nsteps = 9
param_names = ["held_suarez_efolda", "held_suarez_efolds", "held_suarez_delta_T_y", "held_suarez_delta_theta_z"]
theta_zero = [2.0, 2.0, 20.0, 10.0]
C_zero = [1, 1, 1, 1] * 10.0^(covariance_exponent)

# generate calibration scenario scripts
generate_calibration_scenario_scripts(script_sample_dir, case_prefix, script_dir, batch_size, noise_level, sigma_exponent, covariance_exponent, proj_case_dir=proj_case_dir)

# Make hs_param object
hs_params = HS_params(CAM_dir, base_case,ground_truth_output_dir, ground_truth_name, proj_case_dir,scratch_case_dir, postrun_script, ground_truth_length_days, ground_truth_spinup_days, time_step_secs, short_run_nsteps, param_names, theta_zero, C_zero, batch_size)

# Display Parameters
display(hs_params)
# display("Parameter Names:")
# display(hs_params.param_names)

# Make directories if they don't exist 
mkpath(proj_case_dir)
mkpath(scratch_case_dir)

# Write out HS params (object, not the model run parameters)
write_HS_params(hs_params, proj_case_dir * "/hs_params_$(case_prefix).jld2")

#create new cases
# check if cases already exist
if isdir(proj_case_dir * "/hs_uki_case_1_2")
    println("Cases already exist")
else
    println("Creating new cases")
    gen_HS_cases(hs_params)
end

# Check if run complete file exists and delete it
if isfile(proj_case_dir * "/$(case_prefix)_complete.txt")
    rm(proj_case_dir * "/$(case_prefix)_complete.txt")
end

# UKI parameters
ground_truth_filename = "HS_ground_truth.cam.h1i.0001-01-01-10800.nc"
ground_truth_filepath = ground_truth_output_dir * "/run/" * ground_truth_filename
toss_timeslices_count = toss_timeslices(hs_params)
noise_magnitude=noise_level
ground_truth = read_HS_T_lat_sigma_output(ground_truth_filepath, toss = toss_timeslices_count, noise_magnitude = noise_magnitude)

obs_func = nothing

transform_func_to_HS = uki_to_hs_transform

uki_iters = iters

gamma = 1

prior_theta = [2, 2, 20, 20]
trans_prior_theta = [1, 1.5, 20, 20]

m_0 = trans_prior_theta
C_0 = diagm(ones(size(trans_prior_theta, 1))) * 10.0^(covariance_exponent)

Sigma_eta = diagm(ones(size(ground_truth, 1))) * 10.0^(sigma_exponent)

model_func_1 = test_par_setup_HS_cases
model_func_2 = read_HS_cases

param_save_filepath = proj_case_dir * "/uki_params_$(case_prefix).jld2"

uki_params = UKI_params(ground_truth, Sigma_eta, m_0, C_0, gamma, uki_iters, model_func_1, model_func_2, hs_params, obs_func, transform_func_to_HS, batch_size=batch_size, param_save_filepath=param_save_filepath)

println("UKI Params created")

uki_iteration_part1!(uki_params)

# Submit the initial run script
cmd = `qsub $(script_dir)/$(case_prefix)_initial_script_binded.sh`
run(cmd)