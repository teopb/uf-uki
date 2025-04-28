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

# generate calibration scenario scripts
generate_calibration_scenario_scripts(script_sample_dir, case_prefix, script_dir, batch_size, noise_level, sigma_exponent, covariance_exponent, ergodic=true)

# held suarez parameters
CAM_dir = "your_path/CAM_hs_nl"
base_case = "your_path/CESM_cases/HS_GT_600day"
ground_truth_output_dir = "your_path/HS_GT_600day"
ground_truth_name = "HS_GT_600day"
proj_case_dir = "your_path/$(case_prefix)_hs_uki_cases"
scratch_case_dir = "your_path/$(case_prefix)_hs_uki_cases"
postrun_script = script_dir * "/$(case_prefix)_postrun_bash_script.sh"
ground_truth_length_days = 600
ground_truth_spinup_days = 200
time_step_secs = 1200
short_run_nsteps = 9
param_names = ["held_suarez_efolda", "held_suarez_efolds", "held_suarez_delta_T_y", "held_suarez_delta_theta_z"]
theta_zero = [40.78678556439521,
3.5266736800410805,
61.8798319514473,
9.862751352323626]
C_zero = [1, 1, 1, 1] * 10.0^(covariance_exponent)
# batch_size = 100

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
if isdir(proj_case_dir * "/hs_uki_case_2_1")
    println("Cases already exist")
else
    println("Creating new cases")
    gen_HS_cases(hs_params)
end

# UKI parameters
ground_truth_filename = "HS_GT_600day.cam.h1i.0001-01-01-10800.nc"
ground_truth_filepath = ground_truth_output_dir * "/run/" * ground_truth_filename
toss_timeslices_count = toss_timeslices(hs_params)
noise_magnitude=noise_level

println("Reading ground truth")
println("Toss timeslices: $(toss_timeslices_count), Noise magnitude: $(noise_magnitude), Ergodic: true, Ground truth filepath: $(ground_truth_filepath)")

ground_truth = read_HS_T_lat_sigma_output(ground_truth_filepath, toss = toss_timeslices_count, noise_magnitude = noise_magnitude, ergodic=true)

# println("Ground truth:")
# display(ground_truth)

obs_func = nothing

transform_func_to_HS = uki_to_hs_transform

uki_iters = iters

gamma = 1

prior_theta = [40.78678556439521, 3.5266736800410805, 61.8798319514473, 9.862751352323626]
trans_prior_theta = [-39.78678556439521, 3.5021559352332385, 61.8798319514473, 9.862751352323626]

# prior_theta = [40, 4, 60, 10]
# trans_prior_theta = [39, 159/40, 60, 10]

m_0 = trans_prior_theta
C_0 = diagm(ones(size(trans_prior_theta, 1))) * 10.0^(covariance_exponent)

# println("C_0:")
# display(C_0)

Sigma_eta = diagm(ones(size(ground_truth, 1)) * 3) * 10.0^(sigma_exponent)

# println("Sigma_eta:")
# display(Sigma_eta)

model_func_1 = ergodic_par_setup_HS_cases
model_func_2 = ergodic_read_HS_cases

param_save_filepath = proj_case_dir * "/uki_params_$(case_prefix).jld2"

uki_params = UKI_params(ground_truth, Sigma_eta, m_0, C_0, gamma, uki_iters, model_func_1, model_func_2, hs_params, obs_func, transform_func_to_HS, batch_size=batch_size, param_save_filepath=param_save_filepath)

println("UKI Params created")

uki_iteration_part1!(uki_params)