# Helper functions in julia for Held Suarez Model
# export JULIA_NUM_THREADS=128

using Parameters
using FileIO
using NetCDF
using Statistics
using Distributions
using Random

@with_kw mutable struct HS_params
    # Key file paths
    CAM_dir
    base_case
    ground_truth_output_dir
    ground_truth_name
    proj_case_dir
    scratch_case_dir
    postrun_script

    # Configuration for job length related stuff
    ground_truth_length_days
    ground_truth_spinup_days
    time_step_secs
    short_run_nsteps

    # Array: Parameter names to match namelist
    param_names
    # Array: starting values for calibration 
    theta_zero
    # Array: starting spread for parameters
    C_zero
    # Batch size
    batch_size

    function HS_params(CAM_dir,
        base_case,
        ground_truth_output_dir,
        ground_truth_name,
        proj_case_dir,
        scratch_case_dir,
        postrun_script,
        ground_truth_length_days,
        ground_truth_spinup_days,
        time_step_secs,
        short_run_nsteps,
        param_names,
        theta_zero,
        C_zero,
        batch_size)

        return new(CAM_dir,
        base_case,
        ground_truth_output_dir,
        ground_truth_name,
        proj_case_dir,
        scratch_case_dir,
        postrun_script,
        ground_truth_length_days,
        ground_truth_spinup_days,
        time_step_secs,
        short_run_nsteps,
        param_names,
        theta_zero,
        C_zero,
        batch_size)
    end

end

# Write HS Params to file (must end in '.jld2')
function write_HS_params(params, filename)
    save(filename, Dict("params" => params))
end

# Read HS Params from file
function read_HS_params(filename)
    return load(filename, "params")
end

# Create case clones
function gen_HS_cases(hs_params::HS_params)
    CAM_dir = hs_params.CAM_dir
    base_case = hs_params.base_case
    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    n_sigma = length(hs_params.param_names) * 2 + 1
    batch_n = hs_params.batch_size

    case_base_name = "hs_uki_case"

    # create case ensembles of form base_name_sigma#_batch#
    # julia be 1 based indexing
    for sigma in 1:n_sigma
        Threads.@threads for sample in 1:batch_n
            cmd = `$(CAM_dir)/cime/scripts/create_clone --case $(proj_case_dir)/$(case_base_name)_$(sigma)_$(sample) --clone $(base_case) --keepexe --cime-output-root $scratch_case_dir`

            # println(cmd)
            run(cmd)
        end
    end
end

function gen_HS_GT_cases(hs_params::HS_params, uki_params::UKI_params, test_interval)
    CAM_dir = hs_params.CAM_dir
    base_case = hs_params.base_case
    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    case_base_name = "GT_test_case"

    uki_iters = size(uki_params.m_s, 1)

    # create case ensembles of form case_base_name_i
    # julia be 1 based indexing
    Threads.@threads for i in 1:test_interval:uki_iters
        cmd = `$(CAM_dir)/cime/scripts/create_clone --case $(proj_case_dir)/$(case_base_name)_$(i) --clone $(base_case) --keepexe --cime-output-root $scratch_case_dir`
        # println(cmd)
        run(cmd)
    end
end

#TODO
# Read a HS output .nc file and return an global average for each timestep saved. [3 x n_samples]
# If specified, add noise as: abs(value) * noise_magnitude * N(0, 1)
function read_HS_output(filepath, noise_magnitude)
    # Weights for spatial averaging
    gauss_weights = ncread(filename, "gw")

end

# Read HS output .nc file and average temp along longitude and vertical levels. 64 latitude values so output is [64, n_times]
# If specified, add noise as: abs(value) * noise_magnitude * N(0, 1)
function read_HS_T_lat_output(filepath; noise_magnitude = 0.0, toss=0, ergodic=false)
    println(filepath)
    
    temp = ncread(filepath, "T")
    # temp is read in with dim order [lon, lat, lev, time]
    println(size(temp))
    
    # Toss first timesteps if needed
    temp = temp[:, :, :, 1+toss:end]
    # println(size(temp))

    # If noise_magnitude is specified
    if noise_magnitude > 0.0
        println("Adding noise")
        # Get average magnitude of temp
        abs_temp = mean(abs.(temp))
        # println(size(abs_temp))
        # println(abs_temp)

        # Generate a noise matrix
        temp_dims = size(temp)
        noise_matrix = rand(Normal(), temp_dims) * abs_temp * noise_magnitude
        # println(size(noise_matrix))

        # add noise matrix to temp
        temp = temp + noise_matrix

    end

    if ergodic
        # println("Ergodic")
        # Average over longitude and vertical level and time
        output_matrix =  dropdims(mean(mean(mean(temp, dims=[4]), dims=[3]), dims=[1]), dims=(1, 3, 4))
        
        # Duplicate output_matrix to create a second time column
        # This is done so the uki can ignore the first entry like it does for the batch version

        output_matrix = hcat(output_matrix, output_matrix)
        # println(size(output_matrix))
    else
        # println("Not ergodic")
        # Average over longitude and vertical level
        output_matrix =  dropdims(mean(mean(temp, dims=[3]), dims=[1]), dims=(1, 3))
        # println(size(output_matrix))
    end

    # display(output_matrix)

    return output_matrix

end

# Read HS output .nc file and average temp along longitude. Matches up to previous work. 64 latitude values, 30 vertical levels so output is [64 * 30, n_times]
# If specified, add noise as: abs(value) * noise_magnitude * N(0, 1)
function read_HS_T_lat_sigma_output(filepath; noise_magnitude = 0.0, toss=0, ergodic=false)
    println(filepath)
    
    temp = ncread(filepath, "T")
    # temp is read in with dim order [lon, lat, lev, time]
    lon, lat, lev, time = size(temp)
    # println(size(temp))
    
    # Toss first timesteps if needed
    temp = temp[:, :, :, 1+toss:end]
    # println(size(temp))

    # If noise_magnitude is specified
    if noise_magnitude > 0.0
        # println("Adding noise")
        # Get average magnitude of temp
        abs_temp = mean(abs.(temp))
        # println(size(abs_temp))
        # println(abs_temp)

        # Generate a noise matrix
        temp_dims = size(temp)
        noise_matrix = rand(Normal(), temp_dims) * abs_temp * noise_magnitude
        # println(size(noise_matrix))

        # add noise matrix to temp
        temp = temp + noise_matrix

    end

    if ergodic
        # println("Ergodic")
        # Average over longitude and time
        # flatten resulting matrix
        output_matrix =  reshape(dropdims(mean(mean(temp, dims=[4]), dims=[1]), dims=(1, 4)), :)
        
        # Duplicate output_matrix to create a second time column
        # This is done so the uki can ignore the first entry like it does for the batch version

        output_matrix = hcat(output_matrix, output_matrix)
        # println(size(output_matrix))
    else
        # println("Not ergodic")
        # Average over longitude
        # flatten resulting matrix but maintain time dimension
        output_matrix =  reshape(dropdims(mean(temp, dims=[1]), dims=(1)), (lat * lev, :))
        # println(size(output_matrix))
    end

    # println(size(output_matrix))
    # display(output_matrix)

    return output_matrix

end

# Helper function to convert days to timeslices
function days_to_timesteps(days, hs_params::HS_params)
    day_seconds = 24 * 60 * 60
    time_step_secs = hs_params.time_step_secs

    timesteps = trunc(Int, (days * day_seconds) / time_step_secs)

    return timesteps
end

# Helper function returns number of timeslices needed to toss for spinup
function toss_timeslices(hs_params::HS_params)
    ground_truth_spinup_days = hs_params.ground_truth_spinup_days

    toss_timeslices = trunc(Int, days_to_timesteps(ground_truth_spinup_days, hs_params) / hs_params.short_run_nsteps)

    return toss_timeslices
end

# Transform function going from UKI space to HS model
function uki_to_hs_transform(uki_params)
    hs_params = zeros(4)

    # efolda
    hs_params[1] = 1 + abs(uki_params[1])

    #efolds
    hs_params[2] = (1 + abs(uki_params[2]) + abs(uki_params[1]) * abs(uki_params[2]))/(1 + abs(uki_params[1]))

    # delta_T_y
    hs_params[3] = abs(uki_params[3])

    # delta_theta_z
    hs_params[4] = abs(uki_params[4])

    return hs_params
end

# Edit HS case parameters, starting points, and if specified, submit jobs.
function setup_HS_cases(thetas, sample_idxs, moments_func, transform_func, hs_params::HS_params; submit=true)
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    N_samples = size(sample_idxs, 1)
    # println(N_sigma)
    # println(N_samples)

    param_names = hs_params.param_names
    # println(param_names)

    for sigma in 1:N_sigma
        transformed_theta = transform_func(thetas[:, sigma])
        println(transformed_theta)
        for j in 1:N_samples
            case_name = "$(case_base_name)_$(sigma)_$(j)"
            date_string, date_only, seconds_only = hs_date_string(sample_idxs[j], hs_params)

            # Copy fresh version of user_nl_cam from base_case
            cmd = `cp $(base_case)/user_nl_cam $(proj_case_dir)/$(case_name)/user_nl_cam`
            run(cmd)

            # copy modified parameters into user_nl_cam
            open("$(proj_case_dir)/$(case_name)/user_nl_cam", "a") do io
                for k in 1:N_theta
                    formatted_param = fortran_nl_param(transformed_theta[k])
                    println(io, "$(param_names[k]) = $(formatted_param)")
                    # println(cmd)
                end
            end

            # set needed xml values for restart files
            cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_REFCASE=$(ground_truth_name) --caseroot $(proj_case_dir)/$(case_name)`
            run(cmd)
            cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_REFDATE=$(date_only) --caseroot $(proj_case_dir)/$(case_name)`
            run(cmd)
            cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_REFTOD=$(seconds_only) --caseroot $(proj_case_dir)/$(case_name)`
            run(cmd)

            # Prestage restart files with symlinks
            cmd = `ln -sf $(ground_truth_output_dir)/run/$(ground_truth_name).cam.r.$(date_string).nc $(scratch_case_dir)/$case_name/run/$(ground_truth_name).cam.r.$(date_string).nc`
            run(cmd)
            cmd = `ln -sf $(ground_truth_output_dir)/run/$(ground_truth_name).cpl.r.$(date_string).nc $(scratch_case_dir)/$case_name/run/$(ground_truth_name).cpl.r.$(date_string).nc`
            run(cmd)

            # create pointer files
            open("$(scratch_case_dir)/$case_name/run/rpointer.atm", "w") do io
                println(io, "$(ground_truth_name).cam.r.$(date_string).nc")
            end
            open("$(scratch_case_dir)/$case_name/run/rpointer.cpl", "w") do io
                println(io, "$(ground_truth_name).cpl.r.$(date_string).nc")
            end

        end
    end

    # all cases edited
    # Edit postrun script for last case
    # Notification needs to be done in postrun script
    case_name = "$(case_base_name)_$(N_sigma)_$(N_samples)"
    cmd = `$(proj_case_dir)/$(case_name)/xmlchange POSTRUN_SCRIPT=$(postrun_script) --caseroot $(proj_case_dir)/$(case_name)`
    run(cmd)

    # submit jobs
    if submit
        for sigma in 1:N_sigma
            transformed_theta = transform_func(thetas[:, sigma])
            println(transformed_theta)
            for j in 1:N_samples
                case_name = "$(case_base_name)_$(sigma)_$(j)"

                # cmd = `$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`
                cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                run(cmd)
            end
        end
    end

end

# Edit HS case parameters, starting points, and if specified, submit jobs.
# Parallel version
function par_setup_HS_cases(thetas, sample_idxs, moments_func, transform_func, hs_params::HS_params; submit=true)
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    batch_n = hs_params.batch_size
    println(N_theta)
    println(N_sigma)
    # println(N_samples)

    param_names = hs_params.param_names
    println(param_names)

    # display(ENV)

    for sigma in 1:N_sigma
        transformed_theta = transform_func(thetas[:, sigma])
        println(transformed_theta)
        Threads.@threads :static for j in 1:batch_n
            case_name = "$(case_base_name)_$(sigma)_$(j)"
            date_string, date_only, seconds_only = hs_date_string(sample_idxs[j], hs_params)
            
            println(case_name)
            # display(ENV)

            # Copy fresh version of user_nl_cam from base_case
            # cmd = `cp $(base_case)/user_nl_cam $(proj_case_dir)/$(case_name)/user_nl_cam`
            # run(cmd)

            # use julia's built in copy function for user_nl_cam
            cp("$(base_case)/user_nl_cam", "$(proj_case_dir)/$(case_name)/user_nl_cam", force=true)


            # copy modified parameters into user_nl_cam
            open("$(proj_case_dir)/$(case_name)/user_nl_cam", "a", lock=true) do io
                for k in 1:N_theta
                    formatted_param = fortran_nl_param(transformed_theta[k])
                    println(io, "$(param_names[k]) = $(formatted_param)")
                    # println(cmd)
                end
            end

            # set needed xml values for restart files
            cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_REFCASE=$(ground_truth_name),RUN_REFDATE=$(date_only),RUN_REFTOD=$(seconds_only) --caseroot $(proj_case_dir)/$(case_name)`
            run(cmd)
            sleep(10)
            # cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_REFDATE=$(date_only) --caseroot $(proj_case_dir)/$(case_name)`
            # run(cmd)
            # sleep(2)
            # cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_REFTOD=$(seconds_only) --caseroot $(proj_case_dir)/$(case_name)`
            # run(cmd)
            # sleep(2)

            # Prestage restart files with symlinks
            cmd = `ln -sf $(ground_truth_output_dir)/run/$(ground_truth_name).cam.r.$(date_string).nc $(scratch_case_dir)/$case_name/run/$(ground_truth_name).cam.r.$(date_string).nc`
            run(cmd)
            cmd = `ln -sf $(ground_truth_output_dir)/run/$(ground_truth_name).cpl.r.$(date_string).nc $(scratch_case_dir)/$case_name/run/$(ground_truth_name).cpl.r.$(date_string).nc`
            run(cmd)

            # create pointer files
            open("$(scratch_case_dir)/$case_name/run/rpointer.atm", "w") do io
                println(io, "$(ground_truth_name).cam.r.$(date_string).nc")
            end
            open("$(scratch_case_dir)/$case_name/run/rpointer.cpl", "w") do io
                println(io, "$(ground_truth_name).cpl.r.$(date_string).nc")
            end

        end
    end

    # all cases edited
    # Edit postrun script for last case
    # Notification needs to be done in postrun script
    case_name = "$(case_base_name)_$(N_sigma)_$(batch_n)"
    cmd = `$(proj_case_dir)/$(case_name)/xmlchange POSTRUN_SCRIPT=$(postrun_script) --caseroot $(proj_case_dir)/$(case_name)`
    run(cmd)
    sleep(1)

    # submit jobs
    if submit
        for sigma in 1:N_sigma
            # for j in 1:N_samples
            Threads.@threads :static for j in 1:batch_n
                if sigma == N_sigma && j == batch_n
                    continue
                end

                case_name = "$(case_base_name)_$(sigma)_$(j)"

                # cmd = `$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`
                # cmd = addenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, "PWD" => "$(proj_case_dir)/$(case_name)")
                cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                run(cmd)
                sleep(5)
            end
        end

        # submit final case last
        case_name = "$(case_base_name)_$(N_sigma)_$(batch_n)"

        cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
        run(cmd)
        sleep(5)

    end

end

# This is model function part 1
# Needs to have this form.  model_func_part_1(thetas, sample_idxs, moments_func, transform_func, model_params)
# No return values

function test_par_setup_HS_cases(thetas, sample_idxs, moments_func, transform_func, hs_params::HS_params; submit=false)
    # println("Threads: $(Threads.nthreads())")
    
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    batch_n = hs_params.batch_size
    println(N_theta)
    println(N_sigma)
    # println(N_samples)

    param_names = hs_params.param_names
    println(param_names)

    # display(ENV)

    for sigma in 1:N_sigma
        transformed_theta = transform_func(thetas[:, sigma])
        println(transformed_theta)

        Threads.@threads :static for j in 1:batch_n
            case_name = "$(case_base_name)_$(sigma)_$(j)"
            date_string, date_only, seconds_only = hs_date_string(sample_idxs[j], hs_params)
            
            println(case_name)

            # use julia's built in copy function for user_nl_cam
            cp("$(base_case)/user_nl_cam", "$(proj_case_dir)/$(case_name)/user_nl_cam", force=true)

            # copy modified parameters into user_nl_cam
            open("$(proj_case_dir)/$(case_name)/user_nl_cam", "a", lock=true) do io
                for k in 1:N_theta
                    formatted_param = fortran_nl_param(transformed_theta[k])
                    println(io, "$(param_names[k]) = $(formatted_param)")
                    # println(cmd)
                end
            end

            # set needed xml values for restart files
            if sigma == N_sigma && j == batch_n
                # cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_TYPE=branch,RUN_REFCASE=$(ground_truth_name),RUN_REFDATE=$(date_only),RUN_REFTOD=$(seconds_only),POSTRUN_SCRIPT=$(postrun_script) --caseroot $(proj_case_dir)/$(case_name)`
                cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_TYPE=branch,RUN_REFCASE=$(ground_truth_name),RUN_REFDATE=$(date_only),RUN_REFTOD=$(seconds_only) --caseroot $(proj_case_dir)/$(case_name)`
                run(cmd)
                sleep(10)
            else
                cmd = `$(proj_case_dir)/$(case_name)/xmlchange RUN_TYPE=branch,RUN_REFCASE=$(ground_truth_name),RUN_REFDATE=$(date_only),RUN_REFTOD=$(seconds_only) --caseroot $(proj_case_dir)/$(case_name)`
                run(cmd)
                sleep(10)
            end

            # Prestage restart files with symlinks
            cmd = `ln -sf $(ground_truth_output_dir)/run/$(ground_truth_name).cam.r.$(date_string).nc $(scratch_case_dir)/$case_name/run/$(ground_truth_name).cam.r.$(date_string).nc`
            run(cmd)
            cmd = `ln -sf $(ground_truth_output_dir)/run/$(ground_truth_name).cpl.r.$(date_string).nc $(scratch_case_dir)/$case_name/run/$(ground_truth_name).cpl.r.$(date_string).nc`
            run(cmd)

            # create pointer files
            open("$(scratch_case_dir)/$case_name/run/rpointer.atm", "w") do io
                println(io, "$(ground_truth_name).cam.r.$(date_string).nc")
            end
            open("$(scratch_case_dir)/$case_name/run/rpointer.cpl", "w") do io
                println(io, "$(ground_truth_name).cpl.r.$(date_string).nc")
            end

        end
    end

    # submit jobs
    if submit
        for sigma in 1:N_sigma
            # for j in 1:N_samples
            Threads.@threads :static for j in 1:batch_n
                # submit final case last
                if sigma == N_sigma && j == batch_n
                    sleep(5)
                    case_name = "$(case_base_name)_$(sigma)_$(j)"
                    cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                    run(cmd)
                    sleep(5)
                else
                    case_name = "$(case_base_name)_$(sigma)_$(j)"

                    # cmd = `$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`
                    # cmd = addenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, "PWD" => "$(proj_case_dir)/$(case_name)")
                    cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                    run(cmd)
                    sleep(5)
                end
            end
        end

    end


end

# This is model function part 1 for ergodic runs
# Needs to have this form.  model_func_part_1(thetas, sample_idxs, moments_func, transform_func, model_params)
# No return values

function ergodic_par_setup_HS_cases(thetas, sample_idxs, moments_func, transform_func, hs_params::HS_params; submit=true)
    # println("Threads: $(Threads.nthreads())")
    
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    batch_n = hs_params.batch_size
    println(N_theta)
    println(N_sigma)
    # println(N_samples)

    param_names = hs_params.param_names
    println(param_names)

    # display(ENV)

    for sigma in 1:N_sigma
        transformed_theta = transform_func(thetas[:, sigma])
        println(transformed_theta)

        for j in 1:batch_n
            case_name = "$(case_base_name)_$(sigma)_$(j)"
            
            println(case_name)

            # use julia's built in copy function for user_nl_cam
            cp("$(base_case)/user_nl_cam", "$(proj_case_dir)/$(case_name)/user_nl_cam", force=true)

            # copy modified parameters into user_nl_cam
            open("$(proj_case_dir)/$(case_name)/user_nl_cam", "a", lock=true) do io
                for k in 1:N_theta
                    formatted_param = fortran_nl_param(transformed_theta[k])
                    println(io, "$(param_names[k]) = $(formatted_param)")
                    # println(cmd)
                end
            end

            # set needed xml values not using restart files for ergodic files
            if sigma == N_sigma && j == batch_n
                cmd = `$(proj_case_dir)/$(case_name)/xmlchange POSTRUN_SCRIPT=$(postrun_script),REST_OPTION=none --caseroot $(proj_case_dir)/$(case_name)`
                run(cmd)
                sleep(10)
            else
                cmd = `$(proj_case_dir)/$(case_name)/xmlchange REST_OPTION=none --caseroot $(proj_case_dir)/$(case_name)`
                run(cmd)
                sleep(10)
            end

        end
    end

    # submit jobs
    if submit
        for sigma in 1:N_sigma
            # for j in 1:N_samples
            for j in 1:batch_n
                # submit final case last
                if sigma == N_sigma && j == batch_n
                    sleep(5)
                    case_name = "$(case_base_name)_$(sigma)_$(j)"
                    cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                    run(cmd)
                    sleep(5)
                else
                    case_name = "$(case_base_name)_$(sigma)_$(j)"

                    cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                    run(cmd)
                    sleep(5)
                end
            end
        end

    end


end


function par_setup_GT_cases(uki_params::UKI_params, hs_params::HS_params, test_interval)
    # this is the base of the of the string for cases, NOT the case from which they are cloned
    case_base_name = "GT_test_case"
    
    # this is the Case things are cloned from
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    m_s = uki_params.m_s
    uki_iters = size(m_s, 1)
    N_theta = size(m_s, 2)

    transform_func = uki_params.transform_func

    param_names = hs_params.param_names

    Threads.@threads :static for i in 1:test_interval:uki_iters
        case_name = "$(case_base_name)_$(i)"

        # use julia's built in copy function for user_nl_cam
        cp("$(base_case)/user_nl_cam", "$(proj_case_dir)/$(case_name)/user_nl_cam", force=true)

        # Select theta from uki_params
        transformed_theta = transform_func(m_s[i, :])

        # copy modified parameters into user_nl_cam
        open("$(proj_case_dir)/$(case_name)/user_nl_cam", "a", lock=true) do io
            for k in 1:N_theta
                formatted_param = fortran_nl_param(transformed_theta[k])
                println(io, "$(param_names[k]) = $(formatted_param)")
                # println(cmd)
            end
        end

        # submit jobs
        cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
        run(cmd)
        sleep(5)

    end
end

# Read in outputs from cases and return
# This is model_func_part_2
# Needs to have this form.  model_func_part_2(sample_idxs, moments_func, N_theta, N_out, model_params)
# return x_s, fail_bool 

function read_HS_cases(start_idxs, moments_func, N_theta, N_out, hs_params::HS_params)
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    time_step_secs = hs_params.time_step_secs
    short_run_nsteps = hs_params.short_run_nsteps

    # save_secs = time_step_secs * short_run_nsteps
    # save_secs_2 = time_step_secs * short_run_nsteps * 2

    N_samples = size(start_idxs, 1)
    N_sigma = N_theta * 2 + 1

    # Empty 2D array [sigma, j] for failed runs

    failed_runs = Vector{Tuple{Int, Int}}()

    # x_s shape [sigma_count, N_out, batch_size]
    x_s = zeros(N_sigma, N_out, N_samples)

    fail_bool = false
    
    # Check runs for success
    for sigma in 1:N_sigma
        for j in 1:N_samples
            # skip final case
            if sigma == N_sigma && j == N_samples
                continue
            end

            case_name = "$(case_base_name)_$(sigma)_$(j)"

            # check to see if last line of CaseStatus file contains 'case.run success'
            case_status_file = "$(proj_case_dir)/$(case_name)/CaseStatus"
            case_status = readlines(case_status_file)

            # We want to check if a success was reported in the CaseStatus file in the last 6 lines
            success_lines = [occursin("case.run success", line) for line in case_status[end-5:end]]
            if any(success_lines)
                # println("Case $(case_name) passed")
                continue
            else
                println("Case $(case_name) failed")
                push!(failed_runs, (sigma, j))
            end
        end
    end

    # Add final case to failed runs
    push!(failed_runs, (N_sigma, N_samples))

    # Only resubmit if there are failed runs (not just the final case)
    if length(failed_runs) > 1

        fail_bool = true

        println("Failed runs: Resubmitting")
        Threads.@threads :static for (sigma, j) in failed_runs
            # submit final case last
            if sigma == N_sigma && j == N_samples
                sleep(5)
                case_name = "$(case_base_name)_$(sigma)_$(j)"
                cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                run(cmd)
                sleep(5)
            else
                case_name = "$(case_base_name)_$(sigma)_$(j)"
                cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                run(cmd)
                sleep(5)
            end
        end

        # Submit final job again
        # case_name = "$(case_base_name)_$(N_sigma)_$(N_samples)"
        # cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
        # run(cmd)
        # sleep(5)

        # exit with error
        println("Failed runs: Resubmitted")
        exit()


    else
        println("All cases ran successfully")

        # Read in files if all have run
        for sigma in 1:N_sigma
            for j in 1:N_samples
                date_string, _, _ = hs_date_string(start_idxs[j]+1, hs_params)
                case_name = "$(case_base_name)_$(sigma)_$(j)"
                case_file_path = "$(scratch_case_dir)/$(case_name)/run/$(case_name).cam.h1i.$(date_string).nc"
                # println(case_file_path)
                x_s[sigma, :, j] = read_HS_T_lat_sigma_output(case_file_path)
            end
        end
    end

    return x_s, fail_bool

end

# Read in outputs from cases and return
# This is model_func_part_2 for ergodic runs
# Needs to have this form.  model_func_part_2(sample_idxs, moments_func, N_theta, N_out, model_params)
# return x_s, fail_bool 

function ergodic_read_HS_cases(start_idxs, moments_func, N_theta, N_out, hs_params::HS_params)
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    time_step_secs = hs_params.time_step_secs
    short_run_nsteps = hs_params.short_run_nsteps

    # save_secs = time_step_secs * short_run_nsteps
    # save_secs_2 = time_step_secs * short_run_nsteps * 2

    N_samples = size(start_idxs, 1)
    N_sigma = N_theta * 2 + 1

    # Empty 2D array [sigma, j] for failed runs

    failed_runs = Vector{Tuple{Int, Int}}()

    # x_s shape [sigma_count, N_out, batch_size]
    x_s = zeros(N_sigma, N_out, N_samples)

    fail_bool = false
    
    # Check runs for success
    for sigma in 1:N_sigma
        for j in 1:N_samples
            # skip final case
            if sigma == N_sigma && j == N_samples
                continue
            end

            case_name = "$(case_base_name)_$(sigma)_$(j)"

            # check to see if last line of CaseStatus file contains 'case.run success'
            case_status_file = "$(proj_case_dir)/$(case_name)/CaseStatus"
            case_status = readlines(case_status_file)
            if occursin("case.run success", case_status[end-1])
                # println("Case $(case_name) passed")
            else
                println("Case $(case_name) failed")
                push!(failed_runs, (sigma, j))
            end
        end
    end

    # Add final case to failed runs
    push!(failed_runs, (N_sigma, N_samples))

    # Only resubmit if there are failed runs (not just the final case)
    if length(failed_runs) > 1

        fail_bool = true

        println("Failed runs: Resubmitting")
        Threads.@threads :static for (sigma, j) in failed_runs
            # submit final case last
            if sigma == N_sigma && j == N_samples
                sleep(5)
                case_name = "$(case_base_name)_$(sigma)_$(j)"
                cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                run(cmd)
                sleep(5)
            else
                case_name = "$(case_base_name)_$(sigma)_$(j)"
                cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
                run(cmd)
                sleep(5)
            end
        end

        # Submit final job again
        # case_name = "$(case_base_name)_$(N_sigma)_$(N_samples)"
        # cmd = setenv(`$(proj_case_dir)/$(case_name)/case.submit $(proj_case_dir)/$(case_name)`, dir="$(proj_case_dir)/$(case_name)")
        # run(cmd)
        # sleep(5)

        # exit with error
        println("Failed runs: Resubmitted")
        exit()


    else
        println("All cases ran successfully")
        toss_timeslices_count = toss_timeslices(hs_params)

        # Read in files if all have run
        for sigma in 1:N_sigma
            for j in 1:N_samples
                case_name = "$(case_base_name)_$(sigma)_$(j)"
                case_file_path = "$(scratch_case_dir)/$(case_name)/run/$(case_name).cam.h1i.0001-01-01-10800.nc"
                # println(case_file_path)
                x_s[sigma, :, j] = read_HS_T_lat_sigma_output(case_file_path, toss=toss_timeslices_count, ergodic=true)[:, 1]
            end
        end
    end

    return x_s, fail_bool

end

# Read in outputs from cases and return
# Parallel version
function par_read_HS_cases(start_idxs, moments_func, N_theta, N_out, hs_params::HS_params)
    # this is the base of the of the string for uki cases, NOT the case from which they are cloned
    case_base_name = "hs_uki_case"

    # this is the Case things are cloned from 
    base_case = hs_params.base_case

    proj_case_dir = hs_params.proj_case_dir
    scratch_case_dir = hs_params.scratch_case_dir

    ground_truth_name = hs_params.ground_truth_name
    ground_truth_output_dir = hs_params.ground_truth_output_dir

    postrun_script = hs_params.postrun_script

    time_step_secs = hs_params.time_step_secs
    short_run_nsteps = hs_params.short_run_nsteps

    save_secs = time_step_secs * short_run_nsteps

    N_samples = size(start_idxs, 1)
    N_sigma = N_theta * 2 + 1

    # x_s shape [sigma_count, N_out, batch_size]
    x_s = zeros(N_sigma, N_out, N_samples)

    for sigma in 1:N_sigma
        Threads.@threads :static for j in 1:N_samples
            case_name = "$(case_base_name)_$(sigma)_$(j)"
            case_file_path = "$(scratch_case_dir)/$(case_name)/run/$(case_name).cam.h1i.0001-01-01-$(save_secs).nc"
            println(case_file_path)
            x_s[sigma, :, j] = read_HS_T_lat_output(case_file_path)
        end
    end

    return x_s

end


# return date string for held-suarez output name from case index
function hs_date_string(case_idx, hs_params::HS_params)
    month_days = [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]

    day_seconds = 24 * 60 * 60
    year_seconds = day_seconds * 365

    ground_truth_length_days = hs_params.ground_truth_length_days
    ground_truth_spinup_days = hs_params.ground_truth_spinup_days
    time_step_secs = hs_params.time_step_secs
    short_run_nsteps = hs_params.short_run_nsteps

    total_seconds = ground_truth_spinup_days * day_seconds + short_run_nsteps * time_step_secs * (case_idx - 1)
    # println(total_seconds)

    #Year indexing actually starts from 1
    year = div(total_seconds, year_seconds) + 1
    # println(year)

    # Remove year seconds from total seconds
    total_seconds = mod(total_seconds, year_seconds)
    # println(total_seconds)

    # convert remaining seconds into days
    # day indexing also starts from 1
    days = div(total_seconds, day_seconds)
    # println(days)

    # remove day seconds from total_seconds
    total_seconds = mod(total_seconds, day_seconds)
    # println(total_seconds)

    month = 1
    for day_count_per_month in month_days
        if div(days, day_count_per_month) > 0
            days = days - day_count_per_month
            month += 1
        else
            break
        end
    end

     # day indexing also starts from 1
    days += 1
    # println(days)
    # println(month)

    date_string = "$(lpad(year, 4, "0"))-$(lpad(month, 2, "0"))-$(lpad(days, 2, "0"))-$(lpad(total_seconds, 5, "0"))"

    date_only = "$(lpad(year, 4, "0"))-$(lpad(month, 2, "0"))-$(lpad(days, 2, "0"))"

    seconds_only = "$(lpad(total_seconds, 5, "0"))"

    return date_string, date_only, seconds_only

end

# return string version of float for fortran namelist
# Like: 40.04d0
function fortran_nl_param(param_value)
    return "$(param_value)d0"
end

# Read in outputs from GT test cases
function read_GT_cases(test_interval, hs_params::HS_params, uki_params::UKI_params)
    # this is the base of the of the string for cases, NOT the case from which they are cloned
    case_base_name = "GT_test_case"

    scratch_case_dir = hs_params.scratch_case_dir

    m_s = uki_params.m_s
    uki_iters = size(m_s, 1)

    time_step_secs = hs_params.time_step_secs
    short_run_nsteps = hs_params.short_run_nsteps

    save_secs = time_step_secs * short_run_nsteps

    test_idxs = collect(1:test_interval:uki_iters)
    total_tests = length(test_idxs)

    N_samples = uki_params.N_samples
    N_out = uki_params.N_out

    ground_truth_tests = zeros(total_tests, N_out, N_samples)

    toss_timeslices_count = toss_timeslices(hs_params)

    for i in 1:total_tests
        j = test_idxs[i]
        case_name = "$(case_base_name)_$(j)"
        println(case_name)

        case_file_path = "$(scratch_case_dir)/$(case_name)/run/$(case_name).cam.h1i.0001-01-01-$(save_secs).nc"

        ground_truth_tests[i, :, :] = read_HS_T_lat_output(case_file_path, toss = toss_timeslices_count)
    end

    # average over time
    ground_truth_tests_avg = mean(ground_truth_tests, dims=3)

    return ground_truth_tests_avg
end

# function to generate calibration scenario scripts
function generate_calibration_scenario_scripts(sample_directory, scenario_name, case_script_directory, batch_size, noise_level, sigma_exponent, covariance_exponent; proj_case_dir=nothing, ergodic=false)

    if ergodic
         # Make new run_script
        # read in sample_run_script.sh
        sample_run_script_path = sample_directory * "/sample_run_script.sh"
        sample_run_script = readlines(sample_run_script_path)

        # make new run_script
        # Replace all instances of scenario_name with scenario_name variable
        new_run_script = [replace(line, "scenario_name" => scenario_name) for line in sample_run_script]

        # Replace all instances of batch_size with batch_size variable
        new_run_script = [replace(line, "batch_size" => batch_size) for line in new_run_script]

        # Replace all instances of proj_case_dir with proj_case_dir variable
        if proj_case_dir != nothing
            new_run_script = [replace(line, "proj_case_dir" => proj_case_dir) for line in new_run_script]
        end


        # Write new run_script to case_script_directory with scenario_name
        # save file with 777 permissions
        new_run_script_path = case_script_directory * "/$(scenario_name)_run_script.sh"
        open(new_run_script_path, "w") do io
            for line in new_run_script
                println(io, line)
            end
        end
        chmod(new_run_script_path, 0o777)

        # Make new postrun_julia_script
        # read in sample_postrun_julia_script.jl
        sample_postrun_julia_script_path = sample_directory * "/sample_postrun_julia_script.jl"
        sample_postrun_julia_script = readlines(sample_postrun_julia_script_path)

        # Replace all instances of scenario_name with scenario_name variable
        new_postrun_julia_script = [replace(line, "scenario_name" => scenario_name) for line in sample_postrun_julia_script]

        # Write new postrun_julia_script to case_directory with scenario_name
        new_postrun_julia_script_path = case_script_directory * "/$(scenario_name)_postrun_julia_script.jl"
        open(new_postrun_julia_script_path, "w") do io
            for line in new_postrun_julia_script
                println(io, line)
            end
        end
        chmod(new_postrun_julia_script_path, 0o777)

        # Make new postrun_bash_script
        # read in sample_postrun_bash_script.sh
        sample_postrun_bash_script_path = sample_directory * "/sample_postrun_bash_script.sh"
        sample_postrun_bash_script = readlines(sample_postrun_bash_script_path)
        
        # Replace all instances of scenario_name with scenario_name variable
        new_postrun_bash_script = [replace(line, "scenario_name" => scenario_name) for line in sample_postrun_bash_script]
        
        # Write new postrun_bash_script to case_script_directory with scenario_name
        new_postrun_bash_script_path = case_script_directory * "/$(scenario_name)_postrun_bash_script.sh"
        open(new_postrun_bash_script_path, "w") do io
            for line in new_postrun_bash_script
                println(io, line)
            end
        end
        chmod(new_postrun_bash_script_path, 0o777)

    else
        # Make new run_script
        # read in sample_run_script.sh
        sample_run_script_path = sample_directory * "/sample_run_script_binded.sh"
        sample_run_script = readlines(sample_run_script_path)

        # make new run_script
        # Replace all instances of scenario_name with scenario_name variable
        new_run_script = [replace(line, "scenario_name" => scenario_name) for line in sample_run_script]

        # Replace all instances of batch_size with batch_size variable
        new_run_script = [replace(line, "batch_size" => batch_size) for line in new_run_script]

        # Replace all instances of proj_case_dir with proj_case_dir variable
        if proj_case_dir != nothing
            new_run_script = [replace(line, "proj_case_dir" => proj_case_dir) for line in new_run_script]
        end


        # Write new run_script to case_script_directory with scenario_name
        # save file with 777 permissions
        new_run_script_path = case_script_directory * "/$(scenario_name)_run_script_binded.sh"
        open(new_run_script_path, "w") do io
            for line in new_run_script
                println(io, line)
            end
        end
        chmod(new_run_script_path, 0o777)

        # Make new initial_script
        # read in sample_run_script.sh
        sample_initial_script_path = sample_directory * "/sample_initial_script_binded.sh"
        sample_initial_script = readlines(sample_initial_script_path)

        # make new run_script
        # Replace all instances of scenario_name with scenario_name variable
        new_initial_script = [replace(line, "scenario_name" => scenario_name) for line in sample_initial_script]

        # Replace all instances of batch_size with batch_size variable
        new_initial_script = [replace(line, "batch_size" => batch_size) for line in new_initial_script]

        # Replace all instances of proj_case_dir with proj_case_dir variable
        if proj_case_dir != nothing
            new_initial_script = [replace(line, "proj_case_dir" => proj_case_dir) for line in new_initial_script]
        end


        # Write new run_script to case_script_directory with scenario_name
        # save file with 777 permissions
        new_initial_script_path = case_script_directory * "/$(scenario_name)_initial_script_binded.sh"
        open(new_initial_script_path, "w") do io
            for line in new_initial_script
                println(io, line)
            end
        end
        chmod(new_initial_script_path, 0o777)

        # Make new postrun_julia_script
        # read in sample_postrun_julia_script.jl
        sample_postrun_julia_script_path = sample_directory * "/sample_postrun_julia_script.jl"
        sample_postrun_julia_script = readlines(sample_postrun_julia_script_path)

        # Replace all instances of scenario_name with scenario_name variable
        new_postrun_julia_script = [replace(line, "scenario_name" => scenario_name) for line in sample_postrun_julia_script]

        # Write new postrun_julia_script to case_directory with scenario_name
        new_postrun_julia_script_path = case_script_directory * "/$(scenario_name)_postrun_julia_script.jl"
        open(new_postrun_julia_script_path, "w") do io
            for line in new_postrun_julia_script
                println(io, line)
            end
        end
        chmod(new_postrun_julia_script_path, 0o777)
    end
    
    # # Make new postrun_bash_script
    # # read in sample_postrun_bash_script.sh
    # sample_postrun_bash_script_path = sample_directory * "/sample_postrun_bash_script.sh"
    # sample_postrun_bash_script = readlines(sample_postrun_bash_script_path)
    
    # # Replace all instances of scenario_name with scenario_name variable
    # new_postrun_bash_script = [replace(line, "scenario_name" => scenario_name) for line in sample_postrun_bash_script]
    
    # # Write new postrun_bash_script to case_script_directory with scenario_name
    # new_postrun_bash_script_path = case_script_directory * "/$(scenario_name)_postrun_bash_script.sh"
    # open(new_postrun_bash_script_path, "w") do io
    #     for line in new_postrun_bash_script
    #         println(io, line)
    #     end
    # end
    # chmod(new_postrun_bash_script_path, 0o777)

end
