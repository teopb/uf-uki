using Printf
include("Lorenz.jl")
include("UKI.jl")
using JLD2
using Plots
using LaTeXStrings
# Set the default line width for all plots
default(linewidth=2)

# Function to save array data in a format easily read by both Python and Julia
# For reading in Python, use h5py
# import h5py
# with h5py.File('filename.jld', 'r') as f:
# data = f['data'][()]
# 
# For reading in Julia, use JLD
# using JLD
# data = load("filename.jld")["data"]

function save_data(data, filename)
    try
        # Add .jld2 extension if no extension is provided
        if !contains(filename, ".")
            filename = filename * ".jld2"
        elseif !endswith(filename, ".jld2")
            @warn "Filename does not end in .jld2 - adding .jld2 extension"
            filename = filename * ".jld2"
        end

        # Save data to a file in JLD format
        @info "Saving data to $(filename)"
        save(filename, "data", data)
        
        return true
    catch e
        @error "Failed to save data" exception=(e, catch_backtrace())
        return false
    end
end

# Runs to test chaotic loss
function ergodic_run_moments(t_end, timestep, toss, obs_noise, print_output=false)
    # L96 parameters
    if print_output
        println("t_end = $(t_end), timestep = $(timestep), toss = $(toss), obs_noise = $(obs_noise)")
    end

    theta_names  = ["h" "F" "c" "b"]

    theta_true = [1.0, 10.0, log(10), 10.0]

    # L96 run without noise for reference
    noise_mag = 0.0

    # Initial conditions are stored here
    l96_params = L96_two_params(noise=noise_mag, t_end=t_end, toss=toss, timestep=timestep)

    # Don't transform observations yet because we will need them as IC's 
    moments_func = L96_no_moments

    # These transforms only the "c" parameter via an absolute value. 
    # So the L96 model always recieves abs(c)
    transform_func_to_UKI = L96_trans_C2_to_UKI
    transform_func_to_L96 = L96_trans_C2_to_L96

    # This doesn't do anything in this case
    # Could be important with a different transform or theta_true
    trans_theta_true = transform_func_to_UKI(theta_true)

    # This prepends K and J (defining the dimensionality of the L96 system)
    # Default values are K=36, J=10 
    KJ_theta_true = vcat([l96_params.K, l96_params.J], trans_theta_true)

    model_output = L96_model_full_output(KJ_theta_true, moments_func, transform_func_to_L96, l96_params);

    noise_mag = obs_noise
    # Tranform observations 
    # (1/K)Sum_over_k(X_k, y_mean_k, x_k^2, X_k*y_mean_k, y^2_mean_k)
    
    #     First get mean estimates for sigma_eta calculation
    moments_func = L96_obs_func_a_mean
    noisy_y, _ = moments_func(nothing, noise_mag, l96_params, y_matr=model_output)
    
    return noisy_y
end

function chaotic_spread_test(repeats, noise_mags, run_lengths)
    noise_n = length(noise_mags)
    run_lengths_n = length(run_lengths)
    timestep = .1
    moments_dim = 5

    # Initialize the output array
    ergodic_loss_outputs = zeros(repeats, noise_n, run_lengths_n, moments_dim)

    # Run the model for each combination of noise and run length and number of repeats
    for i in 1:repeats
        println("Running repeat $(i) of $(repeats)")
        for j in 1:noise_n
            noise = noise_mags[j]
            for k in 1:run_lengths_n
                t_end = run_lengths[k]
                toss = Int(t_end/10)
                ergodic_loss_outputs[i, j, k, :] = ergodic_run_moments(t_end, timestep, toss, noise, print_output=false)
            end
        end
    end

    combinations = Int(repeats * (repeats - 1) / 2)
    differences = zeros(combinations, noise_n, run_lengths_n, moments_dim)

    # Calculate the differences between each pair of runs
    for j in 1:noise_n
        for k in 1:run_lengths_n
            combo = 1
            for i in 1:repeats
                for n in i+1:repeats
                    differences[combo, j, k, :] = abs.(ergodic_loss_outputs[i, j, k, :] - ergodic_loss_outputs[n, j, k, :])
                    combo += 1
                end
            end
        end
    end

    return ergodic_loss_outputs, differences

end

function plot_ergodic_loss(differences, noise_mags, run_lengths, save_dir)
    y_labels = [L"y_1", L"y_2", L"y_3", L"y_4", L"y_5"]

    # Calculate mean, min, and max
    mean_differences = mean(differences, dims=1)
    min_differences = minimum(differences, dims=1)
    max_differences = maximum(differences, dims=1)

    noise_n = length(noise_mags)
    moments_dim = size(differences, 4)
    
    for i in 1:noise_n
        plot(dpi=300)
        for j in 1:moments_dim
            mean_data = vec(mean_differences[:, i, :, j])
            min_data = vec(min_differences[:, i, :, j])
            max_data = vec(max_differences[:, i, :, j])
            # Plot
            plot!(run_lengths, mean_data, yerror=(mean_data - min_data, max_data - mean_data), label=y_labels[j], legend=:outertopright, markerstrokecolor=:auto, xaxis=:log, yaxis=:log)
    
        end
        xlabel!("Run Length")
        ylabel!("Chaotic Spread")
        title!("Chaotic Spread of Lorenz '96 Outputs\nNoise Magnitude = $(noise_mags[i])")
        display(plot!(show=true))
        # convert noise_mags to string with scientific notation
        noise_mags_str = @sprintf("%.0e", noise_mags[i])
        savefig(joinpath(save_dir, "chaotic_spread_noise_$(noise_mags_str).png"))
    end

    # Normalized Differences
    mean_of_shortest_runs_no_noise = mean(differences, dims=1)[1, 1, 1, :]
    reshaped_divisor = reshape(mean_of_shortest_runs_no_noise, 1, 1, 1, 5)
    normalized_differences = differences ./ reshaped_divisor

    # Calculate mean, min, and max
    mean_differences = mean(normalized_differences, dims=1)
    min_differences = minimum(normalized_differences, dims=1)
    max_differences = maximum(normalized_differences, dims=1)

    for i in 1:noise_n
        plot(dpi=300)
        for j in 1:moments_dim
            mean_data = vec(mean_differences[:, i, :, j])
            min_data = vec(min_differences[:, i, :, j])
            max_data = vec(max_differences[:, i, :, j])
            # Plot
            plot!(run_lengths, mean_data, yerror=(mean_data - min_data, max_data - mean_data), label=y_labels[j], legend=:outertopright, markerstrokecolor=:auto, xaxis=:log, yaxis=:log)
        end
        xlabel!("Run Length")
        ylabel!("Normalized Chaotic Spread")
        title!("Normalized Chaotic Spread of Lorenz '96 Outputs\nNoise Magnitude = $(noise_mags[i])")
        display(plot!(show=true))

        # convert noise_mags to string with scientific notation
        noise_mags_str = @sprintf("%.0e", noise_mags[i])
        savefig(joinpath(save_dir, "normalized_chaotic_spread_noise_$(noise_mags_str).png"))
    end
end

function ergodic_test(t_end, timestep, sigma_eta_mag, C_0_mag, uki_iters; prior_theta = [0.1, 5, 2, 7], obs_noise = .01, toss=0, plot_progress=false, data_filename=nothing, plot_filename=nothing, loss_plot_filename=nothing, show_plot=true)
    # L96 parameters
    println("t_end = $(t_end), timestep = $(timestep), sigma_eta_mag = $(sigma_eta_mag), C_0_mag = $(C_0_mag), noise = $(obs_noise), toss = $(toss)")

    theta_names  = ["h" "F" "c" "b"]

    theta_true = [1.0, 10.0, log(10), 10.0]

    # L96 run without noise for reference
    noise_mag = 0.0

    # Initial conditions are stored here
    l96_params = L96_two_params(noise=noise_mag, t_end=t_end, toss=toss, timestep=timestep)

    # Don't transform observations yet because we will need them as IC's 
    moments_func = L96_no_moments

    # These transforms only the "c" parameter via an absolute value. 
    # So the L96 model always recieves abs(c)
    transform_func_to_UKI = L96_trans_C2_to_UKI
    transform_func_to_L96 = L96_trans_C2_to_L96

    # This doesn't do anything in this case
    # Could be important with a different transform or theta_true
    trans_theta_true = transform_func_to_UKI(theta_true)

    # This prepends K and J (defining the dimensionality of the L96 system)
    # Default values are K=36, J=10 
    KJ_theta_true = vcat([l96_params.K, l96_params.J], trans_theta_true)

    model_output = L96_model_full_output(KJ_theta_true, moments_func, transform_func_to_L96, l96_params);

    # Tranform observations 
    # (1/K)Sum_over_k(X_k, y_mean_k, x_k^2, X_k*y_mean_k, y^2_mean_k)
    moments_func = L96_obs_func_a_mean
    noisy_y, _ = moments_func(nothing, obs_noise, l96_params, y_matr=model_output)

    #     println("noisy_y")
    #     display(noisy_y)
    
    # This basically creates a dummy first observation that won't get used
    noisy_y = hcat(noisy_y, noisy_y)

    l96_params.initial_conditions = model_output[:, 1]
    l96_params.noise = obs_noise

    Sigma_eta = diagm(noisy_y[:,1]) .* sigma_eta_mag
    

    
    batch_size = 1

    model_func_1 = batch_lorenz_96_model_part_1!
    model_func_2 = batch_lorenz_96_model_part_2

    gamma = 1

    m_0 = transform_func_to_UKI(prior_theta)

    C_0 = diagm(transform_func_to_UKI([1, 10, .1 ,1] * C_0_mag))

    uki_params = UKI_params(noisy_y, Sigma_eta, m_0, C_0, 
        gamma, uki_iters, model_func_1, model_func_2, l96_params, 
        moments_func, theta_uki_to_theta, batch_size=batch_size, param_save_filepath = data_filename)
    
    title = "Ergodic L96 Calibration\n" * 
            "T=" * string(l96_params.t_end) * L", O(\textrm{Noise}) = " * string(@sprintf("%.2e", obs_noise)) * L", " *
            L"O(\sigma_\eta) = " * string(@sprintf("%.2e", sigma_eta_mag)) * L", " *
            L"O(C_0) = " * string(@sprintf("%.2e", C_0_mag))

    try
        for i in 1:uki_params.max_iter
            
            if i%10 == 0 && plot_progress
                println(i)
                plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=true, title_str=title)
            end

            uki_iteration_part1!(uki_params)
            fail_bool = uki_iteration_part2!(uki_params)
            if fail_bool
                println("UKI failed at iteration $(i)")
                break
            end
        end   
    catch error
        display(error)
    end

    # Convert t_end, timestep, noise, sigma_eta_mag and C_0_mag to string with scientific notation
    # parameter_string = @sprintf("%.0e", t_end) * "_" * @sprintf("%.0e", timestep) * "_" * @sprintf("%.0e", obs_noise) * "_" * @sprintf("%.0e", sigma_eta_mag) * "_" * @sprintf("%.0e", C_0_mag)
    if show_plot
        plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title)
    end
    if !isnothing(plot_filename)
        plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title, filename=plot_filename)
    end

    # plot_loss_iterations(uki_params, log=true, title_str=title)
    # if !isnothing(loss_plot_filename)
    #     plot_loss_iterations(uki_params, log=true, title_str=title, filename=loss_plot_filename)
    # end


    stability_thresholds = [0.01, 0.015, 0.03]
    stable_iters = zeros(length(stability_thresholds))
    costs = zeros(length(stability_thresholds))
    costs .= -1
    for i in 1:length(stability_thresholds)
        stable_iters[i] = stability_test(stability_thresholds[i], theta_true, uki_params)
        println("For threshold $(stability_thresholds[i]), stable iteration: $(stable_iters[i])")
        if stable_iters[i] > 1
            costs[i] = cost_UKI(t_end, stable_iters[i], batch_size=1, param_count=length(theta_true), parallel=false)
            println("Cost: $(costs[i])")
        end
    end

    true_theta_reached = true_theta_test(0.03, theta_true, uki_params)
    if true_theta_reached
        println("True theta reached")
    else
        println("True theta not reached")
    end

    println("**************************************************")

    return uki_params, stable_iters, costs, true_theta_reached
end

# Plot ergodic test from saved parameter file
function plot_ergodic_test_from_file(data_filename, plot_filename, loss_plot_filename, title, t_end; theta_true = [1.0, 10.0, log(10), 10.0], theta_names = ["h" "F" "c" "b"], theta_uki_to_theta = L96_trans_C2_to_L96, show_plot=true)
    uki_params = load(data_filename, "params")
    if show_plot
        plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title)
    end
    if !isnothing(plot_filename)
        plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title, filename=plot_filename)
    end
    # plot_loss_iterations(uki_params, log=true, title_str=title)
    # plot_loss_iterations(uki_params, log=true, title_str=title, filename=loss_plot_filename)

    stability_thresholds = [0.01, 0.015, 0.03]
    stable_iters = zeros(length(stability_thresholds))
    costs = zeros(length(stability_thresholds))
    costs .= -1
    for i in 1:length(stability_thresholds)
        stable_iters[i] = stability_test(stability_thresholds[i], theta_true, uki_params)
        println("For threshold $(stability_thresholds[i]), stable iteration: $(stable_iters[i])")
        if stable_iters[i] > 1
            costs[i] = cost_UKI(t_end, stable_iters[i], batch_size=1, param_count=length(theta_true), parallel=false)
            println("Cost: $(costs[i])")
        end
    end

    true_theta_reached = true_theta_test(0.03, theta_true, uki_params)
    if true_theta_reached
        println("True theta reached")
    else
        println("True theta not reached")
    end
    println("**************************************************")
    return uki_params, stable_iters, costs, true_theta_reached
end

# Function to measure stability of UKI with threshold.
# Will test if difference between estimated theta and true theta is less than threshold.
# If there is an iteration where the difference is less than threshold, then we will test is that remains true through the rest of the iterations.
# We require that the outputs remain stable for as long 2x as long as it took to get to the first stable iteration.
function stability_test(threshold, theta_true, uki_params)
    transform_func = uki_params.transform_func
    stable = false
    #iterate over of the uki iterations
    for i in 1:uki_params.max_iter
        #check if the difference between estimated theta and true theta is less than threshold
        # Do this for each parameter
        param_stable = true
        for j in 1:length(theta_true)
            # difference as percentage of true theta 
            diff = abs(theta_true[j] - transform_func(uki_params.m_s[i, j])) / theta_true[j]
            # println("iter $(i), param $(j), diff = $(diff)")
            if diff > threshold
                param_stable = false
            end
        end
        if param_stable
            # println("param stable at iter $(i)")
            initial_stable_iter = i
            #check if the difference remains less than threshold through the rest of the iterations
            for k in i+1:uki_params.max_iter
                param_stable = true
                for j in 1:length(theta_true)
                    # difference as percentage of true theta 
                    diff = abs(theta_true[j] - transform_func(uki_params.m_s[k, j])) / theta_true[j]
                    # println("iter $(k), param $(j), diff = $(diff)")
                    if diff > threshold
                        param_stable = false
                    end
                end
                if param_stable
                    #check if the algorithm has been stable for 2x as long as it took to get to the first stable iteration
                    if (k - initial_stable_iter) > (2 * initial_stable_iter)
                        return initial_stable_iter
                    end
                else
                    # println("param unstable at iter $(k)")
                    break
                end
            end
        end
    end
    println("No stable iteration found")
    return -1
end

# Function to check if UKI reached true theta (within threshold)
function true_theta_test(threshold, theta_true, uki_params)
    transform_func = uki_params.transform_func

    for i in 1:uki_params.max_iter
        met_threshold = true
        for j in 1:length(theta_true)
            diff = abs(theta_true[j] - transform_func(uki_params.m_s[i, j])) / theta_true[j]
            if diff > threshold
                met_threshold = false
                break
            end
        end

        if met_threshold
            return true
        end
    end
    return false
end

# Function to calculate cost of UKI
function cost_UKI(t_end, uki_iters; batch_size=1, param_count=1, parallel=false)
    if parallel
        return t_end * uki_iters * param_count
    else
        return t_end * uki_iters * batch_size * param_count
    end
end

function batch_test(t_end, timestep, batch_size, sigma_eta_mag, C_0_mag, uki_iters; prior_theta = [0.1, 5, 2, 7], obs_noise = .01, print_iter=true, batch_increment = nothing, learning_rate_tuple = nothing, data_filename=nothing, plot_filename=nothing, loss_plot_filename=nothing, plot_from_data=false, show_plot=true, title_str=nothing)

    # L96 parameters
    println("t_end = $(t_end), timestep = $(timestep), batch_size = $(batch_size), sigma_eta_mag = $(sigma_eta_mag), noise = $(obs_noise), C_0_mag = $(C_0_mag), learning_rate_tuple = $(learning_rate_tuple)")

    theta_names  = ["h" "F" "c" "b"]

    theta_true = [1.0, 10.0, log(10), 10.0]

    if plot_from_data
        # load data from data_filename
        # check if data_filename is a valid file
        if !isfile(data_filename)
            println("data_filename is not a valid file")
            return
        end
        # check if data_filename is a valid UKI_params file
        uki_params = load(data_filename, "params")
        l96_params = uki_params.model_params

        if isnothing(title_str)
            title = "Batch L96 Calibration, Batch Size = $(batch_size)\n" * 
                    "T=" * string(l96_params.t_end) * L", O(\textrm{Noise}) = " * string(@sprintf("%.2e", obs_noise)) * L", " *
                    L"O(\sigma_\eta) = " * string(@sprintf("%.2e", sigma_eta_mag)) * L", " *
                    L"O(C_0) = " * string(@sprintf("%.2e", C_0_mag))
        else
            title = title_str
        end

        # stop = nothing
        stop = 150

        if show_plot
            plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title, stop=stop)
        end
        if !isnothing(plot_filename)
            plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title, filename=plot_filename, stop=stop)
        end

        stability_thresholds = [0.01, 0.015, 0.03]
        stable_iters = zeros(length(stability_thresholds))
        costs = zeros(length(stability_thresholds))
        costs .= -1
        for i in 1:length(stability_thresholds)
            stable_iters[i] = stability_test(stability_thresholds[i], theta_true, uki_params)
            println("For threshold $(stability_thresholds[i]), stable iteration: $(stable_iters[i])")
            if stable_iters[i] > 1
                costs[i] = cost_UKI(timestep, stable_iters[i], batch_size=batch_size, param_count=length(theta_true), parallel=false)
                println("Cost: $(costs[i])")
            end
        end

        true_theta_reached = true_theta_test(0.03, theta_true, uki_params)
        if true_theta_reached
            println("True theta reached")
        else
            println("True theta not reached")
        end

        println("**************************************************")

        return uki_params, stable_iters, costs, true_theta_reached
    end
    
    toss=0

    # L96 run without noise for reference
    noise_mag = 0.0

    # Initial conditions are stored here
    l96_params = L96_two_params(noise=noise_mag, t_end=t_end, toss=toss, timestep=timestep)

    # Don't transform observations yet because we will need them as IC's 
    moments_func = L96_no_moments

    # These transforms only the "c" parameter via an absolute value. 
    # So the L96 model always recieves abs(c)
    transform_func_to_UKI = L96_trans_C2_to_UKI
    transform_func_to_L96 = L96_trans_C2_to_L96

    # This doesn't do anything in this case
    # Could be important with a different transform or theta_true
    trans_theta_true = transform_func_to_UKI(theta_true)

    # This prepends K and J (defining the dimensionality of the L96 system)
    # Default values are K=36, J=10 
    KJ_theta_true = vcat([l96_params.K, l96_params.J], trans_theta_true)

    model_output = L96_model_full_output(KJ_theta_true, moments_func, transform_func_to_L96, l96_params);

    println("Ground Truth Model Complete")

    noise_mag = obs_noise
    #     First get mean estimates for sigma_eta calculation
    moments_func = L96_obs_func_a_mean
    noisy_y, _ = moments_func(nothing, noise_mag, l96_params, y_matr=model_output)
    noisy_y = hcat(noisy_y, noisy_y)
    Sigma_eta_base = diagm(noisy_y[:,1]) 
    Sigma_eta = Sigma_eta_base .* sigma_eta_mag
    #     display(Sigma_eta)
    
    # Transform observations 
    # (1/K)Sum_over_k(X_k, y_mean_k, x_k^2, X_k*y_mean_k, y^2_mean_k)
    moments_func = L96_obs_func_a
    noisy_y, noise_ranges = moments_func(nothing, noise_mag, l96_params, y_matr=model_output)
    
    #     display(noise_ranges)

    # This basically creates a dummy first observation that won't get used (for ergodic)
    # noisy_y = hcat(noisy_y, noisy_y)

    l96_params.initial_conditions = model_output
    l96_params.noise = noise_mag
    l96_params.noise_ranges = noise_ranges
    l96_params.t_end = timestep

    #     Sigma_eta = diagm(noisy_y[:,1]) .* sigma_eta_mag
    #     display(Sigma_eta)

    model_func_1 = batch_lorenz_96_model_part_1!
    model_func_2 = batch_lorenz_96_model_part_2

    gamma = 1

    m_0 = transform_func_to_UKI(prior_theta)

    # THE BIG CHANGE!!!!
    C_0 = diagm(transform_func_to_UKI([1, 10, .1 ,1] * C_0_mag))

    param_save_filepath = data_filename

    uki_params = UKI_params(noisy_y, Sigma_eta, m_0, C_0, 
        gamma, uki_iters, model_func_1, model_func_2, l96_params, 
        moments_func, theta_uki_to_theta, batch_size=batch_size, param_save_filepath = param_save_filepath)
    
    learning_rate = 1

    try
        for i in 1:uki_params.max_iter
            
            if !isnothing(learning_rate_tuple)
                tau, eps_0, eps_tau = learning_rate_tuple
                alpha = i / tau
                learning_rate = (1 - alpha) * eps_0 + alpha * eps_tau
                if learning_rate > eps_tau
                    learning_rate = eps_tau
                end
                # println("learning rate: $(learning_rate)")
                uki_params.Sigma_eta = Sigma_eta_base .* 10.0^learning_rate
            end
            
            if i%100 == 0 && print_iter
                println("iter $(i)")
                println("batch size: $(uki_params.batch_size)")
                if !isnothing(learning_rate_tuple)
                    println("learning rate: $(learning_rate)")
                    println("sigma_eta magnitude: $(10.0^learning_rate)")
                else
                    println("sigma_eta magnitude: $(sigma_eta_mag)")
                end
            end
            
            uki_iteration_part1!(uki_params)
            uki_iteration_part2!(uki_params)
            
        end
    catch error
        display(error)
    end

    if isnothing(title_str)
        title = "Batch L96 Calibration, Batch Size = $(batch_size)\n" * 
                "T=" * string(l96_params.t_end) * L", O(\textrm{Noise}) = " * string(@sprintf("%.2e", obs_noise)) * L", " *
                L"O(\sigma_\eta) = " * string(@sprintf("%.2e", sigma_eta_mag)) * L", " *
                L"O(C_0) = " * string(@sprintf("%.2e", C_0_mag))
    else
        title = title_str
    end
    
    if show_plot
        plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title)
    end
    if !isnothing(plot_filename)
        plot_iterations(uki_params, theta_true, theta_names, theta_uki_to_theta, with_ribbon=false, title_str=title, filename=plot_filename)
    end

    stability_thresholds = [0.01, 0.015, 0.03]
    stable_iters = zeros(length(stability_thresholds))
    costs = zeros(length(stability_thresholds))
    costs .= -1
    for i in 1:length(stability_thresholds)
        stable_iters[i] = stability_test(stability_thresholds[i], theta_true, uki_params)
        println("For threshold $(stability_thresholds[i]), stable iteration: $(stable_iters[i])")
        if stable_iters[i] > 1
            costs[i] = cost_UKI(timestep, stable_iters[i], batch_size=batch_size, param_count=length(theta_true), parallel=false)
            println("Cost: $(costs[i])")
        end
    end

    true_theta_reached = true_theta_test(0.03, theta_true, uki_params)
    if true_theta_reached
        println("True theta reached")
    else
        println("True theta not reached")
    end
    println("**************************************************")

    # TODO need to amend this to plot loss based on ergodic run.
    #     plot_loss_iterations(uki_params)

    return uki_params, stable_iters, costs, true_theta_reached
end

# function to run L96 model without noise
function run_L96_trajectory(t_end, timestep, theta)
    l96_params = L96_two_params(noise=0.0, t_end=t_end, toss=0, timestep=timestep)

    # These transforms only the "c" parameter via an absolute value. 
    # So the L96 model always receives abs(c)
    transform_func_to_UKI = L96_trans_C2_to_UKI
    transform_func_to_L96 = L96_trans_C2_to_L96

    # Don't transform observations yet because we will need them as IC's 
    moments_func = L96_no_moments

    # This doesn't do anything in this case
    # Could be important with a different transform or theta_true
    trans_theta = transform_func_to_UKI(theta)

    # This prepends K and J (defining the dimensionality of the L96 system)
    # Default values are K=36, J=10 
    KJ_theta = vcat([l96_params.K, l96_params.J], trans_theta)

    model_output = L96_model_full_output(KJ_theta, moments_func, transform_func_to_L96, l96_params)

    # display(shape(model_output))

    return model_output, l96_params
end

# Function to demonstrate impact of noise on L96 trajectory
function plot_noisy_L96_trajectory(t_end, plot_length, timestep, theta, noise_mag; moments_func = L96_obs_func_a, filename=nothing, noise_on_both=false, previous_output=nothing)
    # Run the L96 model long to generate noise ranges
    # long_model_output, long_l96_params = run_L96_trajectory(1000, .01, theta)
    # long_noisy_y, noise_ranges = moments_func(nothing, noise_mag, long_l96_params, y_matr=long_model_output)

    # println("noise ranges: $(noise_ranges)")
    
    # Run the L96 model short for plotting
    # Use small timestep to get high fidelity version
    high_fidelity_timestep = .001
    if isnothing(previous_output)
        model_output, l96_params = run_L96_trajectory(t_end, high_fidelity_timestep, theta)
    else
        model_output = previous_output[1]
        l96_params = previous_output[2]
    end

    # Calculate the moments without noise
    if noise_on_both
        no_noise_y, _ = moments_func(nothing, noise_mag, l96_params, y_matr=model_output)
    else
        no_noise_y, _ = moments_func(nothing, 0.0, l96_params, y_matr=model_output)
    end

    # Calculate the moments with noise
    noisy_y, _ = moments_func(nothing, noise_mag, l96_params, y_matr=model_output)

    # Get number of moments and corresponding colors
    n_moments = size(noisy_y, 1)
    colors = distinguishable_colors(n_moments, [RGB(1,1,1), RGB(0,0,0)], dropseed=true)
    
    # Generate y_labels dynamically
    y_labels = [L"y_%$i" for i in 1:n_moments]

    # Plot the trajectories
    # Plot the different moments in same plot
    # Plot the noisy_y as dotted line
    # Plot the no_noise_y as solid line with markers in same color
    # x values are the same for all moments and are spaced by timestep
    # plot y_1 and y_3 on same plot
    # plot y_2 and y_4 on same plot
    # plot y_5 on its own plot
    # put the 3 plots next to each other

    # plot_length is length of section of trajectory to plot
    # taken from middle of trajectory to avoid transients
    interval = floor(Int, timestep / high_fidelity_timestep)
    start_index = floor(Int, ((t_end/timestep)/2 - (plot_length/timestep)/2)) * interval + 1
    end_index = floor(Int, plot_length/timestep) * interval + start_index
    x_values = range(0, plot_length, step=timestep)
    high_fidelity_x_values = range(0, plot_length, step=high_fidelity_timestep) 
    
    # Create subplot layout
    p1 = plot(dpi=300, layout=(1,3))
    
    # Plot y_1 and y_3 on first subplot
    subplot = 1
    for i in [1,3]
        # Plot high fidelity data as solid line with 30% opacity
        plot!(high_fidelity_x_values, no_noise_y[i, start_index:end_index], linestyle=:solid, color=colors[i], label=nothing, subplot=subplot, alpha=0.3)
        # Plot samples of noisy data with markers
        plot!(x_values, noisy_y[i, start_index:interval:end_index], marker=:circle, markersize=3, markerstrokewidth=0,
              color=colors[i], label=y_labels[i], subplot=subplot)
    end
    xlabel!("Time", subplot=1)
    ylabel!("Y", subplot=1)
    
    # Plot y_2 and y_4 on second subplot 
    subplot = 2
    for i in [2,4]
        # Plot high fidelity data as solid line with 30% opacity
        plot!(high_fidelity_x_values, no_noise_y[i, start_index:end_index], linestyle=:solid, color=colors[i], label=nothing, subplot=subplot, alpha=0.3)
        # Plot samples of noisy data with markers
        plot!(x_values, noisy_y[i, start_index:interval:end_index], marker=:circle, markersize=3, markerstrokewidth=0,
              color=colors[i], label=y_labels[i], subplot=subplot)
    end
    xlabel!("Time", subplot=2)
    ylabel!("Y", subplot=2) 

    # Plot y_5 on third subplot
    subplot = 3
    # Plot high fidelity data as solid line with 30% opacity
    plot!(high_fidelity_x_values, no_noise_y[5, start_index:end_index], linestyle=:solid, color=colors[5], label=nothing, subplot=subplot, alpha=0.3)
    # Plot samples of noisy data with markers
    plot!(x_values, noisy_y[5, start_index:interval:end_index], marker=:circle, markersize=3, markerstrokewidth=0,
          color=colors[5], label=y_labels[5], subplot=subplot)
    xlabel!("Time", subplot=3)
    ylabel!("Y", subplot=3) 
    
    # Add a layout title using plot! with a specific title placement
    plot!(
        plot_title="L96 Trajectories with $(@sprintf("%.0f%%", noise_mag*100)) Noise, Timestep = $(timestep)",
        plot_titlefontsize=12,
        top_margin=5Plots.mm,
        bottom_margin=5Plots.mm,  # Add bottom margin
        left_margin=5Plots.mm,    # Add left margin
        right_margin=5Plots.mm,
        size=(1000, 400),  # Set figure size (width, height)
    )

    display(plot!(show=true))
    if !isnothing(filename)
        # Check if file path exists
        dir = dirname(filename)
        if !isempty(dir) && !isdir(dir)
            mkdir(dir)
        end
        savefig(filename)
    end

    return model_output, l96_params
end

# Function to take df of batch results, determine whether a set of parameters was successful based on percent of runs that met stability threshold, and then return the mean values of the successful runs
function analyze_batch_results(df; save_dir=nothing, success_threshold=0.5)

    # get unique combinations of batch_size, sigma_eta_mag, C_0_mag, and noise_mag
    unique_params = unique(df[:, ["batch_size", "sigma_eta_mag", "C_0_mag"]])

    # display(unique_params)

    # get unique combinations of batch_size, sigma_eta_mag, C_0_mag, and noise_mag
    unique_scenarios = unique(df[:, ["batch_size", "sigma_eta_mag", "C_0_mag", "noise_mag"]])

    # create an empty dataframe to store results (batch_size, sigma_eta_mag, C_0_mag, noise_mag, percent_success, mean_threshold, mean_cost, parallel_cost)
    results = DataFrame(batch_size=Int[], sigma_eta_mag=Float64[], C_0_mag=Float64[], noise_mag=Float64[], percent_success=Float64[], mean_threshold=Float64[], mean_cost=Float64[], parallel_cost=Float64[])
    # for each unique set of parameters, determine whether the set was successful based on percent of runs that met stability threshold
    for i in 1:size(unique_scenarios, 1)
        params = unique_scenarios[i, :]

        # display(params)

        # get all rows in df with these hyperparameters
        rows = df[(df[:, "batch_size"] .== params.batch_size) .& 
                 (df[:, "sigma_eta_mag"] .== params.sigma_eta_mag) .& 
                 (df[:, "C_0_mag"] .== params.C_0_mag) .& 
                 (df[:, "noise_mag"] .== params.noise_mag), :]

        # calculate what percent of those runs met stability threshold (threshold_03 > 0)
        rows_with_success = rows[rows[:, "threshold_03"] .> 0, :]
        percent_success = size(rows_with_success, 1) / size(rows, 1)

        # display(percent_success)

        if percent_success > success_threshold
            # calculate mean values of the successful runs from the df
            mean_threshold = mean(rows_with_success[:, "threshold_03"])
            mean_cost = mean(rows_with_success[:, "threshold_03_cost"])
            # calculate parallel cost (mean_cost / batch_size)
            parallel_cost = mean_cost / params.batch_size
            # display((threshold=mean_threshold, cost=mean_cost, parallel_cost=parallel_cost))

            push!(results, (batch_size=params.batch_size, sigma_eta_mag=params.sigma_eta_mag, C_0_mag=params.C_0_mag, noise_mag=params.noise_mag, percent_success=percent_success, mean_threshold=mean_threshold, mean_cost=mean_cost, parallel_cost=parallel_cost))
        end
        # break
    end

    return results
end

# Function to take df of ergodic results, determine whether a set of parameters was successful based on percent of runs that met stability threshold, and then return the mean values of the successful runs
function analyze_ergodic_results(df; save_dir=nothing, success_threshold=0.5)
    # get unique combinations of t_end, sigma_eta_mag, C_0_mag, and noise_mag
    unique_params = unique(df[:, ["t_end", "sigma_eta_mag", "C_0_mag"]])

    # display(unique_params)

    # get unique combinations of batch_size, sigma_eta_mag, C_0_mag, and noise_mag
    unique_scenarios = unique(df[:, ["t_end", "sigma_eta_mag", "C_0_mag", "noise_mag"]])

    # create an empty dataframe to store results (batch_size, sigma_eta_mag, C_0_mag, noise_mag, percent_success, mean_threshold, mean_cost, parallel_cost)
    results = DataFrame(t_end=Float64[], sigma_eta_mag=Float64[], C_0_mag=Float64[], noise_mag=Float64[], percent_success=Float64[], mean_threshold=Float64[], mean_cost=Float64[])
    # for each unique set of parameters, determine whether the set was successful based on percent of runs that met stability threshold
    for i in 1:size(unique_scenarios, 1)
        params = unique_scenarios[i, :]

        # display(params)

        # get all rows in df with these hyperparameters
        rows = df[(df[:, "t_end"] .== params.t_end) .& 
                 (df[:, "sigma_eta_mag"] .== params.sigma_eta_mag) .& 
                 (df[:, "C_0_mag"] .== params.C_0_mag) .& 
                 (df[:, "noise_mag"] .== params.noise_mag), :]

        # calculate what percent of those runs met stability threshold (threshold_03 > 0)
        rows_with_success = rows[rows[:, "threshold_03"] .> 0, :]
        percent_success = size(rows_with_success, 1) / size(rows, 1)

        # display(percent_success)

        if percent_success > success_threshold
            # calculate mean values of the successful runs from the df
            mean_threshold = mean(rows_with_success[:, "threshold_03"])
            mean_cost = mean(rows_with_success[:, "threshold_03_cost"])
            # display((threshold=mean_threshold, cost=mean_cost))

            push!(results, (t_end=params.t_end, sigma_eta_mag=params.sigma_eta_mag, C_0_mag=params.C_0_mag, noise_mag=params.noise_mag, percent_success=percent_success, mean_threshold=mean_threshold, mean_cost=mean_cost))
        end
        # break
    end

    return results
end

# Plot uf_uki and ergodic uki l96 results
function plot_uf_uki_ergodic_results(;filename=nothing)
    # Data
    noise_mags = [.01, .05, .1]
    min_ergo_costs = [1900.0, 2100.0, 2760.0]
    min_uf_costs = [37.4, 97.6842, 991.0]
    min_uf_par_costs = [3.66, 7.4, 17.88]

    # Plot with markers on the line to show the values
    # use a nice color palette
    palette = :Set1_3

    # use log scale for y axis
    p1 = plot(noise_mags, min_ergo_costs, label="Ergodic UKI", palette=palette, dpi=300, marker=:circle, yscale=:log10, size=(8*100, 6*100), linewidth=3,
              guidefontsize=14, tickfontsize=12, legendfontsize=12)
    plot!(p1, noise_mags, min_uf_costs, label="UF-UKI", palette=palette, marker=:circle, yscale=:log10, linewidth=3)
    plot!(p1, noise_mags, min_uf_par_costs, label="UF-UKI Parallel", palette=palette, marker=:circle, yscale=:log10, linewidth=3)

    # place x ticks at the values of noise_mags and format as percentages
    xticks!(p1, noise_mags, ["$(@sprintf("%.0f%%", noise_mag*100))" for noise_mag in noise_mags])

    # place legend in the middle right
    plot!(p1, legend=:right)

    # Add labels to markers with the values of the costs
    for i in 1:length(noise_mags)
        # Adjust annotation positions to prevent cutoff
        # For ergodic costs, place text above the point with padding
        x_pos = i == 1 ? noise_mags[i] * 1.3 : noise_mags[i]
        annotate!(p1, x_pos, min_ergo_costs[i] * 1.1, text("$(@sprintf("%.0f", min_ergo_costs[i]))", :center, :bottom, 10, :black))
        # For UF costs, place text below the point with padding
        annotate!(p1, x_pos, min_uf_costs[i] * 0.9, text("$(@sprintf("%.3g", min_uf_costs[i]))", :center, :top, 10, :black))
        # For parallel costs, place text above the point with padding
        annotate!(p1, x_pos, min_uf_par_costs[i] * 1.1, text("$(@sprintf("%.3g", min_uf_par_costs[i]))", :center, :bottom, 10, :black))
    end

    # Add labels and title
    xlabel!(p1, "Noise Magnitude", guidefontsize=14)
    ylabel!(p1, "Computational Cost", guidefontsize=14)
    title!(p1, "Lorenz '96 UF-UKI Cost Comparison", titlefontsize=16)
    
    # Add some padding to the plot to prevent text cutoff
    plot!(p1, margin=10Plots.mm)
    
    display(p1)
    if !isnothing(filename)
        # Check if file path exists
        dir = dirname(filename)
        if !isempty(dir) && !isdir(dir)
            mkdir(dir)
        end
        savefig(filename)
    end
end
