# UKI V4
# Based on UKI-2 from Huang, D. Z., Huang, J., Reich, S. & Stuart, A. M. Efficient Derivative-free Bayesian Inference for Large-Scale Inverse Problems. arXiv:2204.04386 [cs, math] (2022).

using FileIO
using Parameters
using LinearAlgebra
using Plots
using StatsBase
using LaTeXStrings
# using Measures

@with_kw mutable struct UKI_params
    #     Problem definition
    y_true_full
    N_samples
    
    #Ongoing values
    m_s
    C_s
    thetas
    m_hat
    C_hat
    loss

    # iteration parameters
    max_iter
    current_iter
    
    # helper sizes
    N_theta
    N_out

    # Hyperparameters
    gamma

    # weights for sigma points
    a
    # weight matrix
    I_N_theta
    
    # Error covariance matrix
    # While constant in the original version, 
    # TODO make function that updates this to account for change to noise during multi-time
    Sigma_eta
    Sigma_nu

    # Artificial evolution covariance
    Sigma_w

    # model function, this might call into another language
    # Required format of function: model_func(thetas, sample_idxs, moments_func, transform_func, param_file)
    model_func_part_1
    model_func_part_2
    # Parameter file to be passed into model function, this might contain file paths for a model like Held Suarez
    model_params
    # Moments functions to be passed into model function moments_func(y, param_file)
    moments_func
    # transform_func to change theta to model space
    transform_func

    # Batching
    batch_size
    batch_sizes
    sample_idxs

    # learning rate
    learning_rates

    # save filepath
    param_save_filepath
    
    function UKI_params(y_true_full, Sigma_eta, m_0, C_0, gamma, max_iter, model_func_1, model_func_2, model_params, moments_func, transform_func; batch_size=10, param_save_filepath="/glade/work/teopb/hs_uki_cases/uki_params.jld2")
        # helper sizes
        N_theta = size(m_0, 1)
        N_out = size(y_true_full, 1)
        N_samples = size(y_true_full, 2)

        # Ongoing Values
        m_s = zeros(max_iter + 1, N_theta)
        C_s = zeros(max_iter + 1, size(C_0, 1), size(C_0, 2))
        m_s[1, :] = m_0
        C_s[1, :, :] = C_0

        sigma_count = N_theta * 2 + 1
        thetas = zeros(N_theta, sigma_count)
        loss = zeros(max_iter, N_out, sigma_count)

        # Hyperparameters
        kappa = 0
        alpha = min(sqrt(4 / (N_theta + kappa)), 1.)
        b = 2

        lambda = alpha^2 * (N_theta + kappa) - N_theta

        # weights for sigma points
        a = max(1/8, 1/(2 * N_theta))

        I_N_theta = zeros(N_theta, 2 * N_theta)
        I_N_theta[:, 1:N_theta] =  Matrix{Float64}(I, N_theta, N_theta) * (1/sqrt(2* a))
        I_N_theta[:, N_theta+1:2*N_theta] =  Matrix{Float64}(I, N_theta, N_theta) * (1/sqrt(2* a)) * -1
        
        # Error covariance matrix
        Sigma_nu = gen_Sigma_nu(gamma, Sigma_eta)

        # Artificial evolution covariance
        Sigma_w = C_0

        # Placeholder values
        m_hat = m_0
        C_hat = (gamma + 1) * C_0

        sample_idxs = zeros(Int64, batch_size)

        batch_sizes = zeros(Int64, max_iter)
        batch_sizes[1] = batch_size

        learning_rates = zeros(max_iter)
        learning_rates[1] = 1

        return new(y_true_full, N_samples, m_s, C_s, thetas, m_hat, C_hat, loss, max_iter, 1, N_theta, N_out, gamma, a, I_N_theta, Sigma_eta, Sigma_nu, Sigma_w, model_func_1, model_func_2, model_params, moments_func, transform_func, batch_size, batch_sizes, sample_idxs, learning_rates, param_save_filepath)
    end
end

# Create Sigma_nu from Sigma_eta, Sigma_0, and gamma according to:
# Sigma_nu = (gamma + 1)/gamma [Sigma_eta, 0; 0, Sigma_0]
# Sizes: Sigma_eta = [N_out, N_out]
# Sigma_0 = [N_theta, N_theta]
# Sigma_nu = [N_out+ N_theta, N_out+ N_theta]
function gen_Sigma_nu(gamma, Sigma_eta)
    N_out = size(Sigma_eta, 1)

    # generate zero matrix for Sigma_nu
    Sigma_nu = zeros(N_out, N_out)

    # Fill in respective blocks
    Sigma_nu[1:N_out, 1:N_out] = Sigma_eta

    # Multiple by gamma term
    Sigma_nu = ((gamma + 1)/gamma) * Sigma_nu

    return Sigma_nu
end


# calculate sigma_eta, used in covariance calculations from split y values
function Sigma_eta_and_mean(y_split)
    splits = size(y_split, 1)
    y_length = size(y_split, 2)

    sigma_eta = zeros(y_length, y_length)

    mean_y = mean(y_split, dims=1)[1, :]

    # print(size(mean_y))
    # print(size(y_split))
    # print(size(sigma_eta))

    for i in 1:splits
        sigma_eta += (y_split[i,:] - mean_y) * (y_split[i,:] - mean_y)'
    end

    sigma_eta ./= (splits - 1)

    return sigma_eta, mean_y

end

# Write UKI Params to file (must end in '.jld2')
function write_UKI_params(params, filename)
    if !isnothing(filename)
        save(filename, Dict("params" => params))
    end
end

# Read UKI Params from file
function read_UKI_params(filename)
    return load(filename, "params")
end


#Enforce Symmetry Pf = 1/2 (Pf + (Pf)T )
function sym_mat(X)
    return .5 * (X + transpose(X)) 
end

# UKI iteration

function uki_iteration_part1!(uki_params; debug=false, alt_C_hat=false)
    iter = uki_params.current_iter
    m = uki_params.m_s[iter, :]
    C = uki_params.C_s[iter, :, :]

    uki_params.batch_sizes[iter] = uki_params.batch_size

    if debug
        println("C")
        display(C)
        println()
    end
    
    # Prediction Step
    m_hat = m
    uki_params.m_hat = m_hat

    # Todo, decay here? ADAM style?
    if alt_C_hat
        C_hat = (uki_params.gamma + 1) * C
    else
        C_hat = C + uki_params.Sigma_w
    end
    # C_hat = (uki_params.gamma + 1) * C
    # C_hat = C + uki_params.Sigma_w

    # enforce symmetry
    # C_hat = sym_mat(C_hat)

    uki_params.C_hat = C_hat
    
    # Construct Ensemble
    sigma_count = uki_params.N_theta * 2 + 1
    thetas = zeros(uki_params.N_theta, sigma_count)
    
    # Generate Sigma Points
    thetas[:, 1] = m_hat
    
    if debug
        println("C_hat")
        display(C_hat)
        if isposdef(Hermitian(C_hat))
            println("C_hat is pos def")
        else
            println("C_hat is not pos def")
        end
    end

    # display(C_hat)

    C_cho = cholesky(Hermitian(C_hat)).L

    # display(C_cho)
    
    for i in 1:(sigma_count-1)
        thetas[:, i+1] = m_hat + (C_cho * uki_params.I_N_theta)[:, i]
    end

    uki_params.thetas = thetas

    # Select indices
    # Start at 2 to avoid initial condition at 0
    sample_idxs = sample(2:uki_params.N_samples, uki_params.batch_size, replace=false, ordered=true)
    uki_params.sample_idxs = sample_idxs

    if hasproperty(uki_params, :param_save_filepath)
        write_UKI_params(uki_params,uki_params.param_save_filepath)
    else
        write_UKI_params(uki_params,"/glade/work/teopb/hs_uki_cases/uki_params.jld2")
    end

    # pass thetas and sample_idxs to model_func_part_1
    uki_params.model_func_part_1(thetas, sample_idxs, uki_params.moments_func, uki_params.transform_func, uki_params.model_params)

end

function uki_iteration_part2!(uki_params; debug=false, batching=true, learning_rate = 1)
    # Helper variables
    iter = uki_params.current_iter
    sigma_count = uki_params.N_theta * 2 + 1

    # ensure sample_idxs are ints
    uki_params.sample_idxs = Int.(uki_params.sample_idxs)

    uki_params.learning_rates[iter] = learning_rate

    # Analysis step
    output_size = size(uki_params.y_true_full, 1)

    if batching

        x_s, fail_bool = uki_params.model_func_part_2(uki_params.sample_idxs, uki_params.moments_func, uki_params.N_theta, uki_params.N_out, uki_params.model_params)

        if fail_bool
            println("Model function part 2 failed")
            return true
        end

        if debug
            println("x_s size")
            display(size(x_s))
            display(x_s)
            println()

            # loss investigation
            println("Thetas:")
            display(uki_params.transform_func(uki_params.thetas))
            println("Index")
            println(uki_params.sample_idxs[1])
            println("Y_true:")
            display(uki_params.y_true_full[:, uki_params.sample_idxs[1]])
            println("Test")
            display(x_s[1, :, 1])
            println("Diff")
            display(uki_params.y_true_full[:, uki_params.sample_idxs[1]] - x_s[1, :, 1])
            # 
        end

        # save loss
        for i in 1:sigma_count
            loss = zeros(output_size, uki_params.batch_size)
            for j in 1:uki_params.batch_size
                loss[:, j] = uki_params.y_true_full[:, uki_params.sample_idxs[j]] - x_s[i, :, j]
            end    
            uki_params.loss[iter, :, i] = mean(abs.(loss[:, :]), dims=2)
        end

        if debug
            println("Losses")
            display(uki_params.loss)
        end
        
        C_theta_x = zeros(uki_params.N_theta, uki_params.N_out, uki_params.batch_size)
        C_xx = zeros(uki_params.N_out, uki_params.N_out, uki_params.batch_size)

        thetas = uki_params.thetas
        m_hat = uki_params.m_hat
        C_hat = uki_params.C_hat

        if debug
            println("C_hat")
            display(C_hat)
            println()
        end

        for i in 2:sigma_count
            for j in 1:uki_params.batch_size
                C_theta_x[:, :, j] += uki_params.a * (thetas[:, i] - m_hat) * transpose(x_s[i, :, j] - x_s[1, :, j])
                C_xx[:, :, j] += uki_params.a * (x_s[i, :, j] - x_s[1, :, j]) * transpose(x_s[i, :, j] - x_s[1, :, j])
            end
        end

        # debug print C_theta_x and C_xx
        if debug
            println("C_theta_x")
            display(C_theta_x)
            println()
            println("C_xx before noise")
            display(C_xx)
            println()
        end

        # Update Sigma_nu
        uki_params.Sigma_nu = gen_Sigma_nu(uki_params.gamma, uki_params.Sigma_eta)


        # println("Sigma_eta")
        # display(uki_params.Sigma_eta)
        # println()
        # println("Sigma_nu")
        # display(uki_params.Sigma_nu)
        # println()

        # Add noise to C_xx
        for j in 1:uki_params.batch_size
            C_xx[:, :, j] += uki_params.Sigma_nu
        end

        # debug print C_xx post noise
        if debug
            println("C_xx after noise")
            display(C_xx)
            println()
        end

        batch_ms = zeros(uki_params.N_theta, uki_params.batch_size)
        batch_Cs = zeros(uki_params.N_theta, uki_params.N_theta, uki_params.batch_size)
        # Right hand solve
        for j in 1:uki_params.batch_size
            tmp = C_theta_x[:, :, j]/C_xx[:, :, j]

            batch_ms[:, j] = m_hat + tmp * (uki_params.y_true_full[:, uki_params.sample_idxs[j]] - x_s[1, :, j])

            batch_Cs[:, :, j] = C_hat - tmp * transpose(C_theta_x[:, :, j])

            if debug
                # tmp matrix
                println("Update Matrix")
                display(tmp * (uki_params.y_true_full[:, uki_params.sample_idxs[j]] - x_s[1, :, j]))
                println()
            end
        end

        uki_params.m_s[iter+1, :] = mean(batch_ms, dims=2)
        
        uki_params.C_s[iter+1, :, :] = mean(batch_Cs, dims=3)

        if debug
            println("m_s")
            display(uki_params.m_s[iter+1,:])
            println()
        end

    else
        x_s = uki_params.model_func_part_2(uki_params.sample_idxs, uki_params.moments_func, uki_params.N_theta, uki_params.N_out, uki_params.model_params)
    
        if debug
            println("x_s size")
            display(size(x_s))
            println()
        end
    
        # save loss
        for i in 1:sigma_count
            uki_params.loss[iter,:,i] = abs.(uki_params.y_true_full - x_s[i,:])
        end
        
        C_theta_x = zeros(uki_params.N_theta, uki_params.N_out)
        C_xx = zeros(uki_params.N_out, uki_params.N_out)
    
        thetas = uki_params.thetas
        m_hat = uki_params.m_hat
        C_hat = uki_params.C_hat
    
        if debug
            println("C_hat")
            display(C_hat)
            println()
        end
    
        for i in 2:sigma_count
            C_theta_x += uki_params.a * (thetas[:, i] - m_hat) * transpose(x_s[i, :] - x_s[1, :])
            C_xx += uki_params.a * (x_s[i, :] - x_s[1, :]) * transpose(x_s[i, :] - x_s[1, :])
        end
        
        C_xx += uki_params.Sigma_nu
    
        # Right hand solve
        tmp = C_theta_x/C_xx
    
        uki_params.m_s[iter+1, :] = m_hat + tmp * (uki_params.y_true_full - x_s[1, :])
        
        uki_params.C_s[iter+1, :, :] = C_hat - tmp * transpose(C_theta_x)
    end

    if debug
        println("C")
        display(uki_params.C_s[iter+1,:, :])
        println()
    end

    if debug
        println("Iteration")
        display(iter)
        println("Thetas")
        display(thetas)
        println("End iteration")
        println("Updated m_s")
        display(uki_params.m_s[iter+1, :])
    end

    uki_params.current_iter = iter + 1

    if hasproperty(uki_params, :param_save_filepath)
        write_UKI_params(uki_params,uki_params.param_save_filepath)
    else
        write_UKI_params(uki_params,"/glade/work/teopb/hs_uki_cases/uki_params.jld2")
    end

    # println("UKI_part_2 complete! current_iter=$(uki_params.current_iter)")

    return false
    
end

function plot_iterations(uki_params, theta_true, theta_names, transform_func; m_s=nothing, filename=nothing, title_str=nothing, with_ribbon=true, x_computation=nothing, vert_line=nothing, stop=nothing)

    line_width = 2
    if m_s === nothing
        m_s = uki_params.m_s
        C_s = uki_params.C_s

        transformed_m_s = zeros(size(m_s))
        transformed_C_s = zeros(size(m_s))

        # display(C_s[1, :, :])

        if stop === nothing
            stop = size(m_s, 1)
        end

        # stop = 4

        for i in 1:stop
            transformed_m_s[i, :] = transform_func(m_s[i, :])
            transformed_C_s[i, :] = transform_func(diag(C_s[i, :, :]))
        end
        
        transformed_m_s = transformed_m_s[1:stop, :]
        transformed_C_s = transformed_C_s[1:stop, :]
    else
        if stop === nothing
            stop = size(m_s, 1)
        end

        transformed_m_s = zeros(size(m_s))

        for i in 1:stop
            transformed_m_s[i, :] = transform_func(m_s[i, :])
        end
        C_s = nothing
    end

    # display(transformed_m_s)
    # display(transformed_C_s)

    title = "Parameters by Iteration"
    titlefontsize = 14

    if !isnothing(title_str)
        title = title_str
        if length(title) > 30
            titlefontsize = 10
        end
    end

    if size(theta_true, 1) == 3
        palette = :Set1_3
    elseif size(theta_true, 1) == 4
        palette = :Set1_4
    else
        palette = :tab10
    end

    # display(transformed_m_s)

    if x_computation === nothing
        if with_ribbon
            plot(transformed_m_s, ribbon = transformed_C_s, fillalpha=.2, label = theta_names, palette = palette,  linewidth=line_width, dpi=300)
        else
            plot(transformed_m_s, fillalpha=.2, label = theta_names, palette = palette, linewidth=line_width, dpi=300)
        end

        for i in 1:size(theta_true, 1)
            hline!([theta_true[i]], label=theta_names[1, i]*" true", line=(:dash), palette = palette, lw=line_width)
        end
        plot!(title = title, xlabel = "Iteration", ylabel = "Parameter Value", legend = :outertopright, titlefontsize = titlefontsize)
        if vert_line !== nothing
            vline!([vert_line], lw=line_width, line=(:dash), linecolor=:black, label="Convergence")
        end

    else
        if with_ribbon
            plot(x_computation, transformed_m_s, ribbon = transformed_C_s, fillalpha=.2, label = theta_names, palette = palette, linewidth=line_width, dpi=300)
        else
            plot(x_computation, transformed_m_s, fillalpha=.2, label = theta_names, palette = palette, lw=line_width, dpi=300)
        end

        for i in 1:size(theta_true, 1)
            hline!([theta_true[i]], label=theta_names[1, i]*" true", line=(:dash), palette = palette, lw=line_width)
        end
        plot!(title = title, xlabel = "Computational Cost", ylabel = "Parameter Value", legend = :outertopright, titlefontsize = titlefontsize,  linewidth=line_width)

        if vert_line !== nothing
            vline!([x_computation[vert_line]], lw=2, line=(:dash), linecolor=:black, label="Convergence", line_width=line_width, dpi=300)
        end
    end

    if filename === nothing
        display(plot!(show=true))
    else
        savefig(filename)
    end
end


function plot_loss_iterations(uki_params; log=false, filename=nothing, title_str=nothing, computation_xaxis=nothing, stop=nothing)
    m_s = uki_params.m_s
    N_out = size(uki_params.loss, 2)  # Get number of output dimensions

    if stop === nothing
        stop = size(m_s, 1) - 1
    end

    title = "Loss by iteration"
    titlefontsize = 14

    if !isnothing(title_str)
        title = title * ": " * title_str
        if length(title) > 30
            titlefontsize = 10
        end
    end

    # Initialize plot with first dimension
    if log
        p = plot(uki_params.loss[1:stop, 1, 1], yaxis=:log, label=L"y_1", dpi=300)
    else
        p = plot(uki_params.loss[1:stop, 1, 1], label=L"y_1", dpi=300)
    end

    # Add remaining dimensions
    for dim in 2:N_out
        plot!(uki_params.loss[1:stop, dim, 1], label=L"y_%$dim", dpi=300)
    end

    plot!(title = title, xlabel = "Iteration", ylabel = "Loss", titlefontsize = titlefontsize, legend = :outertopright, dpi=300)

    if filename === nothing
        display(plot!(show=true))
    else
        savefig(filename)
    end
end

function plot_loss_at(uki_params, iter; filename=nothing, title_str=nothing)
    m_s = uki_params.m_s

    stop = size(m_s, 1) - 1

    title = "Loss at iter=" * string(iter)
    titlefontsize = 14

    if !isnothing(title_str)
        title = title * ": " * title_str
        if length(title) > 30
            titlefontsize = 10
        end
    end


    plot(uki_params.loss[iter, :])
    plot!(title = title, xlabel = "Iteration", ylabel = "RMSE", titlefontsize = titlefontsize)

    if filename===nothing
        plot!()
    else
        png(filename)
    end
end

function plot_mean_iterations(m_s, y_true, moments_func, val_names; title_str=nothing)
    iters = size(m_s, 1)
    obs_dims = size(val_names, 2)
    means = zeros(iters, obs_dims)
    
    for i in 1:iters
    means[i, :], _, _ = lorenz_model(m_s[i, :], moments_func=moments_func)
    end
    
    plot(means, label = val_names)
    
    for i in 1:size(y_true, 1)
        hline!([y_true[i]], label=val_names[1, i]*" true", line=(:dash))
    end

    title = "Observation means by iteration"

    if !isnothing(title_str)
        title = title * ": " * title_str
    end

    plot!(title = title, xlabel = "Iteration", ylabel = "Observation Value")
    plot!()
end

function plot_iterations_multiple(m_s, C_s, theta_true, theta_names)
    run_count = size(m_s, 1)
    plot(abs.(m_s[1, :, :]), ribbon = C_s[1, :, :], fillalpha=.1, legend=false)
    for j in 2:run_count
        plot!(abs.(m_s[j, :, :]), ribbon = C_s[j, :, :], fillalpha=.1, legend=false)
    end
    for i in 1:size(theta_true, 1)
        hline!([theta_true[i]], label=theta_names[1, i]*" true", line=(:dash))
    end
    plot!()
end

function super_sample(theta, diag_C, N_out, model_func, moments_func, transform_func, param_file; n_sample=10)
    n_theta = size(theta,1)
    
    #     Create random matrix n_sample x n_theta
    random_params = rand(Normal(0,1), (n_sample, n_theta))
    
    #     normalize to lie on unit hypersphere
    foreach(normalize!, eachrow(random_params))
    
    #     scale by C values (this C should be much smaller than C used in UKI, it is size of sampling region)
    scaled_params = diag_C' .* random_params

    # Offset by center point
    scaled_params = scaled_params .+ theta'
    
    #     append center point
    scaled_params = vcat(scaled_params, theta')

    # display(scaled_params)
    
    #     create matrix to hold outputs
    outputs = zeros(n_sample + 1, N_out)
    
    Threads.@threads for i in 1:n_sample+1
        outputs[i, :] = model_func(scaled_params[i, :], moments_func, transform_func, param_file)
    end
    
    # display(outputs)

    #     average
    avg_output = mean(outputs, dims=1)

    # display(avg_output)
    
    return avg_output'
end

# sample from normal instead of hyper-shell
function super_sample_v2(theta, diag_C, N_out, model_func, moments_func, transform_func, param_file; n_sample=10)
    n_theta = size(theta,1)

    C = diagm(diag_C)
    
    #     Create random matrix n_sample x n_theta
    d = MvNormal(theta, C)
    random_params = rand(d, n_sample)'
    
    # display(random_params)
    
    #    append center point
    random_params = vcat(random_params, theta')

    # display(random_params)
    
    #     create matrix to hold outputs
    outputs = zeros(n_sample + 1, N_out)
    
    Threads.@threads for i in 1:n_sample+1
        outputs[i, :] = model_func(random_params[i, :], moments_func, transform_func, param_file)
    end
    
    # display(outputs)

    #     average
    avg_output = mean(outputs, dims=1)

    # display(avg_output)
    
    return avg_output'
end

function RMSE(y1, y2)
    return sqrt(mean(((y1 - y2).^2)))
end

# Returns loss not output!
# sample from normal instead of hyper-shell
function super_sample_v3(y, theta, diag_C, N_out, model_func, moments_func, transform_func, param_file; n_sample=10)
    n_theta = size(theta,1)

    C = diagm(diag_C)
    
    #     Create random matrix n_sample x n_theta
    d = MvNormal(theta, C)
    random_params = rand(d, n_sample)'
    
    # display(random_params)
    
    #     append center point
    random_params = vcat(random_params, theta')

    # display(random_params)
    
    #     create matrix to hold outputs
    outputs = zeros(n_sample + 1, N_out)
    
    Threads.@threads for i in 1:n_sample+1
        outputs[i, :] = model_func(random_params[i, :], moments_func, transform_func, param_file)
    end
    
    # display(outputs)
    losses = zeros(n_sample + 1)

    for i in 1:n_sample+1
        losses[i] = RMSE(y, outputs[i, :])
    end

    #     average
    loss = mean(losses)

    # display(avg_output)
    
    return loss
end

# simple and hybrid together
function plot_iterations_two(uki_params_simple, uki_params_hybrid, theta_true, theta_names_long, theta_names_multi, transform_func; inline=false,title_str=nothing, filename=nothing, halfstep=false, ribbon=false)

    # simple
    m_s = uki_params_simple.m_s
    C_s = uki_params_simple.C_s

    transformed_m_s = zeros(size(m_s))
    transformed_C_s = zeros(size(m_s))

    # display(C_s[1, :, :])

    stop = size(m_s, 1)

    # stop = 4

    for i in 1:stop
        transformed_m_s[i, :] = transform_func(m_s[i, :])
        transformed_C_s[i, :] = transform_func(diag(C_s[i, :, :]))
    end
    
    transformed_m_s = transformed_m_s[1:stop, :]
    transformed_C_s = transformed_C_s[1:stop, :]

    # display(transformed_m_s)
    # display(transformed_C_s)

    title = ""
    titlefontsize = 14

    if !isnothing(title_str)
        title = title_str
        if length(title) > 40
            titlefontsize = 10
        end
    else
        title = "Parameters by Iteration"
    end

    if size(theta_true, 1) == 3
        palette = :Set1_3
    elseif size(theta_true, 1) == 4
        palette = :Set1_4
    else
        palette = :tab10
    end

    x = collect(0:stop-1)

    if ribbon
        plot(x, transformed_m_s, ribbon = transformed_C_s, fillalpha=.2, label = theta_names_long, palette = palette, left_margin=10mm)
    else
        plot(x, transformed_m_s, label = theta_names_long, palette = palette)
    end


    # hybrid
    m_s = uki_params_hybrid.m_s
    C_s = uki_params_hybrid.C_s

    transformed_m_s = zeros(size(m_s))
    transformed_C_s = zeros(size(m_s))

    stop = size(m_s, 1)


    for i in 1:stop
        transformed_m_s[i, :] = transform_func(m_s[i, :])
        transformed_C_s[i, :] = transform_func(diag(C_s[i, :, :]))
    end
    
    transformed_m_s = transformed_m_s[1:stop, :]
    transformed_C_s = transformed_C_s[1:stop, :]

    x = collect(0:.5:((stop-1)/2))

    if ribbon
        plot!(x, transformed_m_s, ribbon = transformed_C_s, fillalpha=.2, label = theta_names_multi, palette = palette, line=(:dashdot), left_margin=10mm)
    else
        plot!(x, transformed_m_s, label = theta_names_multi, palette = palette, line=(:dashdot))
    end

    for i in 1:size(theta_true, 1)
        hline!([theta_true[i]], label="", line=(:dash), palette = palette)
    end
    plot!(title = title, xlabel = "Iteration", ylabel = "Parameter Value", legend = :topright, titlefontsize = titlefontsize)

    if inline
        plot!()
    else
        # p = plot!(size=(1200, 800), dpi=300)
        # savefig(p, filename)
        savefig(filename)
    end
end