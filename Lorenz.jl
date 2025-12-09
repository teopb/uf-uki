using DifferentialEquations
using Distributions
using Random
using Statistics
using LinearAlgebra

mutable struct lorenz_params
    noise
    t_end
    toss
    timestep
    u0
    mean_output
    N
    # Unperturbed ground truth to use as initial conditions
    initial_conditions
    uki_thetas
    # Batch size
    batch_size

    function lorenz_params(noise, t_end, toss, timestep, u0, mean_output, N, batch_size)
        return new(noise, t_end, toss, timestep, u0, mean_output, N)
    end

    function lorenz_params(;noise=0.0,
        t_end=500.0,
        toss=0.0,
        timestep=.01,
        u0=[1.0,0.0,0.0],
        mean_output=true,
        N=3, 
        initial_conditions=nothing,
        uki_thetas=nothing,
        batch_size=1)
        return new(noise, t_end, toss, timestep, u0, mean_output, N, initial_conditions, uki_thetas, batch_size)
    end

    # constructor with default values
    function lorenz_params()
        noise=0.0
        t_end=500.0
        toss=0.0
        timestep=.01
        u0=[1.0,0.0,0.0]
        mean_output=true
        N=3
        initial_conditions=nothing
        uki_thetas=nothing
        batch_size=1
        return new(noise, t_end, toss, timestep, u0, mean_output, N, initial_conditions, uki_thetas, batch_size)
    end
end

function lorenz!(du, u, p, t)    
    a, r, b = p

    x, y, z = u

    du[1]= a*(y - x)
    du[2] = r*x - y - x*z
    du[3] = x*y - b*z
end


# Thanks to Stephen Rasp for L96 reference (https://github.com/raspstephan/Lorenz-Online/blob/master/L96.py)
mutable struct L96_two_params
    K
    J
    u0
    dt
    timestep
    t_end
    toss
    noise
    noise_ranges
    # Unperturbed ground truth to use as initial conditions
    initial_conditions
    uki_thetas
    transform_func_to_L96

    function L96_two_params(;K=36, J=10, X_init=nothing, Y_init=nothing,
        dt=0.001, timestep=0.1, t_end=500, toss=0, noise=0.0, noise_ranges=nothing, initial_conditions=nothing, uki_thetas=nothing, transform_func_to_L96 = L96_trans_C2_to_L96)

        if X_init === nothing
            d = Normal(0, 1)
            X_init = rand(d, K)
        end

        if Y_init === nothing
            d = Normal(0, .0001)
            Y_init = rand(d, K*J)
        end

        u0 = vcat(X_init, Y_init)

        return new(K, J, u0, dt, timestep, t_end, toss, noise, noise_ranges, initial_conditions, uki_thetas, transform_func_to_L96)
    end

end

# Translator to enable storing all X and Y in sequence
# Also handles periodicity aka X_(1-1) = X_K
# Y is periodic over the full J*K values, not over each J subset
function L96_index(K, J, k, j, level)
    if k == 0
        k = K
    elseif k == K
        k = K
    else
        k =  mod(k, K)
        # print(typeof(k))
    end

    if level == 1
        return trunc(Int, k)
    else
        if j == J
            j = J
        elseif j == 0 && k == 1
            j = J
            k = K
        else
            # print(typeof(K + mod((k-1) * J + j, J * K)))
            return trunc(Int, K + mod((k-1) * J + j, J * K))
        end
        # print(typeof(K + (k-1) * J + j))
        return trunc(Int, K + (k-1) * J + j)
    end
end

# mean of Y at k level
function Y_mean(u, K, J, k)
    i_1 = K + (k-1) * J + 1
    # println(i_1)
    i_2 = K + (k) * J
    # println(i_2)
    return mean(u[i_1:i_2])
end

# mean of Y^2 at k level
function Y_squared_mean(u, K, J, k)
    i_1 = K + (k-1) * J + 1
    # println(i_1)
    i_2 = K + (k) * J
    # println(i_2)
    return mean(u[i_1:i_2].^2)
end


function L96_two!(du, u, p, t)
    K, J, h, F, c, b = p

    K = trunc(Int, K)
    J = trunc(Int, J)

    # X values
    l = 1
    for k in 1:K
        du[k] = -u[L96_index(K, J,k-1,0,l)] * (u[L96_index(K, J,k-2,0,l)] - u[L96_index(K, J,k+1,0,l)]) - u[L96_index(K, J,k,0,l)] + F - h * c * Y_mean(u, K, J, k)
    end

    # Y values
    l = 2
    for k in 1:K
        for j in 1:J
            du[L96_index(K, J,k, j, l)] = c * (-b * u[L96_index(K, J,k,j+1,l)] * (u[L96_index(K, J,k,j+2,l)] - u[L96_index(K, J,k,j-1,l)]) - u[L96_index(K, J,k,j,l)] + (h/J) * u[k])
        end
    end
end

# A reduced L96 model describing just 1 level with a 4th degree polynomial 
# parameterization in place of the Y variables. Based off Wilks 2005, without stochastic term.
# parameterization term = b0 + b_1X_k + b_2X_k^2 + b_3X_k^3 + b_4X_k^4

#TODO Update transform func
mutable struct L96_one_params
    K
    u0
    dt
    timestep
    t_end
    toss
    noise
    noise_ranges
    # Unperturbed ground truth to use as initial conditions
    initial_conditions
    uki_thetas
    transform_func_to_L96_one

    function L96_one_params(;K=36, X_init=nothing,
        dt=0.001, timestep=0.1, t_end=500, toss=0, noise=0.0, noise_ranges=nothing, initial_conditions=nothing, uki_thetas=nothing, transform_func_to_L96_one = L96_trans_C2_to_L96)

        if X_init === nothing
            d = Normal(0, 1)
            X_init = rand(d, K)
        end

        u0 = vcat(X_init)

        return new(K, u0, dt, timestep, t_end, toss, noise, noise_ranges, initial_conditions, uki_thetas, transform_func_to_L96_one)
    end

end

# Parameterization for one level Lorenz 96 model from Wilks,
function wilks_param_L96_one(x, p)
    K, b0, b1, b2, b3, b4 = p
    g = b0 + b1 * x + b2 * x^2 + b3 * x^3 + b4 * x^4
    return g
end

function L96_one!(du, u, p, t)
    K, b0, b1, b2, b3, b4 = p

    K = trunc(Int, K)

    # X values
    l = 1
    J = 0

    for k in 1:K
        # g = parameterizaion
        g = wilks_param_L96_one(u[L96_index(K, J,k,0,l)], p)
        du[k] = -u[L96_index(K, J,k-1,0,l)] * (u[L96_index(K, J,k-2,0,l)] - u[L96_index(K, J,k+1,0,l)]) - u[L96_index(K, J,k,0,l)] + g
        
    end

end

function Sigma_eta_func(y, means, divs)
    #splits the obervations into divs for the purpose of covariance calculations
    obs = size(y, 2)
    div_size = round(Int, obs / divs)
    obs_box = zeros(size(means, 1), divs)
    
    for i in 1:(divs-1)
        obs_box[:, i] = mean(y[:, (i - 1)*div_size + 1:i*div_size], dims=2)
    end
    obs_box[:, divs] = mean(y[:, (divs - 1)*div_size + 1:end], dims=2)

    # display(obs_box)
    # display(means)
    
    sigma_eta = zeros(size(means, 1), size(means, 1))
    
    for i in 1:divs
        sigma_eta += (obs_box[:,i] - means) * (obs_box[:,i] - means)'
        # display((obs_box[:,i] - means)')
    end
    sigma_eta ./= (divs - 1)
    
    return sigma_eta
end

# Apply noise as a factor of the range of the data
# y as a matrix dim_out x obs
function apply_range_noise(y_matr, noise_mag; noise_ranges=nothing, debug=false)
    noisy_y = zeros(size(y_matr))

    if noise_ranges === nothing
        mins = minimum(y_matr, dims=2)
        maxes = maximum(y_matr, dims=2)

        ranges = maxes - mins
    else
        ranges = noise_ranges
    end

    if debug
        # println("Mins and Maxes")
        # display(mins)
        # display(maxes)
        println("Ranges")
        display(ranges)
    end

    mu = zeros(size(y_matr, 1))

    # noise_distribution = MvNormal([0, 0, 0], Diagonal(vec(ranges)))
    noise_distribution = MvNormal(mu, Diagonal(vec(ranges) .* noise_mag))

    for i in 1:size(y_matr, 2)
        # noise_sample = rand(noise_distribution) * noise_mag
        noise_sample = rand(noise_distribution)

        noisy_y[:, i] = y_matr[:, i] .+ noise_sample
    end

    return noisy_y, ranges
    
end

# Apply noise as a factor of the data
# y_obs = y + eps dot xi, eps = noise_mag * y, xi ~ N(0, I)
# y as a matrix dim_out x obs
function apply_noise(y_matr, noise_mag; noise_ranges=nothing, debug=false)
    # placeholder for range based noise
    ranges = nothing

    noisy_y = zeros(size(y_matr))

    if debug
        println("Direct Noise, not range based")
    end

    mu = zeros(size(y_matr, 1))
    identity_matrix = Diagonal(ones(Float64, size(y_matr, 1)))

    # noise_distribution = MvNormal([0, 0, 0], Diagonal(vec(ranges)))
    noise_distribution = MvNormal(mu, identity_matrix)

    for i in 1:size(y_matr, 2)
        # noise_sample = rand(noise_distribution) * noise_mag
        noise_sample = rand(noise_distribution)

        eps = noise_mag * y_matr[:, i]
        noisy_y[:, i] = y_matr[:, i] .+ (eps .* noise_sample)
    end

    return noisy_y, ranges
    
end

# Square_moments for any dimension size
# Define noise as factor of range of data
function Square_moments(y, noise_mag; y_matr=nothing, debug=false)
    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    # display(dim_out)
    # display(obs)
    expanded_y = zeros(dim_out*2, obs)

    if noise_mag > 0.0
        noisy_y = apply_noise(y_matr, noise_mag, debug=debug)
    else
        noisy_y = y_matr
    end
    
    for i in 1:obs

        for j in 1:dim_out
            expanded_y[j, i] = noisy_y[j, i]
            expanded_y[j + dim_out, i] = expanded_y[j, i]^2
        end
    end

    return expanded_y
end

# Mean Square_moments for any dimension size
# Define noise as factor of range of data
function Square_moments_mean(y, noise_mag; y_matr=nothing)
    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    # display(dim_out)
    # display(obs)
    expanded_y = zeros(dim_out*2, obs)

    if noise_mag > 0.0
        noisy_y = apply_noise(y_matr, noise_mag)
    else
        noisy_y = y_matr
    end
    
    for i in 1:obs

        for j in 1:dim_out
            expanded_y[j, i] = noisy_y[j, i]
            expanded_y[j + dim_out, i] = expanded_y[j, i]^2
        end
    end

    mean_y = mean(expanded_y, dims=2)

    # display(mean_y)

    return mean_y
end

function Cube_moments(y, noise)
    expanded_y = zeros(size(y[1], 1)*2, size(y,1))
    
    for i in 1:size(y,1)
        noise_sample = rand(Normal(), 3) * noise
        expanded_y[1, i] = y[i][1] + noise_sample[1]
        expanded_y[2, i] = y[i][2] + noise_sample[2]
        expanded_y[3, i] = y[i][3] + noise_sample[3]
        expanded_y[4, i] = expanded_y[1, i]^3
        expanded_y[5, i] = expanded_y[2, i]^3
        expanded_y[6, i] = expanded_y[3, i]^3
    end
    return expanded_y
end

# no moments for any dimension size
function No_moments(y, noise)
    dim_out = size(y[1], 1)
    obs = size(y,1)
    expanded_y = zeros(dim_out, obs)
    
    for i in 1:obs
        noise_sample = rand(Normal(), dim_out) * noise

        for j in 1:dim_out
            expanded_y[j, i] = y[i][j] + noise_sample[j]
        end
    end

    return expanded_y
end

function Cross_moments(y, noise)
    expanded_y = zeros(size(y[1], 1)*2, size(y,1))
    
    for i in 1:size(y,1)
        noise_sample = rand(Normal(), 3) * noise
        expanded_y[1, i] = y[i][1] + noise_sample[1]
        expanded_y[2, i] = y[i][2] + noise_sample[2]
        expanded_y[3, i] = y[i][3] + noise_sample[3]
        expanded_y[4, i] = expanded_y[1, i] * expanded_y[2, i]
        expanded_y[5, i] = expanded_y[2, i] * expanded_y[3, i]
        expanded_y[6, i] = expanded_y[3, i] * expanded_y[1, i]
    end
    return expanded_y
end

function Velocity_moments(y, noise)
    expanded_y = zeros(size(y[1], 1)*2, size(y,1))

    for i in 1:size(y,1)-1
        noise_sample = rand(Normal(), 3) * noise
        expanded_y[1, i] = y[i][1] + noise_sample[1]
        expanded_y[2, i] = y[i][2] + noise_sample[2]
        expanded_y[3, i] = y[i][3] + noise_sample[3]
        expanded_y[4, i] = abs(expanded_y[1, i+1] - expanded_y[1, i])
        expanded_y[5, i] = abs(expanded_y[2, i+1] - expanded_y[2, i])
        expanded_y[6, i] = abs(expanded_y[3, i+1] - expanded_y[3, i])
    end
    i = size(y,1)
    noise_sample = rand(Normal(), 3) * noise
    expanded_y[1, i] = y[i][1] + noise_sample[1]
    expanded_y[2, i] = y[i][2] + noise_sample[2]
    expanded_y[3, i] = y[i][3] + noise_sample[3]
    expanded_y[4, i] = expanded_y[4, i-1]
    expanded_y[5, i] = expanded_y[5, i-1]
    expanded_y[6, i] = expanded_y[6, i-1]

    return expanded_y
end

function Velocity_signed_moments(y, noise)
    expanded_y = zeros(size(y[1], 1)*2, size(y,1))

    for i in 1:size(y,1)-1
        noise_sample = rand(Normal(), 3) * noise
        expanded_y[1, i] = y[i][1] + noise_sample[1]
        expanded_y[2, i] = y[i][2] + noise_sample[2]
        expanded_y[3, i] = y[i][3] + noise_sample[3]
        expanded_y[4, i] = expanded_y[1, i+1] - expanded_y[1, i]
        expanded_y[5, i] = expanded_y[2, i+1] - expanded_y[2, i]
        expanded_y[6, i] = expanded_y[3, i+1] - expanded_y[3, i]
    end
    i = size(y,1)
    noise_sample = rand(Normal(), 3) * noise
    expanded_y[1, i] = y[i][1] + noise_sample[1]
    expanded_y[2, i] = y[i][2] + noise_sample[2]
    expanded_y[3, i] = y[i][3] + noise_sample[3]
    expanded_y[4, i] = expanded_y[4, i-1]
    expanded_y[5, i] = expanded_y[5, i-1]
    expanded_y[6, i] = expanded_y[6, i-1]

    return expanded_y
end

function Velocity_squared_moments(y, noise)
    expanded_y = zeros(size(y[1], 1)*2, size(y,1))

    for i in 1:size(y,1)-1
        noise_sample = rand(Normal(), 3) * noise
        expanded_y[1, i] = y[i][1] + noise_sample[1]
        expanded_y[2, i] = y[i][2] + noise_sample[2]
        expanded_y[3, i] = y[i][3] + noise_sample[3]
        expanded_y[4, i] = (expanded_y[1, i+1] - expanded_y[1, i])^2
        expanded_y[5, i] = (expanded_y[2, i+1] - expanded_y[2, i])^2
        # expanded_y[6, i] = (expanded_y[3, i+1] - expanded_y[3, i])^2
    end
    i = size(y,1)
    noise_sample = rand(Normal(), 3) * noise
    expanded_y[1, i] = y[i][1] + noise_sample[1]
    expanded_y[2, i] = y[i][2] + noise_sample[2]
    expanded_y[3, i] = y[i][3] + noise_sample[3]
    expanded_y[4, i] = expanded_y[4, i-1]
    expanded_y[5, i] = expanded_y[5, i-1]
    expanded_y[6, i] = expanded_y[6, i-1]

    return expanded_y
end

# Observation function A, see research note
# (1/K)Sum(X_k, y_mean_k, x_k^2, X_k*y_mean_k, y^2_mean_k)
function L96_obs_func_a(y, noise_mag, params; y_matr=nothing, noise_ranges=nothing, debug=false)
    K = params.K
    J = params.J

    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    # display(dim_out)
    # display(obs)

    expanded_y = zeros(5, obs)

    if noise_mag > 0.0
        if noise_ranges === nothing
            noisy_y, noise_ranges = apply_noise(y_matr, noise_mag, debug=debug)
        else
            noisy_y, _ = apply_noise(y_matr, noise_mag, noise_ranges=noise_ranges, debug=debug)
        end

    else
        noisy_y = y_matr
    end

    if debug
        println("Noisy Y complete")
    end

    for i in 1:size(noisy_y,2)
        z = noisy_y[:,i]
        temp_sum = zeros(5)
        for i in 1:K
            temp_sum += [z[i], Y_mean(z, K, J, i), z[i]^2, z[i] * Y_mean(z,K, J, i), Y_squared_mean(z,K, J, i)]
        end

        phi = temp_sum / K

        expanded_y[:, i] = phi
    end

    if debug
        println("Expanded Y complete")
    end

    return expanded_y, noise_ranges
    
end

# Mean Observation function A, see research note
# (1/K)Sum(X_k, y_mean_k, x_k^2, X_k*y_mean_k, y^2_mean_k)
function L96_obs_func_a_mean(y, noise_mag, params; y_matr=nothing, noise_ranges=nothing, debug=false)
    K = params.K
    J = params.J

    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    # display(dim_out)
    # display(obs)
    expanded_y = zeros(5, obs)

    if noise_mag > 0.0
        if noise_ranges === nothing
            noisy_y, noise_ranges = apply_noise(y_matr, noise_mag, debug=debug)
        else
            noisy_y, _ = apply_noise(y_matr, noise_mag, noise_ranges=noise_ranges, debug=debug)
        end

    else
        noisy_y = y_matr
    end

    for i in 1:size(noisy_y,2)
        z = noisy_y[:,i]
        temp_sum = zeros(5)
        for i in 1:K
            temp_sum += [z[i], Y_mean(z, K, J, i), z[i]^2, z[i] * Y_mean(z,K, J, i), Y_squared_mean(z,K, J, i)]
        end

        phi = temp_sum / K

        expanded_y[:, i] = phi
    end

    mean_y = mean(expanded_y, dims=2)

    return mean_y, noise_ranges
    
end

# A moments function for L96 that returns full output
# Used to get ground truth before noise and moments
function L96_no_moments(y, noise_mag, params; y_matr=nothing,noise_ranges=nothing, debug=false)
    K = params.K
    J = params.J

    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    if noise_mag > 0.0
        if noise_ranges === nothing
            noisy_y, noise_ranges = apply_noise(y_matr, noise_mag, debug=debug)
        else
            noisy_y, _ = apply_noise(y_matr, noise_mag, noise_ranges=noise_ranges, debug=debug)
        end

    else
        noisy_y = y_matr
    end

    # println("Shape of noisy y in L96_no_moments")
    # display(shape(noisy_y))

    return noisy_y
    
end

# A moments function for L96 for use with one level L96
# Includes first and second moments and cross moments of first n X variables (4 by default to match Huang '22)
function L96_one_moments(y, noise_mag, params; y_matr=nothing, n = 4, noise_ranges=nothing, debug=false)
    K = params.K

    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    expanded_y = zeros(n + n + (n*(n-1))÷2, obs)

    if noise_mag > 0.0
        if noise_ranges === nothing
            noisy_y, noise_ranges = apply_noise(y_matr, noise_mag, debug=debug)
        else
            noisy_y, _ = apply_noise(y_matr, noise_mag, noise_ranges=noise_ranges, debug=debug)
        end

    else
        noisy_y = y_matr
    end

    for i in 1:size(noisy_y,2)
        z = noisy_y[:,i]
        # First n moments
        for j in 1:n
            expanded_y[j, i] = z[j]
            expanded_y[j + n, i] = z[j]^2
        end

        # Cross moments
        idx = 2*n + 1
        for j in 1:(n-1)
            for k in (j+1):n
                expanded_y[idx, i] = z[j] * z[k]
                idx += 1
            end
        end
    end

    return expanded_y
    
end

# Mean A moments function for L96 for use with one level L96
L96_one_moments_mean(y, noise_mag, params; y_matr=nothing, n = 4, noise_ranges=nothing, debug=false) = 
    mean(L96_one_moments(y, noise_mag, params; y_matr=y_matr, n=n, noise_ranges=noise_ranges, debug=debug), dims=2)

# A  moments function for L96 for use with one level L96 that returns full output
# Used to get ground truth before noise and moments
function L96_one_no_moments(y, noise_mag, params; y_matr=nothing, noise_ranges=nothing, debug=false)
    K = params.K

    if y_matr === nothing
        dim_out = size(y[1], 1)
        obs = size(y,1)

        y_matr = hcat(y...)
        # display(y_matr)
    else
        dim_out = size(y_matr, 1)
        obs = size(y_matr, 2)
    end

    if noise_mag > 0.0
        if noise_ranges === nothing
            noisy_y, noise_ranges = apply_noise(y_matr, noise_mag, debug=debug)
        else
            noisy_y, _ = apply_noise(y_matr, noise_mag, noise_ranges=noise_ranges, debug=debug)
        end

    else
        noisy_y = y_matr
    end

    # println("Shape of noisy y in L96_one_no_moments")
    # display(shape(noisy_y))

    return noisy_y
    
end


function theta_lorenz_to_uki(theta)
    return theta
end

function theta_uki_to_theta(theta)
    return abs.(theta)
end

function theta_L96_to_uki(theta)
    return (20 .- theta) ./ theta
end

function theta_uki_to_L96(theta)
    # transform to (0, 20)
    return 20 ./ (1 .+ abs.(theta))
end

# Transform A
function L96_trans_A_to_UKI(theta)
    theta_copy = copy(theta)
    theta_copy[3] = exp(theta_copy[3])
    return theta_copy
end

# Transform A
function L96_trans_A_to_L96(theta)
    theta_copy = copy(theta)
    theta_copy[3] = log(theta_copy[3])
    return theta_copy
end

# Transform B
function L96_trans_B_to_UKI(theta)
    return (20 .- theta) ./ theta
end
# Transform B
function L96_trans_B_to_L96(theta)
    # transform to (0, 20)
    return 20 ./ (1 .+ abs.(theta))
end

# Transform C
function L96_trans_C_to_L96(theta)
    theta_copy = abs.(theta)
    return theta_copy
end

# Transform C
function L96_trans_C_to_UKI(theta)
    theta_copy = abs.(theta)
    return theta_copy
end

# Transform C_2
function L96_trans_C2_to_L96(theta)
    theta_copy = copy(theta)
    theta_copy[3] = abs(theta_copy[3])
    return theta_copy
end

# Transform C_2
function L96_trans_C2_to_UKI(theta)
    theta_copy = copy(theta)
    return theta_copy
end

# Transform D
function L96_trans_D_to_L96(theta)
    theta_copy = copy(theta)
    return theta_copy
end

# Transform D
function L96_trans_D_to_UKI(theta)
    theta_copy = copy(theta)
    return theta_copy
end

# Transform E
# Hold h=1, F=10
function L96_trans_E_to_L96(theta)
    theta_copy = copy(theta)
    theta_copy[1] = 1.0
    theta_copy[2] = 10.0
    return theta_copy
end

# Transform E
function L96_trans_E_to_UKI(theta)
    theta_copy = copy(theta)
    return theta_copy
end

# Transform F
# Hold b=10
function L96_trans_F_to_L96(theta)
    theta_copy = copy(theta)
    theta_copy[4] = 10.0
    return theta_copy
end

# Transform F
function L96_trans_F_to_UKI(theta)
    theta_copy = copy(theta)
    return theta_copy
end



# There is a change of variables so theta is always positive
function lorenz_model(theta, moments_func, transform_func, param_file::lorenz_params; abstol=1e-6, reltol=1e-3)
    tspan = (0.0,param_file.t_end)
    # print(tspan)
    prob1 = ODEProblem(lorenz!, param_file.u0, tspan, transform_func(theta))
    sol = solve(prob1, saveat = param_file.timestep, abstol=abstol, reltol=reltol)
    tossed = sol.u[round(Int, param_file.toss/param_file.timestep)+1:end]

    # print(tossed[:, end])
    
    expanded_tossed = moments_func(tossed, param_file.noise)
    
    means = mean(expanded_tossed, dims=2)

    return means, expanded_tossed
end

function lorenz_model_short(theta, moments_func, transform_func, param_file::lorenz_params; abstol=1e-6, reltol=1e-3)
    tspan = (0.0,param_file.t_end)
    prob1 = ODEProblem(lorenz!, param_file.u0, tspan, transform_func(theta))
    sol = solve(prob1, saveat = param_file.timestep, abstol=abstol, reltol=reltol)
    tossed = sol.u

    expanded_tossed = moments_func(tossed, param_file.noise)

    return expanded_tossed
end

# Wrapped L63 model to match form for UKI
function wrapped_lorenz_model(theta, moments_func, transform_func, param_file::lorenz_params)
    y, _ = lorenz_model(theta, moments_func, transform_func, param_file)
    return y
end

# Wrapped short L63 model to match form for UKI
function wrapped_short_lorenz_model(theta, moments_func, transform_func, param_file::lorenz_params)
    y = lorenz_model_short(theta, moments_func, transform_func, param_file)
    return y
end

# Need to match this structure
# uki_params.model_func_part_1(thetas, sample_idxs, uki_params.moments_func, uki_params.transform_func, uki_params.model_params)

function batch_lorenz_63_model_part_1!(thetas, sample_idxs, moments_func, transform_func, model_params::lorenz_params)
    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    transformed_thetas = zeros((N_theta, N_sigma))

    for sigma in 1:N_sigma
        transformed_thetas[:, sigma] = transform_func(thetas[:, sigma])
    end

    model_params.uki_thetas = transformed_thetas

    # display(model_params.uki_thetas)

end

# Need to match this structure
# uki_params.model_func_part_2(uki_params.sample_idxs, uki_params.moments_func, uki_params.N_theta, uki_params.N_out, uki_params.model_params)
# returning:
# x_s, fail_bool
# Where:
# x_s shape [sigma_count, N_out, batch_size]

# Wrapped short L63 model to match form for UKI part 2
# This is where model runs actually take place
function batch_lorenz_63_model_part_2(sample_idxs, moments_func, N_theta, N_out, model_params::lorenz_params)
    # Helper sizes
    N_samples = size(sample_idxs, 1)
    N_sigma = N_theta * 2 + 1

    # array to store outputs
    x_s = zeros(N_sigma, N_out, N_samples)

    # theta values
    thetas = model_params.uki_thetas

    # Solver parameters
    abstol=1e-6
    reltol=1e-3

    for sigma in 1:N_sigma
        for j in 1:N_samples
            sample_idx = sample_idxs[j]
            u_0 = model_params.initial_conditions[1:3, sample_idx - 1]
            # display(u_0)
            tspan = (0.0, model_params.t_end)
            prob1 = ODEProblem(lorenz!, u_0, tspan, thetas[:, sigma])
            sol = solve(prob1, saveat = model_params.timestep, abstol=abstol, reltol=reltol)
            # output = [last(sol.u)]
            output = sol.u

            # display(output)
            # display(sol.t)
            moments_output = moments_func(output, model_params.noise)[:, end]
            # display(moments_output)
            x_s[sigma, :, j] = moments_output
        end
    end

    return x_s, false
end

function drop_Y(output, K)
    obs = size(output, 1)
    for i in 1:obs
        output[i] = output[i][1:K]
    end
    
    return output

end

function L96_model(theta, moments_func, transform_func, param_file::L96_two_params; abstol=1e-6, reltol=1e-3)
    tspan = (0.0,param_file.t_end)

    # take absolute value of only parameter parts to avoid indexing float
    theta[3:6] = transform_func(theta[3:6])

    println(theta)

    prob1 = ODEProblem(L96_two!, param_file.u0, tspan, theta)
    # sol = solve(prob1, RK4(), dtmin=0.0001, saveat = param_file.timestep)
    sol = solve(prob1, saveat = param_file.timestep, abstol=abstol, reltol=reltol)
    tossed = sol.u[round(Int, param_file.toss/param_file.timestep)+1:end]

    # tossed = drop_Y(tossed, param_file.K)
    
    expanded_tossed = moments_func(tossed, param_file)
    
    means = mean(expanded_tossed, dims=2)

    return means, expanded_tossed
end

function L96_model_short(theta, moments_func, transform_func, param_file::L96_two_params; abstol=1e-9, reltol=1e-6, u0=nothing)
    tspan = (0.0,param_file.t_end)

    # take absolute value of only parameter parts to avoid indexing float
    theta[3:6] = transform_func(theta[3:6])

    if u0 == nothing
        prob1 = ODEProblem(L96_two!, param_file.u0, tspan, theta)
    else
        prob1 = ODEProblem(L96_two!, u0, tspan, theta)
    end

    sol = solve(prob1, saveat = param_file.timestep, abstol=abstol, reltol=reltol)
    tossed = sol.u
    
    expanded_tossed = moments_func(tossed, param_file)

    return expanded_tossed[:, end]
end

function wrapped_lorenz_96_model(theta, moments_func, transform_func, param_file::L96_two_params)
    theta = vcat([param_file.K, param_file.J], theta)

    y, _ = L96_model(theta, moments_func, transform_func, param_file)
    return y
end

function wrapped_short_lorenz_96_model(theta, moments_func, transform_func, param_file::L96_two_params; abstol=1e-9, reltol=1e-6, u0=nothing)
    theta = vcat([param_file.K, param_file.J], theta)

    y= L96_model_short(theta, moments_func, transform_func, param_file, abstol=abstol, reltol=reltol, u0=u0)
    return y
end

function L96_model_testing(theta, moments_func, transform_func, param_file::L96_two_params)
    tspan = (0.0,param_file.t_end)

    # take absolute value of only parameter parts to avoid indexing float
    theta[3:6] = transform_func(theta[3:6])

    # println(theta)

    prob1 = ODEProblem(L96_two!, param_file.u0, tspan, theta)
    sol = solve(prob1, saveat = param_file.timestep)
    tossed = sol.u

    # tossed = drop_Y(tossed, param_file.K)
    
    expanded_tossed = moments_func(tossed, param_file)
    
    # means = mean(expanded_tossed, dims=2)

    return expanded_tossed[:, end], tossed
end

function L96_model_full_output(theta, moments_func, transform_func, param_file::L96_two_params; abstol=1e-6, reltol=1e-3)
    tspan = (0.0,param_file.t_end)

    # take absolute value of only parameter parts to avoid indexing float
    theta[3:6] = transform_func(theta[3:6])

    # println(theta)

    prob1 = ODEProblem(L96_two!, param_file.u0, tspan, theta)

    sol = solve(prob1, AutoTsit5(Rosenbrock23()), saveat = param_file.timestep, abstol=abstol, reltol=reltol, maxiters=1e7)
    tossed = sol.u[round(Int, param_file.toss/param_file.timestep)+1:end]

    # display(size(sol.u))
    # display(tossed)
    # println(round(Int, param_file.toss/param_file.timestep))
    
    y = moments_func(tossed, param_file.noise, param_file)

    # println("Shape of y in L96_model_full_output")
    # display(shape(y))

    return y
end

# A version of L96 model full output for L96 one
function L96_one_model_full_output(theta, moments_func, transform_func, param_file::L96_one_params; abstol=1e-6, reltol=1e-3)
    tspan = (0.0,param_file.t_end)

    # take absolute value of only parameter parts to avoid indexing float
    theta[2:6] = transform_func(theta[2:6])

    # println(theta)

    prob1 = ODEProblem(L96_one!, param_file.u0, tspan, theta)

    sol = solve(prob1, AutoTsit5(Rosenbrock23()), saveat = param_file.timestep, abstol=abstol, reltol=reltol, maxiters=1e7)
    tossed = sol.u[round(Int, param_file.toss/param_file.timestep)+1:end]

    # display(size(sol.u))
    # display(tossed)
    # println(round(Int, param_file.toss/param_file.timestep))
    
    y = moments_func(tossed, param_file)

    # println("Shape of y in L96_one_model_full_output")
    # display(shape(y))

    return y
end

function gen_noise_mat(full_output, constant)
    full_mat = hcat(full_output...)'
    noise_var = diagm(sqrt.((mean(full_mat.^2, dims=1))*constant)[1, :])
    return noise_var
end

function batch_lorenz_96_model_part_1!(thetas, sample_idxs, moments_func, transform_func, model_params::L96_two_params)
    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    transformed_thetas = zeros((N_theta, N_sigma))

    # for sigma in 1:N_sigma
    #     transformed_thetas[:, sigma] = transform_func(thetas[:, sigma])
    # end

    # model_params.uki_thetas = transformed_thetas

    model_params.uki_thetas = thetas
    # display(model_params.uki_thetas)
end

function batch_lorenz_96_model_part_2(sample_idxs, moments_func, N_theta, N_out, model_params::L96_two_params)
    # println("Threads: $(Threads.nthreads())")
    # Helper sizes
    N_samples = size(sample_idxs, 1)
    N_sigma = N_theta * 2 + 1

    # println("N_theta: $(N_theta)")

    # array to store outputs
    x_s = zeros(N_sigma, N_out, N_samples)

    # theta values
    thetas = model_params.uki_thetas

    # Solver parameters
    abstol=1e-6
    reltol=1e-3

    # Thread-safe failure flag and lock
    failure_flag = false
    failure_lock = ReentrantLock()

    # List of return codes that indicate failure
    failure_codes = [:MaxIters, :DtLessThanMin, :Unstable, :InitialFailure, :ConvergenceFailure, :Failure]

    Threads.@threads for sigma in 1:N_sigma
        # Check if we should continue
        lock(failure_lock) do
            if failure_flag
                return
            end
        end

        for j in 1:N_samples
            # Check if we should continue
            lock(failure_lock) do
                if failure_flag
                    return
                end
            end

            sample_idx = sample_idxs[j]
            u_0 = model_params.initial_conditions[:, sample_idx - 1]
            tspan = (0.0, model_params.t_end)

            # Need to use transform function to get correct theta
            transformed_theta = model_params.transform_func_to_L96(thetas[:, sigma])

            KJ_theta = vcat([model_params.K, model_params.J], transformed_theta)
            prob1 = ODEProblem(L96_two!, u_0, tspan, KJ_theta)
            sol = solve(prob1, saveat = model_params.timestep, abstol=abstol, reltol=reltol, maxiters=1e5)

            # display(sol.alg)
            
            # Convert retcode to symbol for comparison
            retcode_symbol = Symbol(string(sol.retcode))
            # println("Converted retcode to symbol: $(retcode_symbol)")
            
            # Check if the solver failed
            if retcode_symbol in failure_codes
                println("ODE solver failed with retcode: $(sol.retcode)")
                lock(failure_lock) do
                    failure_flag = true
                end
                break
            end
            
            output = sol.u[round(Int, model_params.toss/model_params.timestep)+1:end]
            sol = nothing

            moments_output, _ = moments_func(output, model_params.noise, model_params, noise_ranges=model_params.noise_ranges)
            moments_output = moments_output[:, end]

            output = nothing

            x_s[sigma, :, j] = moments_output
        end
    end

    return x_s, failure_flag
end

function batch_lorenz_96_one_model_part_1!(thetas, sample_idxs, moments_func, transform_func, model_params::L96_one_params)
    N_sigma = size(thetas, 2)
    N_theta = size(thetas, 1)
    transformed_thetas = zeros((N_theta, N_sigma))

    # for sigma in 1:N_sigma
    #     transformed_thetas[:, sigma] = transform_func(thetas[:, sigma])
    # end

    # model_params.uki_thetas = transformed_thetas

    model_params.uki_thetas = thetas
    # display(model_params.uki_thetas)
end

function batch_lorenz_96_one_model_part_2(sample_idxs, moments_func, N_theta, N_out, model_params::L96_one_params)
    # println("Threads: $(Threads.nthreads())")
    # Helper sizes
    N_samples = size(sample_idxs, 1)
    N_sigma = N_theta * 2 + 1

    # println("N_theta: $(N_theta)")

    # array to store outputs
    x_s = zeros(N_sigma, N_out, N_samples)

    # theta values
    thetas = model_params.uki_thetas

    # Solver parameters
    abstol=1e-6
    reltol=1e-3

    # Thread-safe failure flag and lock
    failure_flag = false
    failure_lock = ReentrantLock()

    # List of return codes that indicate failure
    failure_codes = [:MaxIters, :DtLessThanMin, :Unstable, :InitialFailure, :ConvergenceFailure, :Failure]

    Threads.@threads for sigma in 1:N_sigma
        # Check if we should continue
        lock(failure_lock) do
            if failure_flag
                return
            end
        end

        for j in 1:N_samples
            # Check if we should continue
            lock(failure_lock) do
                if failure_flag
                    return
                end
            end

            sample_idx = sample_idxs[j]
            u_0 = model_params.initial_conditions[:, sample_idx - 1]
            tspan = (0.0, model_params.t_end)

            # Need to use transform function to get correct theta
            transformed_theta = model_params.transform_func_to_L96(thetas[:, sigma])

            K_theta = vcat([model_params.K], transformed_theta)
            prob1 = ODEProblem(L96_one!, u_0, tspan, K_theta)
            sol = solve(prob1, saveat = model_params.timestep, abstol=abstol, reltol=reltol, maxiters=1e5)

            # display(sol.alg)
            
            # Convert retcode to symbol for comparison
            retcode_symbol = Symbol(string(sol.retcode))
            # println("Converted retcode to symbol: $(retcode_symbol)")
            
            # Check if the solver failed
            if retcode_symbol in failure_codes
                println("ODE solver failed with retcode: $(sol.retcode)")
                lock(failure_lock) do
                    failure_flag = true
                end
                break
            end
            
            output = sol.u[round(Int, model_params.toss/model_params.timestep)+1:end]
            sol = nothing

            moments_output, _ = moments_func(output, model_params.noise, model_params, noise_ranges=model_params.noise_ranges)
            moments_output = moments_output[:, end]

            output = nothing

            x_s[sigma, :, j] = moments_output
        end
    end

    return x_s, failure_flag
end