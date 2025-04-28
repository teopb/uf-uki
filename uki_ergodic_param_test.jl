include("Lorenz.jl")
include("UKI.jl")
include("uki_v4_paper_functions.jl")
using CSV, DataFrames

println("Threads: $(Threads.nthreads())")

# Read in optional command line parameter "reverse", a boolean
reverse_order = length(ARGS) > 0 ? parse(Bool, ARGS[1]) : false

if reverse_order
    println("Running in reverse order")
else
    println("Running in forward order")
end

# Read in candidate parameters
data_save_dir = "l96_spring_paper_saved/saved_data/ergodic_tests"
filename = joinpath(data_save_dir, "uki_ergodic_candidate_params.jld2")
candidate_params = load(filename)["data"]

uki_iters = 150
timestep = .1

filename = joinpath(data_save_dir, "uki_ergodic_result_repeats.jld2")
filename_reverse = joinpath(data_save_dir, "uki_ergodic_result_repeats_reverse.jld2")
# Check if the file exists
if reverse_order && isfile(filename_reverse)
    println("Loading existing data from $(filename_reverse)")
    best_results_repeats_df = load(filename_reverse)["data"]
elseif isfile(filename)
    # Load the existing data
    println("Loading existing data from $(filename)")
    best_results_repeats_df = load(filename)["data"]
else
    # Create a new DataFrame if the file doesn't exist
    best_results_repeats_df = DataFrame(t_end = Float64[], sigma_eta_mag = Float64[], noise_mag = Float64[], C_0_mag = Float64[], threshold_01 = Float64[], threshold_015 = Float64[], threshold_03 = Float64[], threshold_01_cost = Float64[], threshold_015_cost = Float64[], threshold_03_cost = Float64[], repeat=Int64[])
end

if reverse_order
    filename = filename_reverse
end

repeat_count = 20
noise_mags = [.01, .05, .1]

for row in (reverse_order ? reverse(eachrow(candidate_params)) : eachrow(candidate_params))
    for noise_mag in noise_mags
        t_end = row.t_end
        sigma_eta_mag = row.sigma_eta_mag
        C_0_mag = row.C_0_mag
        toss = Int(t_end/10)

        # Check if the combination of parameters has already been run
        if any(best_results_repeats_df.t_end .== t_end .&& 
            best_results_repeats_df.sigma_eta_mag .== sigma_eta_mag .&& 
               best_results_repeats_df.C_0_mag .== C_0_mag .&& 
               best_results_repeats_df.noise_mag .== noise_mag)
            println("Skipping $(t_end), $(sigma_eta_mag), $(C_0_mag), $(noise_mag) because it has already been run")
        else
            for repeat in 1:repeat_count

                _, stable_iters, costs, _ = ergodic_test(t_end, timestep, sigma_eta_mag, C_0_mag, uki_iters,
                    prior_theta = [0.1, 5, 2, 7], obs_noise = noise_mag, toss=toss, plot_progress=false,show_plot=false);

                df_row = [t_end, sigma_eta_mag, noise_mag, C_0_mag, stable_iters..., costs..., repeat]
                push!(best_results_repeats_df, df_row)
            end
            # Save the results
            save_data(best_results_repeats_df, filename)
        end
    end
end

println("Repeats completed, results saved to $(filename)")
