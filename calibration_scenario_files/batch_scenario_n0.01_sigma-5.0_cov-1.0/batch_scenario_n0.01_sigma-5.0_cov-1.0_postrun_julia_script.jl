include("../../UKI.jl")
include("../../held_suarez_uki_funcs.jl")
using FileIO

case_prefix = "batch_scenario_n0.01_sigma-5.0_cov-1.0"

proj_case_dir = "/glade/work/teopb/$(case_prefix)_hs_uki_cases"

uki_filename = "uki_params_$(case_prefix).jld2"

uki_params = read_UKI_params("$(proj_case_dir)/$(uki_filename)")

if uki_iteration_part2!(uki_params)

    println("UKI_part_2 complete! current_iter=$(uki_params.current_iter)")

    uki_params = read_UKI_params("$(proj_case_dir)/$(uki_filename)")

    if uki_params.current_iter > uki_params.max_iter
        println("Iterations Complete!")
    else
        uki_iteration_part1!(uki_params)
        println("UKI_part_1 complete! current_iter=$(uki_params.current_iter)")
    end

else

    println("UKI_part_2 failed! current_iter=$(uki_params.current_iter)")

end
