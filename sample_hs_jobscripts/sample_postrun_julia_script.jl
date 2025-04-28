include("../../UKI.jl")
include("../../held_suarez_uki_funcs.jl")
using FileIO

case_prefix = "scenario_name"

proj_case_dir = "/glade/work/teopb/$(case_prefix)_hs_uki_cases"

uki_filename = "uki_params_$(case_prefix).jld2"

uki_params = read_UKI_params("$(proj_case_dir)/$(uki_filename)")

if uki_iteration_part2!(uki_params, debug=false)

    println("UKI_part_2 complete! current_iter=$(uki_params.current_iter)")

    uki_params = read_UKI_params("$(proj_case_dir)/$(uki_filename)")

    if uki_params.current_iter > uki_params.max_iter
        println("Iterations Complete!")

        # write a file to indicate that the iterations are complete
        open("$(proj_case_dir)/$(case_prefix)_complete.txt", "w") do f
            write(f, "Iterations Complete!")
        end
    else
        uki_iteration_part1!(uki_params, debug=false)
        println("UKI_part_1 complete! current_iter=$(uki_params.current_iter)")
    end

else

    println("UKI_part_2 failed! current_iter=$(uki_params.current_iter)")

end