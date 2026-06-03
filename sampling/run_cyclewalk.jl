import Pkg
Pkg.activate(joinpath(@__DIR__, "runCycleWalkEnv"))
Pkg.instantiate()

using CycleWalk
using JSON
using RandomNumbers

function _usage()
    println("usage: sampling/run_cyclewalk.jl --map-file <graph.json> --output-file <atlas.jsonl.gz> [--pop-dev X] [--gamma X] [--iso-weight X] [--cycle-walk-steps X] [--cycle-walk-out-freq N] [--run-diagnostics true|false] [--rng-seed N]")
end

function _parse_kv_args(args::Vector{String})
    kv = Dict{String, String}()
    if length(args) == 0
        return kv
    end
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--help" || a == "-h"
            kv["help"] = "true"
            i += 1
            continue
        end
        if !startswith(a, "--")
            error("unexpected argument: " * a)
        end
        if i == length(args)
            error("missing value for " * a)
        end
        key = replace(a[3:end], "-" => "_")
        kv[key] = args[i+1]
        i += 2
    end
    return kv
end

function _resolve_path(base_dir::String, p::String)
    if isabspath(p)
        return p
    end
    return abspath(joinpath(base_dir, p))
end

function _get_or(kv::Dict{String, String}, k::String, default::String)
    if haskey(kv, k)
        return kv[k]
    end
    return default
end

function _required(kv::Dict{String, String}, k::String)
    if !haskey(kv, k)
        error("missing required arg: --" * replace(k, "_" => "-"))
    end
    return kv[k]
end

function _parse_bool(s::String)
    t = lowercase(strip(s))
    if t == "true" || t == "1"
        return true
    end
    if t == "false" || t == "0"
        return false
    end
    error("invalid bool: " * s)
end

function main()
    kv = _parse_kv_args(ARGS)
    if haskey(kv, "help")
        _usage()
        exit(0)
    end
    allowed = Set([
        "pop_dev",
        "map_file",
        "output_file",
        "gamma",
        "iso_weight",
        "cycle_walk_steps",
        "cycle_walk_out_freq",
        "run_diagnostics",
        "rng_seed",
    ])
    for k in keys(kv)
        if k in allowed
            continue
        end
        error("unknown arg: --" * replace(k, "_" => "-"))
    end

    repo_root = abspath(joinpath(@__DIR__, ".."))

    pop_dev = parse(Float64, _get_or(kv, "pop_dev", "0.02"))
    map_file = _resolve_path(repo_root, _required(kv, "map_file"))
    num_dists = Int(JSON.parsefile(map_file)["num_districts"])
    output_file_path = _resolve_path(repo_root, _required(kv, "output_file"))
    gamma = parse(Float64, _get_or(kv, "gamma", "0.0"))
    iso_weight = parse(Float64, _get_or(kv, "iso_weight", "0.0"))
    cycle_walk_steps = parse(Float64, _get_or(kv, "cycle_walk_steps", "1e3"))
    cycle_walk_out_freq = parse(Float64, _get_or(kv, "cycle_walk_out_freq", "100"))
    run_diagnostics = _parse_bool(_get_or(kv, "run_diagnostics", "false"))
    rng_seed = parse(UInt64, _get_or(kv, "rng_seed", "4541901234"))

    node_data = Set([
        "county",
        "precinct_id",
        "precinct_id_str",
        "population",
        "area",
        "border_length",
        "x_location",
        "y_location",
    ])
    geo_units = ["precinct_id_str"]
    pop_col = "population"
    area_col = "area"
    node_border_col = "border_length"
    edge_perimeter_col = "length"
    edge_weights = "connections"
    two_cycle_walk_frac = 0.1

    if !endswith(output_file_path, ".jsonl.gz")
        error("output_file must end with .jsonl.gz")
    end

    rng = PCG.PCGStateOneseq(UInt64, rng_seed)

    steps = Int(ceil(cycle_walk_steps / two_cycle_walk_frac))
    outfreq = Int(floor(cycle_walk_out_freq / two_cycle_walk_frac))
    if outfreq < 1
        outfreq = 1
    end

    base_graph = CycleWalk.BaseGraph(map_file, pop_col,
                                     inc_node_data=node_data,
                                     edge_weights=edge_weights,
                                     area_col=area_col,
                                     node_border_col=node_border_col,
                                     edge_perimeter_col=edge_perimeter_col)
    graph = CycleWalk.MultiLevelGraph(base_graph, geo_units)

    constraints = CycleWalk.initialize_constraints()
    CycleWalk.add_constraint!(constraints, CycleWalk.PopulationConstraint(graph, num_dists, pop_dev))
    pc = constraints[CycleWalk.PopulationConstraint]
    if pc.min_pop > pc.max_pop || base_graph.total_pop < num_dists * pc.min_pop || base_graph.total_pop > num_dists * pc.max_pop
        ideal_pop = base_graph.total_pop / num_dists
        floor_ideal = floor(Int, ideal_pop)
        ceil_ideal = ceil(Int, ideal_pop)
        t_required = ceil_ideal == floor_ideal ? 0.0 : max(1.0 - floor_ideal / ideal_pop, ceil_ideal / ideal_pop - 1.0)
        error("PopulationConstraint infeasible; use larger --pop-dev (need ≥ $(round(t_required, digits=4)) for total_pop=$(base_graph.total_pop), num_dists=$(num_dists))")
    end

    initial_partition = CycleWalk.MultiLevelPartition(graph, constraints, num_dists; rng=rng)
    partition = CycleWalk.LinkCutPartition(initial_partition, rng)

    cycle_walk = CycleWalk.build_two_tree_cycle_walk(constraints)
    internal_walk = CycleWalk.build_one_tree_cycle_walk(constraints)
    proposal = [(two_cycle_walk_frac, cycle_walk), (1.0 - two_cycle_walk_frac, internal_walk)]

    measure = CycleWalk.Measure()
    CycleWalk.push_energy!(measure, CycleWalk.get_log_spanning_forests, gamma)
    CycleWalk.push_energy!(measure, CycleWalk.get_isoperimetric_score, iso_weight)

    ad_param = Dict{String, Any}("popdev" => pop_dev)
    writer = CycleWalk.Writer(measure, constraints, partition, output_file_path; additional_parameters=ad_param)
    CycleWalk.push_writer!(writer, CycleWalk.get_log_spanning_trees)
    CycleWalk.push_writer!(writer, CycleWalk.get_isoperimetric_scores)

    if run_diagnostics
        run_diags = CycleWalk.RunDiagnostics()
        CycleWalk.push_diagnostic!(run_diags, cycle_walk, CycleWalk.AcceptanceRatios(), desc="cycle_walk")
        CycleWalk.push_diagnostic!(run_diags, cycle_walk, CycleWalk.CycleLengthDiagnostic())
        CycleWalk.push_diagnostic!(run_diags, cycle_walk, CycleWalk.DeltaNodesDiagnostic())
        println("output_file_path = " * output_file_path)
        CycleWalk.run_metropolis_hastings!(partition, proposal, measure, steps, rng; writer=writer, output_freq=outfreq, run_diagnostics=run_diags)
    else
        println("output_file_path = " * output_file_path)
        CycleWalk.run_metropolis_hastings!(partition, proposal, measure, steps, rng; writer=writer, output_freq=outfreq)
    end

    CycleWalk.close_writer(writer)
end

main()
