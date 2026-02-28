// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <iostream>
#include <cstdlib>
#include <string>
#include <cstdio>
#include <chrono>
#include <iomanip>
#include <fstream>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
#include "util/cli_parser.hpp"
#include "util/stats.hpp"
#include "util/util.hpp"
#include "bdd/pareto_frontier.hpp"

// Knapsack includes
#include "instances/knapsack_instance.hpp"
#include "bdd/knapsack_bdd.hpp"

// Set packing / Independent set includes
#include "instances/indepset_instance.hpp"
#include "instances/setpacking_instance.hpp"
#include "bdd/indepset_bdd.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"

using namespace std;

struct RunSummaryStats
{
    bool is_tsp_branch;
    bool postprocess_sort_applied;
    int num_solutions;
    long original_width;
    long reduced_width;
    long original_num_nodes;
    long reduced_num_nodes;
    int layer_coupling;
    int dominance_filtered_total;
    double cpu_dominance_s;
    double cpu_compile_s;
    double cpu_enumeration_s;
    double cpu_total_s;
    double wall_compile_s;
    double wall_enumeration_s;
    double wall_total_end_to_end_s;
    string status_state;
    string status_error_message;

    RunSummaryStats()
        : is_tsp_branch(false),
          postprocess_sort_applied(true),
          num_solutions(0),
          original_width(-1),
          reduced_width(-1),
          original_num_nodes(-1),
          reduced_num_nodes(-1),
          layer_coupling(0),
          dominance_filtered_total(0),
          cpu_dominance_s(0.0),
          cpu_compile_s(0.0),
          cpu_enumeration_s(0.0),
          cpu_total_s(0.0),
          wall_compile_s(0.0),
          wall_enumeration_s(0.0),
          wall_total_end_to_end_s(0.0),
          status_state("ok"),
          status_error_message("")
    {
    }
};

static string shell_single_quote(const string &value)
{
    string quoted = "'";
    for (char c : value)
    {
        if (c == '\'')
        {
            quoted += "'\"'\"'";
        }
        else
        {
            quoted += c;
        }
    }
    quoted += "'";
    return quoted;
}

static bool write_frontier_gzip_csv(const ParetoFrontier *frontier, const string &out_path, string *error)
{
    if (frontier == NULL)
    {
        if (error != NULL)
        {
            *error = "frontier is null";
        }
        return false;
    }

    if (out_path.empty())
    {
        if (error != NULL)
        {
            *error = "output path is empty";
        }
        return false;
    }

    if (frontier->sols.size() % NOBJS != 0)
    {
        if (error != NULL)
        {
            *error = "frontier has invalid dimension";
        }
        return false;
    }

    const string command = "gzip -c > " + shell_single_quote(out_path);
    FILE *pipe = popen(command.c_str(), "w");
    if (pipe == NULL)
    {
        if (error != NULL)
        {
            *error = "could not launch gzip";
        }
        return false;
    }

    bool ok = true;
    for (size_t i = 0; i < frontier->sols.size() && ok; i += NOBJS)
    {
        for (int o = 0; o < NOBJS; ++o)
        {
            if (o > 0 && fputc(',', pipe) == EOF)
            {
                ok = false;
                break;
            }
            if (fprintf(pipe, "%d", frontier->sols[i + o]) < 0)
            {
                ok = false;
                break;
            }
        }
        if (ok && fputc('\n', pipe) == EOF)
        {
            ok = false;
        }
    }

    int close_status = pclose(pipe);
    if (!ok)
    {
        if (error != NULL)
        {
            *error = "failed while writing compressed frontier";
        }
        return false;
    }

    if (close_status != 0)
    {
        if (error != NULL)
        {
            *error = "gzip exited with non-zero status";
        }
        return false;
    }

    return true;
}

static string json_escape(const string &value)
{
    string escaped;
    escaped.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i)
    {
        const unsigned char c = static_cast<unsigned char>(value[i]);
        switch (c)
        {
        case '\"':
            escaped += "\\\"";
            break;
        case '\\':
            escaped += "\\\\";
            break;
        case '\b':
            escaped += "\\b";
            break;
        case '\f':
            escaped += "\\f";
            break;
        case '\n':
            escaped += "\\n";
            break;
        case '\r':
            escaped += "\\r";
            break;
        case '\t':
            escaped += "\\t";
            break;
        default:
            if (c < 0x20)
            {
                static const char hex[] = "0123456789abcdef";
                escaped += "\\u00";
                escaped += hex[(c >> 4) & 0x0F];
                escaped += hex[c & 0x0F];
            }
            else
            {
                escaped += static_cast<char>(c);
            }
            break;
        }
    }
    return escaped;
}

static bool write_stats_jsonl(const string &out_path,
                              const CliOptions &opts,
                              const EnumerationStats *stats,
                              const RunSummaryStats &record,
                              string *error)
{
    ofstream out(out_path.c_str(), ios::out | ios::app);
    if (!out.is_open())
    {
        if (error != NULL)
        {
            *error = "could not open output path";
        }
        return false;
    }

    out << fixed << setprecision(6);
    const double cpu_dominance_s = stats != NULL ? ((double)stats->cpu_ticks_dominance) / CLOCKS_PER_SEC : record.cpu_dominance_s;
    const int layer_coupling = stats != NULL ? stats->layer_coupling : record.layer_coupling;
    const int dominance_filtered_total = stats != NULL ? stats->dominance_filtered_total : record.dominance_filtered_total;
    const long long work_candidates_total = stats != NULL ? stats->work_candidates_total : 0;
    const long long work_frontier_survivors_total = stats != NULL ? stats->work_frontier_survivors_total : 0;
    const long long work_frontier_peak_points = stats != NULL ? stats->work_frontier_peak_points : 0;
    const long long work_join_products_total = stats != NULL ? stats->work_join_products_total : 0;

    out << "{";
    out << "\"schema_version\":1,";
    out << "\"identity\":{";
    out << "\"input_path\":\"" << json_escape(opts.input_path) << "\",";
    out << "\"problem_type\":" << opts.problem_type << ",";
    out << "\"method\":" << opts.method << ",";
    out << "\"dominance\":" << opts.dominance << ",";
    out << "\"backend\":\"" << backend_to_string(opts.backend) << "\",";
    out << "\"cpu_threads\":" << opts.cpu_threads << ",";
    out << "\"kernel_version\":" << opts.kernel_version;
    out << "},";

    out << "\"outputs\":{";
    out << "\"num_solutions\":" << record.num_solutions << ",";
    out << "\"save_frontier\":" << (opts.save_frontier ? "true" : "false") << ",";
    out << "\"frontier_out_path\":\"" << json_escape(opts.frontier_out_path) << "\"";
    out << "},";

    out << "\"timing\":{";
    out << "\"cpu\":{";
    out << "\"cpu_compile_s\":" << record.cpu_compile_s << ",";
    out << "\"cpu_enumeration_s\":" << record.cpu_enumeration_s << ",";
    out << "\"cpu_total_s\":" << record.cpu_total_s << ",";
    out << "\"cpu_dominance_s\":" << cpu_dominance_s;
    out << "},";
    out << "\"wall\":{";
    out << "\"wall_compile_s\":" << record.wall_compile_s << ",";
    out << "\"wall_enumeration_s\":" << record.wall_enumeration_s << ",";
    out << "\"wall_total_end_to_end_s\":" << record.wall_total_end_to_end_s;
    out << "}";
    out << "},";

    out << "\"work\":{";
    out << "\"work_candidates_total\":" << work_candidates_total << ",";
    out << "\"work_frontier_survivors_total\":" << work_frontier_survivors_total << ",";
    out << "\"work_frontier_peak_points\":" << work_frontier_peak_points << ",";
    out << "\"work_join_products_total\":" << work_join_products_total;
    out << "},";

    out << "\"dominance\":{";
    out << "\"dominance_filtered_total\":" << dominance_filtered_total << ",";
    out << "\"cpu_ticks_dominance\":" << (stats != NULL ? stats->cpu_ticks_dominance : 0);
    out << "},";

    out << "\"structure\":{";
    out << "\"is_tsp_branch\":" << (record.is_tsp_branch ? "true" : "false") << ",";
    out << "\"postprocess_sort_applied\":" << (record.postprocess_sort_applied ? "true" : "false") << ",";
    out << "\"original_width\":" << record.original_width << ",";
    out << "\"reduced_width\":" << record.reduced_width << ",";
    out << "\"original_num_nodes\":" << record.original_num_nodes << ",";
    out << "\"reduced_num_nodes\":" << record.reduced_num_nodes << ",";
    out << "\"layer_coupling\":" << layer_coupling;
    out << "},";

    out << "\"perf\":{";
    out << "\"wall_expand_td_s\":" << (stats != NULL ? stats->wall_expand_td_s : 0.0) << ",";
    out << "\"wall_expand_bu_s\":" << (stats != NULL ? stats->wall_expand_bu_s : 0.0) << ",";
    out << "\"wall_recompute_td_s\":" << (stats != NULL ? stats->wall_recompute_td_s : 0.0) << ",";
    out << "\"wall_recompute_bu_s\":" << (stats != NULL ? stats->wall_recompute_bu_s : 0.0) << ",";
    out << "\"wall_dominance_s\":" << (stats != NULL ? stats->wall_dominance_s : 0.0) << ",";
    out << "\"wall_cutset_sort_s\":" << (stats != NULL ? stats->wall_cutset_sort_s : 0.0) << ",";
    out << "\"wall_cutset_convolution_s\":" << (stats != NULL ? stats->wall_cutset_convolution_s : 0.0) << ",";
    out << "\"wall_cutset_partial_merge_s\":" << (stats != NULL ? stats->wall_cutset_partial_merge_s : 0.0) << ",";
    out << "\"wall_pack_transfer_s\":" << (stats != NULL ? stats->wall_pack_transfer_s : 0.0) << ",";
    out << "\"wall_join_s\":" << (stats != NULL ? stats->wall_join_s : 0.0) << ",";
    out << "\"cpu_layers_td\":" << (stats != NULL ? stats->cpu_layers_td : 0) << ",";
    out << "\"cpu_layers_bu\":" << (stats != NULL ? stats->cpu_layers_bu : 0) << ",";
    out << "\"cpu_nodes_expanded\":" << (stats != NULL ? stats->cpu_nodes_expanded : 0) << ",";
    out << "\"cpu_cutset_size\":" << (stats != NULL ? stats->cpu_cutset_size : 0);
    out << "},";

    out << "\"status\":{";
    out << "\"status_state\":\"" << json_escape(record.status_state) << "\",";
    out << "\"status_error_message\":\"" << json_escape(record.status_error_message) << "\"";
    out << "}";
    out << "}\n";

    if (!out.good())
    {
        if (error != NULL)
        {
            *error = "failed while writing JSONL output";
        }
        return false;
    }
    return true;
}

//
// Main function
//
int main(int argc, char *argv[])
{
    CliOptions options;
    string parse_error;
    if (!parse_cli_args(argc, argv, &options, &parse_error))
    {
        if (!parse_error.empty())
        {
            cout << parse_error << endl;
        }
        print_usage();
        exit(1);
    }

    const string input_path = options.input_path;
    const int problem_type = options.problem_type;
    const int method = options.method;
    bool maximization = true;
    const int dominance = options.dominance;
    const Backend backend = options.backend;
    const int kernel_version = options.kernel_version;
    const int cpu_threads = options.cpu_threads;
    const bool save_frontier = options.save_frontier;
    const string frontier_out_path = options.frontier_out_path;
    const bool save_stats = options.save_stats;
    const string stats_out_path = options.stats_out_path;

    typedef std::chrono::steady_clock WallClock;
    const WallClock::time_point run_wall_begin = WallClock::now();
    double compilation_wall_s = 0.0;
    double pareto_wall_enumeration_s = 0.0;

    // For statistical analysis
    Stats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int pareto_time = timers.register_name("BDD pareto time");
    long int original_width;
    long int reduced_width;
    long int original_num_nodes;
    long int reduced_num_nodes;

    // Read problem instance and construct BDD
    BDD *bdd = NULL;
    vector<vector<int>> obj_coeffs;
    const WallClock::time_point compilation_wall_begin = WallClock::now();
    timers.start_timer(bdd_compilation_time);

    // --- Knapsack ---
    if (problem_type == 1)
    {

        // Read instance
        KnapsackInstance inst;
        inst.read(const_cast<char *>(input_path.c_str()));

        // Construct BDD
        KnapsackBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();
        // obj_coeffs = inst.obj_coeffs;

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // cout << "Original width: " << original_width << " - number of nodes: " << original_num_nodes << endl;

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        // Reduce BDD
        BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // cout << "Reduced-2 width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);

        //        bdd->print();
    }

    // --- Set Packing ---
    else if (problem_type == 2)
    {

        // read instance
        SetPackingInstance setpack(input_path.c_str());

        // create associated independent set instance
        IndepSetInst *inst = setpack.create_indepset_instance();

        // generate independent set BDD
        IndepSetBDDConstructor bddConstructor(inst, setpack.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }

    // --- TSP ---
    else if (problem_type == 3)
    {
        // Read instance
        TSPInstance inst;
        inst.read(input_path.c_str());

        // Construct MDD
        const WallClock::time_point compilation_tsp_wall_begin = WallClock::now();
        clock_t compilation_tsp = clock();

        MDDTSPConstructor mddCons(&inst);
        MDD *mdd = mddCons.generate_exact();
        assert(mdd != NULL);

        compilation_tsp = clock() - compilation_tsp;
        compilation_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - compilation_tsp_wall_begin).count();

        // Generate frontier (timed region excludes final lexicographic sort)
        const WallClock::time_point pareto_tsp_wall_begin = WallClock::now();
        clock_t pareto_tsp_cpu = clock();

        // Solver-owned stats populated during frontier enumeration.
        EnumerationStats *enumeration_stats = new EnumerationStats;
        enumeration_stats->cpu_perf_enabled = (backend == BACKEND_CPU) && save_stats;
        ParetoFrontier *pareto_frontier = NULL;

        if (method == 1) { // Top-down
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(mdd, enumeration_stats, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but top-down enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown(mdd, enumeration_stats, cpu_threads);
            }
        } else if (method == 3) { // Coupled
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(mdd, enumeration_stats, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but coupled enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, enumeration_stats, cpu_threads);
            }
        } else {
            cout << "Error - method " << method << " not valid for TSP" << endl;
            exit(1);
        }
        pareto_tsp_cpu = clock() - pareto_tsp_cpu;
        pareto_wall_enumeration_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_tsp_wall_begin).count();

        assert(pareto_frontier != NULL);
        pareto_frontier->sort_lexicographic_ascending();

        if (save_frontier)
        {
            string save_error;
            if (!write_frontier_gzip_csv(pareto_frontier, frontier_out_path, &save_error))
            {
                cout << "Error - failed to save frontier to '" << frontier_out_path << "'";
                if (!save_error.empty())
                {
                    cout << ": " << save_error;
                }
                cout << endl;
                exit(1);
            }
        }

        const double wall_total_end_to_end_s =
            std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

        // Run-level summary assembled in main for reporting/output only.
        RunSummaryStats run_summary;
        run_summary.is_tsp_branch = true;
        run_summary.postprocess_sort_applied = true;
        run_summary.num_solutions = pareto_frontier->get_num_sols();
        run_summary.original_width = -1;
        run_summary.reduced_width = -1;
        run_summary.original_num_nodes = -1;
        run_summary.reduced_num_nodes = -1;
        run_summary.layer_coupling = enumeration_stats->layer_coupling;
        run_summary.dominance_filtered_total = enumeration_stats->dominance_filtered_total;
        run_summary.cpu_dominance_s = ((double)enumeration_stats->cpu_ticks_dominance) / CLOCKS_PER_SEC;
        run_summary.cpu_compile_s = ((double)compilation_tsp) / CLOCKS_PER_SEC;
        run_summary.cpu_enumeration_s = ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC;
        run_summary.cpu_total_s = run_summary.cpu_compile_s + run_summary.cpu_enumeration_s;
        run_summary.wall_compile_s = compilation_wall_s;
        run_summary.wall_enumeration_s = pareto_wall_enumeration_s;
        run_summary.wall_total_end_to_end_s = wall_total_end_to_end_s;

        cout << pareto_frontier->get_num_sols() << endl;
        cout << (double)(compilation_tsp + pareto_tsp_cpu) / CLOCKS_PER_SEC << endl;
        cout << (double)compilation_tsp / CLOCKS_PER_SEC;
        cout << "\t" << ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC;
        cout << "\t" << compilation_wall_s;
        cout << "\t" << pareto_wall_enumeration_s;
        cout << "\t" << wall_total_end_to_end_s;
        cout << endl;

        if (save_stats)
        {
            string stats_error;
            if (!write_stats_jsonl(stats_out_path, options, enumeration_stats, run_summary, &stats_error))
            {
                cerr << "Error - failed to save stats to '" << stats_out_path << "'";
                if (!stats_error.empty())
                {
                    cerr << ": " << stats_error;
                }
                cerr << '\n';
                exit(1);
            }
        }

        return 0;
    }
    else
    {
        cout << "Error - problem type not recognized" << endl;
        exit(1);
    }

    timers.end_timer(bdd_compilation_time);
    compilation_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - compilation_wall_begin).count();

    // cout << "\nBDD Info:\n";
    // cout << "\tOriginal width: " << original_width << endl;
    // cout << "\tOriginal number of nodes: " << original_num_nodes << endl;
    // cout << "\n\tReduced width: " << reduced_width << endl;
    // cout << "\tReduced number of nodes: " << reduced_num_nodes << endl;
    // cout << "\n\tBDD compilation total time: " << timers.get_time(bdd_compilation_time) << endl;

    // Initialize enumeration stats
    // Solver-owned stats populated during frontier enumeration.
    EnumerationStats *enumeration_stats = new EnumerationStats;
    enumeration_stats->cpu_perf_enabled = (backend == BACKEND_CPU) && save_stats;

    // Compute pareto frontier based on methodology
    // cout << "\n\nComputing pareto frontier..." << endl;
    ParetoFrontier *pareto_frontier = NULL;
    const WallClock::time_point pareto_wall_begin = WallClock::now();
    timers.start_timer(pareto_time);

    if (method == 1)
    {
        // -- Optimal BFS algorithm: top-down --
        if (backend == BACKEND_GPU)
        {
            string cuda_reason;
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(bdd, maximization, problem_type, dominance, enumeration_stats, &cuda_reason, kernel_version);
            if (pareto_frontier == NULL)
            {
                cout << "Error - GPU backend requested but top-down enumeration failed";
                if (!cuda_reason.empty())
                {
                    cout << ": " << cuda_reason;
                }
                cout << endl;
                exit(1);
            }
        }
        else
        {
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, enumeration_stats, cpu_threads);
        }
    }
    else if (method == 2)
    {
        // -- Optimal BFS algorithm: bottom-up --
        if (backend == BACKEND_GPU)
        {
            cout << "Error - GPU backend is unsupported for method 2." << endl;
            exit(1);
        }
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, enumeration_stats, cpu_threads);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset --
        if (backend == BACKEND_GPU)
        {
            cout << "Error - GPU backend is unsupported for method 3." << endl;
            exit(1);
        }
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, enumeration_stats, cpu_threads);
    }
    else
    {
        cout << "Error - method not recognized" << endl;
        exit(1);
    }

    if (pareto_frontier == NULL)
    {
        cout << "\nError - pareto frontier not computed" << endl;
        exit(1);
    }
    timers.end_timer(pareto_time);
    pareto_wall_enumeration_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_wall_begin).count();

    pareto_frontier->sort_lexicographic_ascending();

    if (save_frontier)
    {
        string save_error;
        if (!write_frontier_gzip_csv(pareto_frontier, frontier_out_path, &save_error))
        {
            cout << "Error - failed to save frontier to '" << frontier_out_path << "'";
            if (!save_error.empty())
            {
                cout << ": " << save_error;
            }
            cout << endl;
            exit(1);
        }
    }

    // cout << "\nPareto frontier: " << endl;
    // cout << "\tNumber of solutions: " << pareto_frontier->get_num_sols() << endl;
    // cout << "\n\tBDD time: " << timers.get_time(bdd_compilation_time) << endl;
    // cout << "\tPareto time: " << timers.get_time(pareto_time) << endl;
    // cout << "\tTotal time: " << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time)) << endl;
    // cout << endl;

    // cout << "\n\nPareto frontier: " << endl;
    // pareto_frontier->print();
    // cout << endl;

    // // Statistics file
    // ofstream stats("stats.txt", ios::app);
    // stats << argv[1];
    // stats << "\t" << problem_type;
    // stats << "\t" << NOBJS;
    // stats << "\t" << method;
    // stats << "\t" << pareto_frontier->get_num_sols();
    // stats << "\t" << original_width;
    // stats << "\t" << original_num_nodes;
    // stats << "\t" << reduced_width;
    // stats << "\t" << reduced_num_nodes;
    // stats << "\t" << timers.get_time(bdd_compilation_time);
    // stats << "\t" << timers.get_time(pareto_time);
    // stats << "\t" << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time));
    // stats << endl;
    // stats.close();

    const double wall_total_end_to_end_s =
        std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

    // Run-level summary assembled in main for reporting/output only.
    RunSummaryStats run_summary;
    run_summary.is_tsp_branch = false;
    run_summary.postprocess_sort_applied = true;
    run_summary.num_solutions = pareto_frontier->get_num_sols();
    run_summary.original_width = original_width;
    run_summary.reduced_width = reduced_width;
    run_summary.original_num_nodes = original_num_nodes;
    run_summary.reduced_num_nodes = reduced_num_nodes;
    run_summary.layer_coupling = enumeration_stats->layer_coupling;
    run_summary.dominance_filtered_total = enumeration_stats->dominance_filtered_total;
    run_summary.cpu_dominance_s = ((double)enumeration_stats->cpu_ticks_dominance) / CLOCKS_PER_SEC;
    run_summary.cpu_compile_s = timers.get_time(bdd_compilation_time);
    run_summary.cpu_enumeration_s = timers.get_time(pareto_time);
    run_summary.cpu_total_s = run_summary.cpu_compile_s + run_summary.cpu_enumeration_s;
    run_summary.wall_compile_s = compilation_wall_s;
    run_summary.wall_enumeration_s = pareto_wall_enumeration_s;
    run_summary.wall_total_end_to_end_s = wall_total_end_to_end_s;

    cout << pareto_frontier->get_num_sols() << endl;
    cout << (timers.get_time(bdd_compilation_time) + timers.get_time(pareto_time)) << endl;

    cout << method;
    cout << "\t" << dominance;
    cout << "\t" << original_width;
    cout << "\t" << reduced_width;
    cout << "\t" << original_num_nodes;
    cout << "\t" << reduced_num_nodes;
    cout << "\t" << timers.get_time(bdd_compilation_time);
    cout << "\t" << timers.get_time(pareto_time);
    cout << "\t" << enumeration_stats->layer_coupling;
    cout << "\t" << enumeration_stats->dominance_filtered_total;
    cout << "\t" << ((double)enumeration_stats->cpu_ticks_dominance) / CLOCKS_PER_SEC;
    cout << "\t" << compilation_wall_s;
    cout << "\t" << pareto_wall_enumeration_s;
    cout << "\t" << wall_total_end_to_end_s;
    cout << endl;

    if (save_stats)
    {
        string stats_error;
        if (!write_stats_jsonl(stats_out_path, options, enumeration_stats, run_summary, &stats_error))
        {
            cerr << "Error - failed to save stats to '" << stats_out_path << "'";
            if (!stats_error.empty())
            {
                cerr << ": " << stats_error;
            }
            cerr << '\n';
            exit(1);
        }
    }

    return 0;
}
