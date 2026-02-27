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

// Set covering includes
#include "instances/setcovering_instance.hpp"
#include "bdd/setcovering_bdd.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"

using namespace std;

struct RunStatsRecord
{
    bool is_tsp_branch;
    bool postprocess_sort_applied;
    int num_solutions;
    long original_width;
    long reduced_width;
    long original_num_nodes;
    long reduced_num_nodes;
    int layer_coupling;
    int pareto_dominance_filtered;
    double pareto_dominance_cpu_s;
    double compile_cpu_s;
    double enum_cpu_s;
    double total_cpu_s;
    double compile_wall_s;
    double enum_wall_s;
    double total_wall_s_end_to_end;

    RunStatsRecord()
        : is_tsp_branch(false),
          postprocess_sort_applied(true),
          num_solutions(0),
          original_width(-1),
          reduced_width(-1),
          original_num_nodes(-1),
          reduced_num_nodes(-1),
          layer_coupling(0),
          pareto_dominance_filtered(0),
          pareto_dominance_cpu_s(0.0),
          compile_cpu_s(0.0),
          enum_cpu_s(0.0),
          total_cpu_s(0.0),
          compile_wall_s(0.0),
          enum_wall_s(0.0),
          total_wall_s_end_to_end(0.0)
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

static bool write_frontier_gzip_csv(const ParetoFrontier *frontier, const int problem_type, const string &out_path, string *error)
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
            ObjType value = frontier->sols[i + o];
            if (problem_type == 3)
            {
                value = -value;
            }

            if (o > 0 && fputc(',', pipe) == EOF)
            {
                ok = false;
                break;
            }
            if (fprintf(pipe, "%d", value) < 0)
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

static void print_perf_log(const string &input_path,
                           const int problem_type,
                           const int method,
                           const string &backend_name,
                           const int cpu_threads,
                           const MultiObjectiveStats *stats,
                           const double compile_wall_s,
                           const double enum_wall_s,
                           const double total_wall_s,
                           const double compile_cpu_s,
                           const double enum_cpu_s,
                           const bool postprocess_sort_applied)
{
    cerr << fixed << setprecision(6);
    cerr << "[perf] input=" << input_path
         << " problem_type=" << problem_type
         << " method=" << method
         << " backend=" << backend_name
         << " cpu_threads=" << cpu_threads << '\n';
    cerr << "[perf] wall_s compile=" << compile_wall_s
         << " enum=" << enum_wall_s
         << " total=" << total_wall_s
         << " postprocess_sort=" << (postprocess_sort_applied ? "applied" : "skipped") << '\n';
    cerr << "[perf] cpu_s compile=" << compile_cpu_s
         << " enum=" << enum_cpu_s << '\n';
    if (stats != NULL)
    {
        cerr << "[perf] phases_s expand_td=" << stats->cpu_expand_td_wall_s
             << " expand_bu=" << stats->cpu_expand_bu_wall_s
             << " recompute_td=" << stats->cpu_recompute_td_wall_s
             << " recompute_bu=" << stats->cpu_recompute_bu_wall_s
             << " dominance=" << stats->cpu_dominance_wall_s
             << " cutset_sort=" << stats->cpu_cutset_sort_wall_s
             << " cutset_convolution=" << stats->cpu_cutset_convolution_wall_s
             << " cutset_partial_merge=" << stats->cpu_cutset_partial_merge_wall_s << '\n';
        cerr << "[perf] counters layers_td=" << stats->cpu_layers_td
             << " layers_bu=" << stats->cpu_layers_bu
             << " nodes_expanded=" << stats->cpu_nodes_expanded
             << " cutset_size=" << stats->cpu_cutset_size
             << " dominance_filtered=" << stats->pareto_dominance_filtered
             << " dominance_cpu_s=" << ((double)stats->pareto_dominance_time) / CLOCKS_PER_SEC << '\n';
    }
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
                              const MultiObjectiveStats *stats,
                              const RunStatsRecord &record,
                              string *error)
{
    ofstream out(out_path.c_str(), ios::out | ios::trunc);
    if (!out.is_open())
    {
        if (error != NULL)
        {
            *error = "could not open output path";
        }
        return false;
    }

    out << fixed << setprecision(6);
    bool first = true;
    auto key_prefix = [&]() {
        if (!first)
        {
            out << ',';
        }
        first = false;
    };

    auto write_string = [&](const string &key, const string &value) {
        key_prefix();
        out << "\"" << key << "\":\"" << json_escape(value) << "\"";
    };
    auto write_bool = [&](const string &key, const bool value) {
        key_prefix();
        out << "\"" << key << "\":" << (value ? "true" : "false");
    };
    auto write_int = [&](const string &key, const long long value) {
        key_prefix();
        out << "\"" << key << "\":" << value;
    };
    auto write_double = [&](const string &key, const double value) {
        key_prefix();
        out << "\"" << key << "\":" << value;
    };

    const double dominance_cpu_s = stats != NULL ? ((double)stats->pareto_dominance_time) / CLOCKS_PER_SEC : 0.0;
    const int layer_coupling = stats != NULL ? stats->layer_coupling : 0;
    const int dominance_filtered = stats != NULL ? stats->pareto_dominance_filtered : 0;

    out << '{';
    write_string("input_path", opts.input_path);
    write_int("problem_type", opts.problem_type);
    write_bool("preprocess", opts.preprocess);
    write_int("method", opts.method);
    write_int("approx_s", opts.approx_s);
    write_int("approx_t", opts.approx_t);
    write_int("dominance", opts.dominance);
    write_string("backend", backend_to_string(opts.backend));
    write_int("cpu_threads", opts.cpu_threads);
    write_int("kernel_version", opts.kernel_version);
    write_bool("save_frontier", opts.save_frontier);
    write_string("frontier_out_path", opts.frontier_out_path);
    write_bool("perf_log", opts.perf_log);
    write_bool("save_stats", opts.save_stats);
    write_string("stats_out_path", opts.stats_out_path);
    write_bool("is_tsp_branch", record.is_tsp_branch);
    write_bool("postprocess_sort_applied", record.postprocess_sort_applied);

    write_int("num_solutions", record.num_solutions);
    write_int("original_width", record.original_width);
    write_int("reduced_width", record.reduced_width);
    write_int("original_num_nodes", record.original_num_nodes);
    write_int("reduced_num_nodes", record.reduced_num_nodes);
    write_int("layer_coupling", stats != NULL ? layer_coupling : record.layer_coupling);
    write_int("pareto_dominance_filtered", stats != NULL ? dominance_filtered : record.pareto_dominance_filtered);

    write_double("compile_cpu_s", record.compile_cpu_s);
    write_double("enum_cpu_s", record.enum_cpu_s);
    write_double("total_cpu_s", record.total_cpu_s);
    write_double("compile_wall_s", record.compile_wall_s);
    write_double("enum_wall_s", record.enum_wall_s);
    write_double("total_wall_s_end_to_end", record.total_wall_s_end_to_end);
    write_double("pareto_dominance_cpu_s", stats != NULL ? dominance_cpu_s : record.pareto_dominance_cpu_s);

    write_double("cpu_expand_td_wall_s", stats != NULL ? stats->cpu_expand_td_wall_s : 0.0);
    write_double("cpu_expand_bu_wall_s", stats != NULL ? stats->cpu_expand_bu_wall_s : 0.0);
    write_double("cpu_recompute_td_wall_s", stats != NULL ? stats->cpu_recompute_td_wall_s : 0.0);
    write_double("cpu_recompute_bu_wall_s", stats != NULL ? stats->cpu_recompute_bu_wall_s : 0.0);
    write_double("cpu_dominance_wall_s", stats != NULL ? stats->cpu_dominance_wall_s : 0.0);
    write_double("cpu_cutset_sort_wall_s", stats != NULL ? stats->cpu_cutset_sort_wall_s : 0.0);
    write_double("cpu_cutset_convolution_wall_s", stats != NULL ? stats->cpu_cutset_convolution_wall_s : 0.0);
    write_double("cpu_cutset_partial_merge_wall_s", stats != NULL ? stats->cpu_cutset_partial_merge_wall_s : 0.0);
    write_int("cpu_layers_td", stats != NULL ? stats->cpu_layers_td : 0);
    write_int("cpu_layers_bu", stats != NULL ? stats->cpu_layers_bu : 0);
    write_int("cpu_nodes_expanded", stats != NULL ? stats->cpu_nodes_expanded : 0);
    write_int("cpu_cutset_size", stats != NULL ? stats->cpu_cutset_size : 0);
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
    const bool preprocess = options.preprocess;
    const int method = options.method;
    bool maximization = true;
    const int approx_S = options.approx_s;
    const int approx_T = options.approx_t;
    const int dominance = options.dominance;
    const Backend backend = options.backend;
    const int kernel_version = options.kernel_version;
    const int cpu_threads = options.cpu_threads;
    const bool save_frontier = options.save_frontier;
    const string frontier_out_path = options.frontier_out_path;
    const bool perf_log = options.perf_log;
    const bool save_stats = options.save_stats;
    const string stats_out_path = options.stats_out_path;

    typedef std::chrono::steady_clock WallClock;
    const WallClock::time_point run_wall_begin = WallClock::now();
    double compilation_wall_s = 0.0;
    double pareto_enum_wall_s = 0.0;

    // For statistical analysis
    Stats timers;
    int bdd_compilation_time = timers.register_name("BDD compilation time");
    int pareto_time = timers.register_name("BDD pareto time");
    int approx_time = timers.register_name("BDD approximation time");
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

        // if (preprocess) {
        //     // Reorder variables
        //     inst.reorder_coefficients();
        // }

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

        // Compute approximation
        if (approx_S != 0)
        {
            timers.start_timer(approx_time);
            // BDDMultiObj::approximate_pareto_frontier_bottomup(bdd, approx_S, approx_T);
            // BDDMultiObj::approximate_pareto_frontier_topdown(bdd, approx_S, approx_T);
            // BDDMultiObj::approximate_pareto_frontier_topdown_dominance(bdd, approx_S, approx_T);
            timers.end_timer(approx_time);
        }

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

    // --- Set Covering ---
    else if (problem_type == 3)
    {
        // set objective sense
        maximization = false;

        // read instance
        SetCoveringInstance setcover(input_path.c_str());

        // preprocess
        if (preprocess)
        {
            setcover.minimize_bandwidth();
        }

        // create BDD
        SetCoveringBDDConstructor bddConstructor(&setcover, setcover.objs);
        bdd = bddConstructor.generate_exact();

        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();

        // Reduce BDD
        // BDDAlg::reduce(bdd);

        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();

        // --- TSP ---
    }
    else if (problem_type == 4)
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

        MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
        statsMultiObj->cpu_perf_enabled = (backend == BACKEND_CPU) && (perf_log || save_stats);
        ParetoFrontier *pareto_frontier = NULL;

        if (method == 1) { // Top-down
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(mdd, statsMultiObj, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but top-down enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_topdown(mdd, statsMultiObj, cpu_threads);
            }
        } else if (method == 3) { // Coupled
            if (backend == BACKEND_GPU) {
                string cuda_reason;
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(mdd, statsMultiObj, &cuda_reason, kernel_version);
                if (pareto_frontier == NULL) {
                    cout << "Error - GPU backend requested but coupled enumeration failed";
                    if (!cuda_reason.empty()) cout << ": " << cuda_reason;
                    cout << endl;
                    exit(1);
                }
            } else {
                pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj, cpu_threads);
            }
        } else {
            cout << "Error - method " << method << " not valid for TSP" << endl;
            exit(1);
        }
        pareto_tsp_cpu = clock() - pareto_tsp_cpu;
        pareto_enum_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_tsp_wall_begin).count();

        assert(pareto_frontier != NULL);
        pareto_frontier->sort_lexicographic_ascending();

        if (save_frontier)
        {
            string save_error;
            if (!write_frontier_gzip_csv(pareto_frontier, problem_type, frontier_out_path, &save_error))
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

        const double total_wall_s_end_to_end =
            std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

        RunStatsRecord stats_record;
        stats_record.is_tsp_branch = true;
        stats_record.postprocess_sort_applied = true;
        stats_record.num_solutions = pareto_frontier->get_num_sols();
        stats_record.original_width = -1;
        stats_record.reduced_width = -1;
        stats_record.original_num_nodes = -1;
        stats_record.reduced_num_nodes = -1;
        stats_record.layer_coupling = statsMultiObj->layer_coupling;
        stats_record.pareto_dominance_filtered = statsMultiObj->pareto_dominance_filtered;
        stats_record.pareto_dominance_cpu_s = ((double)statsMultiObj->pareto_dominance_time) / CLOCKS_PER_SEC;
        stats_record.compile_cpu_s = ((double)compilation_tsp) / CLOCKS_PER_SEC;
        stats_record.enum_cpu_s = ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC;
        stats_record.total_cpu_s = stats_record.compile_cpu_s + stats_record.enum_cpu_s;
        stats_record.compile_wall_s = compilation_wall_s;
        stats_record.enum_wall_s = pareto_enum_wall_s;
        stats_record.total_wall_s_end_to_end = total_wall_s_end_to_end;

        cout << pareto_frontier->get_num_sols() << endl;
        cout << (double)(compilation_tsp + pareto_tsp_cpu) / CLOCKS_PER_SEC << endl;
        cout << (double)compilation_tsp / CLOCKS_PER_SEC;
        cout << "\t" << pareto_tsp_cpu / CLOCKS_PER_SEC;
        cout << "\t" << compilation_wall_s;
        cout << "\t" << pareto_enum_wall_s;
        cout << "\t" << total_wall_s_end_to_end;
        cout << endl;

        if (perf_log)
        {
            const double total_wall_s = total_wall_s_end_to_end;
            const string backend_name = backend_to_string(backend);
            print_perf_log(input_path,
                           problem_type,
                           method,
                           backend_name,
                           cpu_threads,
                           statsMultiObj,
                           compilation_wall_s,
                           pareto_enum_wall_s,
                           total_wall_s,
                           ((double)compilation_tsp) / CLOCKS_PER_SEC,
                           ((double)pareto_tsp_cpu) / CLOCKS_PER_SEC,
                           true);
        }

        if (save_stats)
        {
            string stats_error;
            if (!write_stats_jsonl(stats_out_path, options, statsMultiObj, stats_record, &stats_error))
            {
                cerr << "Warning - failed to save stats to '" << stats_out_path << "'";
                if (!stats_error.empty())
                {
                    cerr << ": " << stats_error;
                }
                cerr << '\n';
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

    // Initialize multiobjective stats
    MultiObjectiveStats *statsMultiObj = new MultiObjectiveStats;
    statsMultiObj->cpu_perf_enabled = (backend == BACKEND_CPU) && (perf_log || save_stats);

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
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(bdd, maximization, problem_type, dominance, statsMultiObj, &cuda_reason, kernel_version);
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
            pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, statsMultiObj, cpu_threads);
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
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, statsMultiObj, cpu_threads);
    }
    else if (method == 3)
    {
        // -- Dynamic layer cutset --
        if (backend == BACKEND_GPU)
        {
            cout << "Error - GPU backend is unsupported for method 3." << endl;
            exit(1);
        }
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, statsMultiObj, cpu_threads);
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
    pareto_enum_wall_s = std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - pareto_wall_begin).count();

    pareto_frontier->sort_lexicographic_ascending();

    if (save_frontier)
    {
        string save_error;
        if (!write_frontier_gzip_csv(pareto_frontier, problem_type, frontier_out_path, &save_error))
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

    double total_time = (timers.get_time(bdd_compilation_time) + timers.get_time(approx_time) + timers.get_time(pareto_time));
    (void)total_time;

    // cout << "\nPareto frontier: " << endl;
    // cout << "\tNumber of solutions: " << pareto_frontier->get_num_sols() << endl;
    // cout << "\n\tBDD time: " << timers.get_time(bdd_compilation_time) << endl;
    // cout << "\tApproximation filtering time: " << timers.get_time(approx_time) << endl;
    // cout << "\tPareto time: " << timers.get_time(pareto_time) << endl;
    // cout << "\tTotal time: " << total_time << endl;
    // cout << endl;

    // cout << "\n\nPareto frontier: " << endl;
    // pareto_frontier->print();
    // cout << endl;

    // // Statistics file
    // ofstream stats("stats.txt", ios::app);
    // stats << argv[1];
    // stats << "\t" << problem_type;
    // stats << "\t" << NOBJS;
    // stats << "\t" << preprocess;
    // stats << "\t" << method;
    // stats << "\t" << approx_S;
    // stats << "\t" << approx_T;
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

    const double total_wall_s_end_to_end =
        std::chrono::duration_cast<std::chrono::duration<double> >(WallClock::now() - run_wall_begin).count();

    RunStatsRecord stats_record;
    stats_record.is_tsp_branch = false;
    stats_record.postprocess_sort_applied = true;
    stats_record.num_solutions = pareto_frontier->get_num_sols();
    stats_record.original_width = original_width;
    stats_record.reduced_width = reduced_width;
    stats_record.original_num_nodes = original_num_nodes;
    stats_record.reduced_num_nodes = reduced_num_nodes;
    stats_record.layer_coupling = statsMultiObj->layer_coupling;
    stats_record.pareto_dominance_filtered = statsMultiObj->pareto_dominance_filtered;
    stats_record.pareto_dominance_cpu_s = ((double)statsMultiObj->pareto_dominance_time) / CLOCKS_PER_SEC;
    stats_record.compile_cpu_s = timers.get_time(bdd_compilation_time);
    stats_record.enum_cpu_s = timers.get_time(pareto_time);
    stats_record.total_cpu_s = stats_record.compile_cpu_s + stats_record.enum_cpu_s;
    stats_record.compile_wall_s = compilation_wall_s;
    stats_record.enum_wall_s = pareto_enum_wall_s;
    stats_record.total_wall_s_end_to_end = total_wall_s_end_to_end;

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
    cout << "\t" << statsMultiObj->layer_coupling;
    cout << "\t" << statsMultiObj->pareto_dominance_filtered;
    cout << "\t" << ((double)statsMultiObj->pareto_dominance_time) / CLOCKS_PER_SEC;
    cout << "\t" << compilation_wall_s;
    cout << "\t" << pareto_enum_wall_s;
    cout << "\t" << total_wall_s_end_to_end;
    cout << endl;

    if (perf_log)
    {
        const double total_wall_s = total_wall_s_end_to_end;
        const string backend_name = backend_to_string(backend);
        print_perf_log(input_path,
                       problem_type,
                       method,
                       backend_name,
                       cpu_threads,
                       statsMultiObj,
                       compilation_wall_s,
                       pareto_enum_wall_s,
                       total_wall_s,
                       timers.get_time(bdd_compilation_time),
                       timers.get_time(pareto_time),
                       true);
    }

    if (save_stats)
    {
        string stats_error;
        if (!write_stats_jsonl(stats_out_path, options, statsMultiObj, stats_record, &stats_error))
        {
            cerr << "Warning - failed to save stats to '" << stats_out_path << "'";
            if (!stats_error.empty())
            {
                cerr << ": " << stats_error;
            }
            cerr << '\n';
        }
    }

    return 0;
}
