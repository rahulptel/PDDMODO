// --------------------------------------------------
// CLI parser
// --------------------------------------------------

#include "cli_parser.hpp"
#include "omp_compat.hpp"

#include <cerrno>
#include <climits>
#include <cstdlib>
#include <iostream>

using namespace std;

CliOptions::CliOptions()
    : problem_type(0),
      method(0),
      state_dominance(0),
      backend(BACKEND_CPU),
      cpu_threads(1),
      kernel_version(-1),
      cpu_kernel(CPU_KERNEL_1),
      save_frontier(false),
      save_stats(false)
{
}

static string derive_default_frontier_path(const string &input_path)
{
    string base = input_path;
    size_t slash_pos = base.find_last_of("/\\");
    if (slash_pos != string::npos)
    {
        base = base.substr(slash_pos + 1);
    }

    size_t dot_pos = base.find_last_of('.');
    if (dot_pos != string::npos)
    {
        base = base.substr(0, dot_pos);
    }

    if (base.empty())
    {
        base = "frontier";
    }

    return base + ".frontier.csv.gz";
}

static string derive_default_stats_path(const string &input_path)
{
    string base = input_path;
    size_t slash_pos = base.find_last_of("/\\");
    if (slash_pos != string::npos)
    {
        base = base.substr(slash_pos + 1);
    }

    size_t dot_pos = base.find_last_of('.');
    if (dot_pos != string::npos)
    {
        base = base.substr(0, dot_pos);
    }

    if (base.empty())
    {
        base = "stats";
    }

    return base + ".stats.jsonl";
}

static bool parse_positive_int(const string &value, int *out_value)
{
    if (value.empty())
    {
        return false;
    }

    errno = 0;
    char *end_ptr = NULL;
    long parsed = strtol(value.c_str(), &end_ptr, 10);
    if (errno != 0 || end_ptr == value.c_str() || *end_ptr != '\0')
    {
        return false;
    }
    if (parsed <= 0 || parsed > INT_MAX)
    {
        return false;
    }

    if (out_value != NULL)
    {
        *out_value = static_cast<int>(parsed);
    }
    return true;
}

const char *backend_to_string(const Backend backend)
{
    return backend == BACKEND_GPU ? "gpu" : "cpu";
}

void print_usage()
{
    cout << '\n';
    cout << "Usage: multiobj_nobjs<NUM_OBJS> [input file] [problem type] [method] [state_dominance] [options]\n";

    cout << "\n\twhere:";

    cout << "\n";
    cout << "\t\tproblem_type = 1: knapsack\n";
    cout << "\t\tproblem_type = 2: set packing\n";
    cout << "\t\tproblem_type = 3: TSP\n";

    cout << "\n";
    cout << "\t\tmethod = 1: top-down BFS\n";
    cout << "\t\tmethod = 2: bottom-up BFS\n";
    cout << "\t\tmethod = 3: dynamic layer cutset\n";

    cout << "\n";
    cout << "\t\tstate_dominance = 0:  disable state dominance\n";
    cout << "\t\tstate_dominance = 1:  state dominance strategy 1\n";

    cout << "\n";
    cout << "\t\tNamed backend options:\n";
    cout << "\t\t\t--backend cpu|gpu\n";
    cout << "\t\t\t--cpu-threads <N>   (cpu only)\n";
    cout << "\t\t\t--cpu-kernel <K>    (cpu only, methods 1 and 3, K in {1,3})\n";
    cout << "\t\t\t--kernel <K>        (gpu only, K in {1,2,3})\n";
    cout << "\t\tShorthand backend options:\n";
    cout << "\t\t\tcpu [N]\n";
    cout << "\t\t\tgpu [K]\n";
    cout << "\t\tbackend omitted defaults to cpu\n";
#if CUMODD_HAS_OPENMP
    cout << "\t\tcpu threads default to OMP_NUM_THREADS if valid, otherwise 1\n";
#else
    cout << "\t\tthis binary was built without OpenMP: CPU runs serially (1 thread)\n";
    cout << "\t\t--cpu-threads and cpu [N] are unsupported; rebuild with ENABLE_OPENMP=1\n";
#endif

    cout << "\n";
    cout << "\t\tkernel = 1: one block per node\n";
    cout << "\t\tkernel = 2: fixed number of blocks per node (2D grid)\n";
    cout << "\t\tkernel = 3: dynamic blocks per node with binary-search destination lookup (1D grid)\n";
    cout << "\t\tkernel omitted with backend=gpu: defaults by problem type\n";
    cout << "\t\t\tproblem_type=1 (knapsack): 1\n";
    cout << "\t\t\tproblem_type=2 (set packing): 2\n";
    cout << "\t\t\tproblem_type=3 (TSP): 3\n";

    cout << "\n";
    cout << "\t\t--save-frontier: save Pareto frontier to <input_stem>.frontier.csv.gz\n";
    cout << "\t\t--frontier-out <path>: save Pareto frontier to explicit gzip CSV path\n";
    cout << "\t\t--save-stats: save one JSONL record with run statistics\n";
    cout << "\t\t--stats-out <path>: save JSONL statistics to explicit path\n";
    cout << "\t\toptional arguments can be provided in any order\n";

    cout << endl;
}

bool parse_cli_args(int argc, char *argv[], CliOptions *out, string *error)
{
    if (out == NULL)
    {
        if (error != NULL)
        {
            *error = "internal parser error";
        }
        return false;
    }

    if (argc < 5)
    {
        if (error != NULL)
        {
            *error = "";
        }
        return false;
    }

    CliOptions opts;
    opts.input_path = argv[1];
    opts.problem_type = atoi(argv[2]);
    opts.method = atoi(argv[3]);
    opts.state_dominance = atoi(argv[4]);

    if (opts.problem_type < 1 || opts.problem_type > 3)
    {
        if (error != NULL)
        {
            *error = "Error - invalid problem type '" + string(argv[2]) + "'. Valid problem types are 1 (knapsack), 2 (set packing), 3 (TSP).";
        }
        return false;
    }

    bool backend_set = false;
    bool backend_from_named = false;
    bool backend_from_shorthand = false;
    bool kernel_version_set = false;
    bool cpu_threads_set = false;
    bool cpu_kernel_set = false;

    for (int i = 5; i < argc; ++i)
    {
        string token(argv[i]);
        if (token == "--backend")
        {
            if (backend_from_shorthand)
            {
                if (error != NULL)
                {
                    *error = "Error - cannot mix --backend with shorthand backend token.";
                }
                return false;
            }
            if (backend_set)
            {
                if (error != NULL)
                {
                    *error = "Error - backend provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc)
            {
                if (error != NULL)
                {
                    *error = "Error - --backend requires a value (cpu or gpu).";
                }
                return false;
            }
            string backend_token(argv[++i]);
            if (backend_token == "cpu")
            {
                opts.backend = BACKEND_CPU;
            }
            else if (backend_token == "gpu")
            {
                opts.backend = BACKEND_GPU;
            }
            else if (backend_token == "cuda")
            {
                if (error != NULL)
                {
                    *error = "Error - backend token 'cuda' is unsupported; use 'gpu'.";
                }
                return false;
            }
            else
            {
                if (error != NULL)
                {
                    *error = "Error - invalid backend '" + backend_token + "'. Use cpu or gpu.";
                }
                return false;
            }
            backend_set = true;
            backend_from_named = true;
        }
        else if (token == "--cpu-threads")
        {
            if (cpu_threads_set)
            {
                if (error != NULL)
                {
                    *error = "Error - cpu thread count provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc)
            {
                if (error != NULL)
                {
                    *error = "Error - --cpu-threads requires a positive integer.";
                }
                return false;
            }
            string value(argv[++i]);
            if (!parse_positive_int(value, &opts.cpu_threads))
            {
                if (error != NULL)
                {
                    *error = "Error - invalid --cpu-threads value '" + value + "' (expected positive integer).";
                }
                return false;
            }
            cpu_threads_set = true;
        }
        else if (token == "--kernel")
        {
            if (kernel_version_set)
            {
                if (error != NULL)
                {
                    *error = "Error - kernel provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc)
            {
                if (error != NULL)
                {
                    *error = "Error - --kernel requires a value in {1,2,3}.";
                }
                return false;
            }
            string value(argv[++i]);
            if (!parse_positive_int(value, &opts.kernel_version) || opts.kernel_version < 1 || opts.kernel_version > 3)
            {
                if (error != NULL)
                {
                    *error = "Error - invalid --kernel value '" + value + "' (expected 1, 2, or 3).";
                }
                return false;
            }
            kernel_version_set = true;
        }
        else if (token == "--cpu-kernel")
        {
            if (cpu_kernel_set)
            {
                if (error != NULL)
                {
                    *error = "Error - --cpu-kernel provided multiple times.";
                }
                return false;
            }
            if (i + 1 >= argc)
            {
                if (error != NULL)
                {
                    *error = "Error - --cpu-kernel requires a value in {1,3}.";
                }
                return false;
            }
            string value(argv[++i]);
            int parsed_kernel = 0;
            if (!parse_positive_int(value, &parsed_kernel) ||
                (parsed_kernel != CPU_KERNEL_1 && parsed_kernel != CPU_KERNEL_3))
            {
                if (error != NULL)
                {
                    *error = "Error - invalid --cpu-kernel value '" + value + "' (expected 1 or 3).";
                }
                return false;
            }
            opts.cpu_kernel = parsed_kernel;
            cpu_kernel_set = true;
        }
        else if (token == "cpu" || token == "gpu")
        {
            if (backend_from_named)
            {
                if (error != NULL)
                {
                    *error = "Error - cannot mix shorthand backend token with --backend.";
                }
                return false;
            }
            if (backend_set)
            {
                if (error != NULL)
                {
                    *error = "Error - backend provided multiple times.";
                }
                return false;
            }

            opts.backend = (token == "cpu" ? BACKEND_CPU : BACKEND_GPU);
            backend_set = true;
            backend_from_shorthand = true;

            if (i + 1 < argc)
            {
                string next_token(argv[i + 1]);
                errno = 0;
                char *end_ptr = NULL;
                long parsed_raw = strtol(next_token.c_str(), &end_ptr, 10);
                const bool next_is_integer = (errno == 0 && end_ptr != next_token.c_str() && *end_ptr == '\0');
                if (next_is_integer)
                {
                    if (token == "cpu")
                    {
                        if (cpu_threads_set)
                        {
                            if (error != NULL)
                            {
                                *error = "Error - cpu threads provided multiple times.";
                            }
                            return false;
                        }
                        if (parsed_raw <= 0 || parsed_raw > INT_MAX)
                        {
                            if (error != NULL)
                            {
                                *error = "Error - invalid cpu shorthand thread count '" + next_token + "' (expected positive integer).";
                            }
                            return false;
                        }
                        opts.cpu_threads = static_cast<int>(parsed_raw);
                        cpu_threads_set = true;
                    }
                    else
                    {
                        if (kernel_version_set)
                        {
                            if (error != NULL)
                            {
                                *error = "Error - kernel provided multiple times.";
                            }
                            return false;
                        }
                        if (parsed_raw < 1 || parsed_raw > 3)
                        {
                            if (error != NULL)
                            {
                                *error = "Error - invalid gpu shorthand kernel '" + next_token + "' (expected 1, 2, or 3).";
                            }
                            return false;
                        }
                        opts.kernel_version = static_cast<int>(parsed_raw);
                        kernel_version_set = true;
                    }
                    ++i;
                }
            }
        }
        else if (token == "cuda")
        {
            if (error != NULL)
            {
                *error = "Error - backend token 'cuda' is unsupported; use 'gpu'.";
            }
            return false;
        }
        else if (token == "--save-frontier")
        {
            opts.save_frontier = true;
        }
        else if (token == "--frontier-out")
        {
            if (i + 1 >= argc)
            {
                if (error != NULL)
                {
                    *error = "Error - --frontier-out requires a file path.";
                }
                return false;
            }
            opts.frontier_out_path = argv[++i];
            if (opts.frontier_out_path.empty())
            {
                if (error != NULL)
                {
                    *error = "Error - --frontier-out path cannot be empty.";
                }
                return false;
            }
            opts.save_frontier = true;
        }
        else if (token == "--save-stats")
        {
            opts.save_stats = true;
        }
        else if (token == "--stats-out")
        {
            if (i + 1 >= argc)
            {
                if (error != NULL)
                {
                    *error = "Error - --stats-out requires a file path.";
                }
                return false;
            }
            opts.stats_out_path = argv[++i];
            if (opts.stats_out_path.empty())
            {
                if (error != NULL)
                {
                    *error = "Error - --stats-out path cannot be empty.";
                }
                return false;
            }
            opts.save_stats = true;
        }
        else
        {
            int parsed_numeric = 0;
            if (parse_positive_int(token, &parsed_numeric))
            {
                if (error != NULL)
                {
                    *error = "Error - positional numeric argument '" + token + "' must immediately follow shorthand backend token cpu|gpu.";
                }
            }
            else
            {
                if (error != NULL)
                {
                    *error = "Error - unrecognized optional argument '" + token + "'.";
                }
            }
            return false;
        }
    }

    if (opts.save_frontier && opts.frontier_out_path.empty())
    {
        opts.frontier_out_path = derive_default_frontier_path(opts.input_path);
    }
    if (opts.save_stats && opts.stats_out_path.empty())
    {
        opts.stats_out_path = derive_default_stats_path(opts.input_path);
    }

    if (opts.backend == BACKEND_GPU && cpu_threads_set)
    {
        if (error != NULL)
        {
            *error = "Error - cpu thread options are not valid with backend=gpu.";
        }
        return false;
    }
    if (opts.backend == BACKEND_CPU && kernel_version_set)
    {
        if (error != NULL)
        {
            *error = "Error - --kernel is only valid with backend=gpu.";
        }
        return false;
    }
    if (opts.backend == BACKEND_GPU && cpu_kernel_set)
    {
        if (error != NULL)
        {
            *error = "Error - --cpu-kernel is only valid with backend=cpu.";
        }
        return false;
    }
    if (cpu_kernel_set && opts.method != 1 && opts.method != 3)
    {
        if (error != NULL)
        {
            *error = "Error - --cpu-kernel is only supported for method 1 (top-down BFS) and method 3 (dynamic layer cutset).";
        }
        return false;
    }
    if (opts.backend == BACKEND_GPU && !kernel_version_set)
    {
        if (opts.problem_type == 1)
        {
            opts.kernel_version = 1;
        }
        else if (opts.problem_type == 2)
        {
            opts.kernel_version = 2;
        }
        else if (opts.problem_type == 3)
        {
            opts.kernel_version = 3;
        }
    }
    else if (opts.backend == BACKEND_CPU)
    {
#if CUMODD_HAS_OPENMP
        if (!cpu_threads_set)
        {
            const char *env_threads = getenv("OMP_NUM_THREADS");
            int parsed_env_threads = 0;
            if (env_threads != NULL && parse_positive_int(string(env_threads), &parsed_env_threads))
            {
                opts.cpu_threads = parsed_env_threads;
            }
            else
            {
                opts.cpu_threads = 1;
            }
        }
#else
        if (cpu_threads_set)
        {
            if (error != NULL)
            {
                *error = "Error - cpu thread options are unsupported in this build (OpenMP disabled). Rebuild with ENABLE_OPENMP=1.";
            }
            return false;
        }
        opts.cpu_threads = 1;
#endif
    }

    if (opts.backend == BACKEND_GPU && opts.method != 1 && opts.method != 3)
    {
        if (error != NULL)
        {
            *error = "Error - GPU backend is unsupported for method " + to_string(opts.method) + ".";
        }
        return false;
    }

    *out = opts;
    return true;
}
