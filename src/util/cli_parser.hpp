// --------------------------------------------------
// CLI parser
// --------------------------------------------------

#ifndef CLI_PARSER_HPP_
#define CLI_PARSER_HPP_

#include <string>

enum Backend
{
    BACKEND_CPU = 0,
    BACKEND_GPU = 1
};

struct CliOptions
{
    std::string input_path;
    int problem_type;
    int method;
    int state_dominance;
    Backend backend;
    int cpu_threads;
    bool save_frontier;
    std::string frontier_out_path;
    bool save_stats;
    std::string stats_out_path;

    CliOptions();
};

const char *backend_to_string(Backend backend);
void print_usage();
bool parse_cli_args(int argc, char *argv[], CliOptions *out, std::string *error);

#endif
