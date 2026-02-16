// ----------------------------------------------------------
// CUDA Dynamic-Cutset Coupled Enumeration for BDD
// ----------------------------------------------------------

#ifndef COUPLED_CUDA_HPP_
#define COUPLED_CUDA_HPP_

#include <string>

#include "../bdd/bdd.hpp"
#include "../bdd/pareto_frontier.hpp"

struct MultiObjectiveStats;

// Checks whether at least one CUDA device is available.
bool coupled_cuda_available(std::string* reason);

// Runs dynamic-cutset coupled frontier enumeration on CUDA for BDDs.
// Returns NULL on failure and fills reason when provided.
ParetoFrontier* coupled_cuda_enumerate(BDD* bdd,
                                       bool maximization,
                                       const int problem_type,
                                       const int dominance_strategy,
                                       MultiObjectiveStats* stats,
                                       std::string* reason);

#endif
