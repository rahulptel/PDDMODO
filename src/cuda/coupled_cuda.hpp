// ----------------------------------------------------------
// CUDA Coupled (Dynamic Layer Cutset) Enumeration for MDD
// ----------------------------------------------------------

#ifndef COUPLED_CUDA_HPP_
#define COUPLED_CUDA_HPP_

#include <string>

#include "../mdd/mdd.hpp"
#include "../bdd/pareto_frontier.hpp"

struct MultiObjectiveStats;

// Checks whether at least one CUDA device is available.
bool coupled_cuda_available(std::string* reason);

// Runs coupled (dynamic layer cutset) frontier enumeration on CUDA for MDDs.
// Returns NULL on failure and fills reason when provided.
ParetoFrontier* coupled_cuda_enumerate(MDD* mdd,
                                       MultiObjectiveStats* stats,
                                       std::string* reason);

#endif
