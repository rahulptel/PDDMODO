// ----------------------------------------------------------
// CUDA Coupled (Dynamic Layer Cutset) Enumeration for MDD
// ----------------------------------------------------------

#ifndef COUPLED_CUDA_HPP_
#define COUPLED_CUDA_HPP_

#include <string>

#include "../mdd/mdd.hpp"
#include "../bdd/pareto_frontier.hpp"

struct EnumerationStats;
using MultiObjectiveStats = EnumerationStats;

// Checks whether at least one CUDA device is available.
bool coupled_cuda_available(std::string* reason);

// Runs coupled (dynamic layer cutset) frontier enumeration on CUDA for MDDs.
// Returns NULL on failure and fills reason when provided.
// kernel_version:
//   1 = one block per node
//   2 = fixed number of blocks per node (2D grid)
//   3 = dynamic number of blocks per node (1D grid + binary-search destination lookup)
ParetoFrontier* coupled_cuda_enumerate(MDD* mdd,
                                       EnumerationStats* stats,
                                       std::string* reason,
                                       int kernel_version = 3);

#endif
