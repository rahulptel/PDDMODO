// ----------------------------------------------------------
// CUDA Top-Down Enumeration for BDD
// ----------------------------------------------------------

#ifndef TOPDOWN_CUDA_HPP_
#define TOPDOWN_CUDA_HPP_

#include <string>

#include "../bdd/bdd.hpp"
#include "../mdd/mdd.hpp"
#include "../bdd/pareto_frontier.hpp"
#include <thrust/device_vector.h>

struct MultiObjectiveStats;

// Checks whether at least one CUDA device is available.
bool topdown_cuda_available(std::string* reason);

// Runs top-down frontier enumeration on CUDA for BDDs.
// Returns NULL on failure and fills reason when provided.
ParetoFrontier* topdown_cuda_enumerate(BDD* bdd,
                                       bool maximization,
                                       const int problem_type,
                                       const int dominance_strategy,
                                       MultiObjectiveStats* stats,
                                       std::string* reason);

// Runs top-down frontier enumeration on CUDA for MDDs.
// Returns NULL on failure and fills reason when provided.
ParetoFrontier* topdown_mdd_cuda_enumerate(MDD* mdd,
                                           MultiObjectiveStats* stats,
                                           std::string* reason);

// ---------------------------------------------------------------
// Shared MDD CUDA definitions used by both top-down and coupled approaches
// ---------------------------------------------------------------

// Flat representation of one MDD layer's connectivity, on device.
struct PackedMDDLayer {
    int num_nodes;

    // Top-down: incoming arcs grouped by destination node (this layer).
    // edge_src indices refer to nodes in layer-1.
    int td_num_edges;
    thrust::device_vector<int> td_in_edge_offsets;
    thrust::device_vector<int> td_edge_src;
    thrust::device_vector<ObjType> td_edge_weights;

    // Bottom-up: outgoing arcs grouped by node (this layer).
    // edge_src indices refer to head nodes in layer+1.
    int bu_num_edges;
    thrust::device_vector<int> bu_in_edge_offsets;
    thrust::device_vector<int> bu_edge_src;
    thrust::device_vector<ObjType> bu_edge_weights;

    // Per-node arc counts for the layer-value heuristic.
    thrust::device_vector<int> out_arc_counts;
    thrust::device_vector<int> in_arc_counts;
};

// Computes the node counts for the heuristic.
int compute_layer_value(const thrust::device_vector<int>& offsets,
                        const thrust::device_vector<int>& arc_counts,
                        int num_nodes);

// Expands a layer in CUDA (either top-down or bottom-up).
bool expand_layer_cuda(
    const thrust::device_vector<int>& in_edge_offsets,
    const thrust::device_vector<int>& edge_src,
    const thrust::device_vector<ObjType>& edge_weights,
    int num_edges,
    int next_nodes,
    const thrust::device_vector<int>& d_prev_offsets,
    const thrust::device_vector<ObjType>& d_prev_points,
    thrust::device_vector<int>& d_next_sizes,
    thrust::device_vector<int>& d_next_offsets,
    thrust::device_vector<ObjType>& d_next_points,
    std::string* reason);

#endif
