// ------------------------------------------------------------------
// CUDA Dynamic-Cutset Coupled Enumeration for BDD
// ------------------------------------------------------------------

#include "coupled_cuda.hpp"

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "../bdd/bdd_multiobj.hpp"

#include <thrust/copy.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/host_vector.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/scan.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>

namespace {

constexpr int kThreadsPerBlock = 128;

struct DeviceLayerFrontier {
    int num_nodes;
    thrust::device_vector<int> sizes;
    thrust::device_vector<int> offsets;
    thrust::device_vector<ObjType> points;

    DeviceLayerFrontier() : num_nodes(0) {}
};

inline bool set_reason(std::string* reason, const std::string& message) {
    if (reason != NULL) {
        *reason = message;
    }
    return false;
}

inline bool cuda_ok(cudaError_t err, const char* where, std::string* reason) {
    if (err == cudaSuccess) {
        return true;
    }
    std::string msg = where;
    msg += ": ";
    msg += cudaGetErrorString(err);
    return set_reason(reason, msg);
}

inline bool sync_kernel(const char* where, std::string* reason) {
    if (!cuda_ok(cudaGetLastError(), where, reason)) {
        return false;
    }
    return cuda_ok(cudaDeviceSynchronize(), where, reason);
}

inline bool launch_ok(const char* where, std::string* reason) {
    return cuda_ok(cudaGetLastError(), where, reason);
}

inline bool sync_stream0(const char* where, std::string* reason) {
    return cuda_ok(cudaStreamSynchronize(0), where, reason);
}

inline int ceil_div(const int a, const int b) {
    return (a + b - 1) / b;
}

__global__ void compute_edge_counts_from_sizes_kernel(const int* edge_src,
                                                      const int* src_sizes,
                                                      int* edge_counts,
                                                      int num_edges) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) {
        return;
    }
    edge_counts[e] = src_sizes[edge_src[e]];
}

__global__ void expand_candidates_kernel(const int* edge_src,
                                         const int* edge_dst,
                                         const ObjType* edge_weights,
                                         const int* edge_offsets,
                                         const int* edge_counts,
                                         const int* src_offsets,
                                         const ObjType* src_points,
                                         int num_edges,
                                         int* cand_dst,
                                         ObjType* cand_points) {
    const int e = blockIdx.x * blockDim.x + threadIdx.x;
    if (e >= num_edges) {
        return;
    }

    const int src = edge_src[e];
    const int dst = edge_dst[e];
    const int src_begin = src_offsets[src];
    const int out_begin = edge_offsets[e];
    const int count = edge_counts[e];
    const ObjType* w = edge_weights + (e * NOBJS);

    for (int k = 0; k < count; ++k) {
        const int src_idx = src_begin + k;
        const int out_idx = out_begin + k;

        cand_dst[out_idx] = dst;
        for (int o = 0; o < NOBJS; ++o) {
            cand_points[out_idx * NOBJS + o] = src_points[src_idx * NOBJS + o] + w[o];
        }
    }
}

__global__ void gather_points_kernel(const int* order,
                                     const ObjType* in_points,
                                     ObjType* out_points,
                                     int num_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points) {
        return;
    }

    const int src = order[i];
    for (int o = 0; o < NOBJS; ++o) {
        out_points[i * NOBJS + o] = in_points[src * NOBJS + o];
    }
}

__global__ void mark_dominated_segments_kernel(const ObjType* points,
                                               const int* seg_offsets,
                                               const int* seg_counts,
                                               int num_segments,
                                               int* alive) {
    const int seg = blockIdx.x;
    if (seg >= num_segments) {
        return;
    }

    const int begin = seg_offsets[seg];
    const int len = seg_counts[seg];
    if (len <= 0) {
        return;
    }

    extern __shared__ ObjType sh_points[];

    for (int i_base = 0; i_base < len; i_base += blockDim.x) {
        const int local_i = i_base + threadIdx.x;
        const bool active_i = (local_i < len);
        ObjType point_i[NOBJS];
        if (active_i) {
#pragma unroll
            for (int o = 0; o < NOBJS; ++o) {
                point_i[o] = points[(begin + local_i) * NOBJS + o];
            }
        }

        bool dominated = false;

        for (int tile_begin = 0; tile_begin < len; tile_begin += blockDim.x) {
            const int tile_count = min(blockDim.x, len - tile_begin);
            if (threadIdx.x < tile_count) {
                const int src_idx = begin + tile_begin + threadIdx.x;
#pragma unroll
                for (int o = 0; o < NOBJS; ++o) {
                    sh_points[threadIdx.x * NOBJS + o] = points[src_idx * NOBJS + o];
                }
            }
            __syncthreads();

            if (active_i && !dominated) {
                for (int t = 0; t < tile_count && !dominated; ++t) {
                    const int local_j = tile_begin + t;
                    if (local_j == local_i) {
                        continue;
                    }

                    const ObjType a0 = sh_points[t * NOBJS];
                    if (a0 < point_i[0]) {
                        continue;
                    }

                    bool ge_all = true;
                    bool strict = (a0 > point_i[0]);
#pragma unroll
                    for (int o = 1; o < NOBJS; ++o) {
                        const ObjType a = sh_points[t * NOBJS + o];
                        const ObjType b = point_i[o];
                        ge_all = ge_all && (a >= b);
                        strict = strict || (a > b);
                    }

                    if (ge_all && (strict || (local_j < local_i))) {
                        dominated = true;
                    }
                }
            }
            __syncthreads();
        }

        if (active_i) {
            alive[begin + local_i] = dominated ? 0 : 1;
        }
    }
}

__global__ void write_segment_sizes_kernel(const int* unique_dst,
                                           const int* seg_offsets,
                                           const int* seg_counts,
                                           const int* alive,
                                           int num_segments,
                                           int* next_sizes) {
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= num_segments) {
        return;
    }

    const int begin = seg_offsets[seg];
    const int len = seg_counts[seg];
    int live = 0;
    for (int i = 0; i < len; ++i) {
        live += alive[begin + i];
    }
    next_sizes[unique_dst[seg]] = live;
}

__global__ void write_segment_live_counts_kernel(const int* seg_offsets,
                                                 const int* seg_counts,
                                                 const int* alive,
                                                 int num_segments,
                                                 int* out_counts) {
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= num_segments) {
        return;
    }

    const int begin = seg_offsets[seg];
    const int len = seg_counts[seg];
    int live = 0;
    for (int i = 0; i < len; ++i) {
        live += alive[begin + i];
    }
    out_counts[seg] = live;
}

__global__ void scatter_node_live_counts_kernel(const int* node_ids,
                                                const int* live_counts,
                                                int num_segments,
                                                int* node_sizes) {
    const int seg = blockIdx.x * blockDim.x + threadIdx.x;
    if (seg >= num_segments) {
        return;
    }
    node_sizes[node_ids[seg]] = live_counts[seg];
}

__global__ void scatter_alive_points_kernel(const int* alive,
                                            const int* alive_prefix,
                                            const ObjType* in_points,
                                            ObjType* out_points,
                                            int num_points) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_points || !alive[i]) {
        return;
    }

    const int out_idx = alive_prefix[i];
    for (int o = 0; o < NOBJS; ++o) {
        out_points[out_idx * NOBJS + o] = in_points[i * NOBJS + o];
    }
}

__global__ void build_coupling_candidates_kernel(const int* node_ids,
                                                 const int* top_offsets,
                                                 const int* top_sizes,
                                                 const ObjType* top_points,
                                                 const int* bottom_offsets,
                                                 const int* bottom_sizes,
                                                 const ObjType* bottom_points,
                                                 const int* pair_offsets,
                                                 int num_batch_nodes,
                                                 ObjType* out_points) {
    const int batch_idx = blockIdx.x;
    if (batch_idx >= num_batch_nodes) {
        return;
    }

    const int node = node_ids[batch_idx];
    const int t_size = top_sizes[node];
    const int b_size = bottom_sizes[node];
    const int t_begin = top_offsets[node];
    const int b_begin = bottom_offsets[node];
    const int out_begin = pair_offsets[batch_idx];
    const int total = t_size * b_size;

    for (int local = threadIdx.x; local < total; local += blockDim.x) {
        const int t_local = local / b_size;
        const int b_local = local % b_size;
        const int out_idx = out_begin + local;

        const int t_idx = t_begin + t_local;
        const int b_idx = b_begin + b_local;
        for (int o = 0; o < NOBJS; ++o) {
            out_points[out_idx * NOBJS + o] = top_points[t_idx * NOBJS + o] + bottom_points[b_idx * NOBJS + o];
        }
    }
}

__global__ void compute_union_pair_counts_kernel(const int* sizes,
                                                 int num_nodes,
                                                 int* pair_counts) {
    const int pair = blockIdx.x * blockDim.x + threadIdx.x;
    const int num_pairs = (num_nodes + 1) / 2;
    if (pair >= num_pairs) {
        return;
    }

    const int left = 2 * pair;
    const int right = left + 1;
    int count = sizes[left];
    if (right < num_nodes) {
        count += sizes[right];
    }
    pair_counts[pair] = count;
}

__global__ void build_union_candidates_kernel(const int* offsets,
                                              const int* sizes,
                                              int num_nodes,
                                              const int* pair_offsets,
                                              int num_pairs,
                                              const ObjType* in_points,
                                              ObjType* out_points) {
    const int pair = blockIdx.x;
    if (pair >= num_pairs) {
        return;
    }

    const int left = 2 * pair;
    const int right = left + 1;

    const int left_size = sizes[left];
    const int right_size = (right < num_nodes ? sizes[right] : 0);
    const int left_begin = offsets[left];
    const int right_begin = (right < num_nodes ? offsets[right] : 0);
    const int out_begin = pair_offsets[pair];
    const int total = left_size + right_size;

    for (int local = threadIdx.x; local < total; local += blockDim.x) {
        int src_idx = 0;
        if (local < left_size) {
            src_idx = left_begin + local;
        } else {
            src_idx = right_begin + (local - left_size);
        }

        const int out_idx = out_begin + local;
        for (int o = 0; o < NOBJS; ++o) {
            out_points[out_idx * NOBJS + o] = in_points[src_idx * NOBJS + o];
        }
    }
}

inline bool compute_offsets_from_sizes(thrust::device_vector<int>* sizes,
                                       thrust::device_vector<int>* offsets,
                                       int* total,
                                       std::string* reason) {
    if (sizes == NULL || offsets == NULL || total == NULL) {
        return set_reason(reason, "Null pointer in compute_offsets_from_sizes");
    }

    const int n = static_cast<int>(sizes->size());
    offsets->assign(n + 1, 0);
    if (n > 0) {
        thrust::exclusive_scan(sizes->begin(), sizes->end(), offsets->begin());
        *total = thrust::reduce(sizes->begin(), sizes->end(), 0);
        (*offsets)[n] = *total;
    } else {
        *total = 0;
        (*offsets)[0] = 0;
    }
    return true;
}

inline void build_topdown_edges(BDD* bdd,
                                int layer,
                                bool maximization,
                                std::vector<int>* edge_src,
                                std::vector<int>* edge_dst,
                                std::vector<ObjType>* edge_weights) {
    edge_src->clear();
    edge_dst->clear();
    edge_weights->clear();

    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;

    const int next_nodes = static_cast<int>(bdd->layers[layer].size());
    for (int dst_idx = 0; dst_idx < next_nodes; ++dst_idx) {
        Node* dst_node = bdd->layers[layer][dst_idx];
        const int arc_order[2] = {first_arc_type, second_arc_type};

        for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
            const int arc_type = arc_order[arc_pos];
            for (std::vector<Node*>::iterator it = dst_node->prev[arc_type].begin();
                 it != dst_node->prev[arc_type].end(); ++it)
            {
                Node* src_node = *it;
                edge_src->push_back(src_node->index);
                edge_dst->push_back(dst_idx);

                ObjType* w = src_node->weights[arc_type];
                for (int o = 0; o < NOBJS; ++o) {
                    edge_weights->push_back(w != NULL ? w[o] : 0);
                }
            }
        }
    }
}

inline void build_bottomup_edges(BDD* bdd,
                                 int layer,
                                 bool maximization,
                                 std::vector<int>* edge_src,
                                 std::vector<int>* edge_dst,
                                 std::vector<ObjType>* edge_weights) {
    edge_src->clear();
    edge_dst->clear();
    edge_weights->clear();

    const int first_arc_type = maximization ? 1 : 0;
    const int second_arc_type = maximization ? 0 : 1;

    const int cur_nodes = static_cast<int>(bdd->layers[layer].size());
    for (int src_idx = 0; src_idx < cur_nodes; ++src_idx) {
        Node* src_node = bdd->layers[layer][src_idx];
        const int arc_order[2] = {first_arc_type, second_arc_type};

        for (int arc_pos = 0; arc_pos < 2; ++arc_pos) {
            const int arc_type = arc_order[arc_pos];
            Node* next_node = src_node->arcs[arc_type];
            if (next_node == NULL) {
                continue;
            }

            edge_src->push_back(next_node->index);
            edge_dst->push_back(src_idx);

            ObjType* w = src_node->weights[arc_type];
            for (int o = 0; o < NOBJS; ++o) {
                edge_weights->push_back(w != NULL ? w[o] : 0);
            }
        }
    }
}

inline int compute_top_layer_value(BDD* bdd,
                                   int layer,
                                   const thrust::host_vector<int>& h_sizes) {
    int val = 0;
    for (int i = 0; i < static_cast<int>(h_sizes.size()); ++i) {
        int out_deg = 0;
        if (bdd->layers[layer][i]->arcs[0] != NULL) {
            ++out_deg;
        }
        if (bdd->layers[layer][i]->arcs[1] != NULL) {
            ++out_deg;
        }
        val += h_sizes[i] * out_deg;
    }
    return val;
}

inline int compute_bottom_layer_value(BDD* bdd,
                                      int layer,
                                      const thrust::host_vector<int>& h_sizes) {
    int val = 0;
    for (int i = 0; i < static_cast<int>(h_sizes.size()); ++i) {
        const int indeg = static_cast<int>(bdd->layers[layer][i]->prev[0].size() +
                                           bdd->layers[layer][i]->prev[1].size());
        val += static_cast<int>(1.5 * h_sizes[i] * indeg);
    }
    return val;
}

inline bool expand_from_edges(const DeviceLayerFrontier& src,
                              int dst_nodes,
                              const std::vector<int>& h_edge_src,
                              const std::vector<int>& h_edge_dst,
                              const std::vector<ObjType>& h_edge_weights,
                              DeviceLayerFrontier* dst,
                              std::string* reason) {
    if (dst == NULL) {
        return set_reason(reason, "Destination frontier is NULL");
    }

    dst->num_nodes = dst_nodes;
    dst->sizes.assign(dst_nodes, 0);
    dst->offsets.assign(dst_nodes + 1, 0);
    dst->points.clear();

    const int num_edges = static_cast<int>(h_edge_src.size());
    if (num_edges <= 0 || src.points.empty()) {
        return true;
    }

    thrust::device_vector<int> d_edge_src = h_edge_src;
    thrust::device_vector<int> d_edge_dst = h_edge_dst;
    thrust::device_vector<ObjType> d_edge_weights = h_edge_weights;

    thrust::device_vector<int> d_edge_counts(num_edges, 0);
    thrust::device_vector<int> d_edge_offsets(num_edges, 0);

    compute_edge_counts_from_sizes_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_edge_src.data()),
        thrust::raw_pointer_cast(src.sizes.data()),
        thrust::raw_pointer_cast(d_edge_counts.data()),
        num_edges);
    if (!sync_kernel("compute_edge_counts_from_sizes_kernel", reason)) {
        return false;
    }

    thrust::exclusive_scan(d_edge_counts.begin(), d_edge_counts.end(), d_edge_offsets.begin());
    const int last_offset = d_edge_offsets[num_edges - 1];
    const int last_count = d_edge_counts[num_edges - 1];
    const int total_candidates = last_offset + last_count;

    if (total_candidates <= 0) {
        return true;
    }

    thrust::device_vector<int> d_cand_dst(total_candidates, 0);
    thrust::device_vector<ObjType> d_cand_points(total_candidates * NOBJS, 0);

    expand_candidates_kernel<<<ceil_div(num_edges, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_edge_src.data()),
        thrust::raw_pointer_cast(d_edge_dst.data()),
        thrust::raw_pointer_cast(d_edge_weights.data()),
        thrust::raw_pointer_cast(d_edge_offsets.data()),
        thrust::raw_pointer_cast(d_edge_counts.data()),
        thrust::raw_pointer_cast(src.offsets.data()),
        thrust::raw_pointer_cast(src.points.data()),
        num_edges,
        thrust::raw_pointer_cast(d_cand_dst.data()),
        thrust::raw_pointer_cast(d_cand_points.data()));
    if (!sync_kernel("expand_candidates_kernel", reason)) {
        return false;
    }

    thrust::device_vector<int> d_order(total_candidates, 0);
    thrust::sequence(d_order.begin(), d_order.end());
    thrust::sort_by_key(d_cand_dst.begin(), d_cand_dst.end(), d_order.begin());

    thrust::device_vector<ObjType> d_sorted_points(total_candidates * NOBJS, 0);
    gather_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_order.data()),
        thrust::raw_pointer_cast(d_cand_points.data()),
        thrust::raw_pointer_cast(d_sorted_points.data()),
        total_candidates);
    if (!sync_kernel("gather_points_kernel", reason)) {
        return false;
    }

    thrust::device_vector<int> d_unique_dst(total_candidates, 0);
    thrust::device_vector<int> d_seg_counts(total_candidates, 0);
    typedef thrust::device_vector<int>::iterator It;
    thrust::pair<It, It> seg_end = thrust::reduce_by_key(
        d_cand_dst.begin(),
        d_cand_dst.end(),
        thrust::make_constant_iterator(1),
        d_unique_dst.begin(),
        d_seg_counts.begin());

    const int num_segments = static_cast<int>(seg_end.first - d_unique_dst.begin());
    d_unique_dst.resize(num_segments);
    d_seg_counts.resize(num_segments);

    if (num_segments <= 0) {
        return true;
    }

    thrust::device_vector<int> d_seg_offsets(num_segments + 1, 0);
    thrust::exclusive_scan(d_seg_counts.begin(), d_seg_counts.end(), d_seg_offsets.begin());
    d_seg_offsets[num_segments] = total_candidates;

    thrust::device_vector<int> d_alive(total_candidates, 0);
    mark_dominated_segments_kernel<<<num_segments, kThreadsPerBlock, static_cast<size_t>(kThreadsPerBlock) * NOBJS * sizeof(ObjType)>>>(
        thrust::raw_pointer_cast(d_sorted_points.data()),
        thrust::raw_pointer_cast(d_seg_offsets.data()),
        thrust::raw_pointer_cast(d_seg_counts.data()),
        num_segments,
        thrust::raw_pointer_cast(d_alive.data()));
    if (!sync_kernel("mark_dominated_segments_kernel", reason)) {
        return false;
    }

    thrust::device_vector<int> d_alive_prefix(total_candidates, 0);
    thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
    const int total_next = thrust::reduce(d_alive.begin(), d_alive.end(), 0);

    write_segment_sizes_kernel<<<ceil_div(num_segments, kThreadsPerBlock), kThreadsPerBlock>>>(
        thrust::raw_pointer_cast(d_unique_dst.data()),
        thrust::raw_pointer_cast(d_seg_offsets.data()),
        thrust::raw_pointer_cast(d_seg_counts.data()),
        thrust::raw_pointer_cast(d_alive.data()),
        num_segments,
        thrust::raw_pointer_cast(dst->sizes.data()));
    if (!sync_kernel("write_segment_sizes_kernel", reason)) {
        return false;
    }

    thrust::exclusive_scan(dst->sizes.begin(), dst->sizes.end(), dst->offsets.begin());
    dst->offsets[dst_nodes] = total_next;

    dst->points.resize(total_next * NOBJS);
    if (total_next > 0) {
        scatter_alive_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_alive.data()),
            thrust::raw_pointer_cast(d_alive_prefix.data()),
            thrust::raw_pointer_cast(d_sorted_points.data()),
            thrust::raw_pointer_cast(dst->points.data()),
            total_candidates);
        if (!sync_kernel("scatter_alive_points_kernel", reason)) {
            return false;
        }
    }

    return true;
}

inline bool determine_max_pairs(std::string* reason, size_t* out_max_pairs) {
    if (out_max_pairs == NULL) {
        return set_reason(reason, "Output pointer is NULL in determine_max_pairs");
    }

    const char* env_pairs = std::getenv("MULTIOBJ_GPU_COUPLE_MAX_PAIRS");
    if (env_pairs != NULL && env_pairs[0] != '\0') {
        char* end_ptr = NULL;
        long long parsed = std::strtoll(env_pairs, &end_ptr, 10);
        if (end_ptr == env_pairs || parsed <= 0) {
            return set_reason(reason, "Invalid MULTIOBJ_GPU_COUPLE_MAX_PAIRS value");
        }
        const long long max_int = static_cast<long long>(std::numeric_limits<int>::max());
        if (parsed > max_int) {
            parsed = max_int;
        }
        *out_max_pairs = static_cast<size_t>(parsed);
        return true;
    }

    size_t free_mem = 0;
    size_t total_mem = 0;
    if (!cuda_ok(cudaMemGetInfo(&free_mem, &total_mem), "cudaMemGetInfo", reason)) {
        return false;
    }

    if (free_mem == 0) {
        return set_reason(reason, "cudaMemGetInfo reported zero free memory");
    }

    const size_t safety_free = (free_mem * 7) / 10;
    const size_t bytes_per_pair = sizeof(ObjType) * NOBJS + sizeof(int) * 4;
    if (bytes_per_pair == 0) {
        return set_reason(reason, "Invalid bytes-per-pair estimate");
    }

    size_t max_pairs = safety_free / bytes_per_pair;
    if (max_pairs == 0) {
        return set_reason(reason, "Not enough GPU memory for any coupling pair");
    }

    const size_t max_int = static_cast<size_t>(std::numeric_limits<int>::max());
    if (max_pairs > max_int) {
        max_pairs = max_int;
    }

    *out_max_pairs = max_pairs;
    return true;
}

inline bool couple_layer_batched(const DeviceLayerFrontier& top,
                                 const DeviceLayerFrontier& bottom,
                                 size_t max_pairs_per_batch,
                                 DeviceLayerFrontier* coupled,
                                 std::string* reason) {
    if (coupled == NULL) {
        return set_reason(reason, "Coupled frontier is NULL");
    }
    if (top.num_nodes != bottom.num_nodes) {
        return set_reason(reason, "Top and bottom layer sizes differ at coupling");
    }

    const int num_nodes = top.num_nodes;
    coupled->num_nodes = num_nodes;
    coupled->sizes.assign(num_nodes, 0);

    thrust::host_vector<int> h_top_sizes = top.sizes;
    thrust::host_vector<int> h_bottom_sizes = bottom.sizes;

    thrust::device_vector<ObjType> d_coupled_points;
    size_t coupled_capacity_elems = 0;
    size_t pair_capacity = 0;
    int batch_node_capacity = 0;

    thrust::device_vector<int> d_node_ids;
    thrust::device_vector<int> d_pair_counts;
    thrust::device_vector<int> d_pair_offsets;
    thrust::device_vector<int> d_alive;
    thrust::device_vector<int> d_alive_prefix;
    thrust::device_vector<int> d_live_counts;
    thrust::device_vector<ObjType> d_cand_points;
    thrust::device_vector<ObjType> d_live_points;

    std::vector<int> batch_nodes;
    std::vector<int> batch_pair_counts;
    std::vector<int> h_pair_offsets;

    int node = 0;
    int running_total = 0;

    while (node < num_nodes) {
        batch_nodes.clear();
        batch_pair_counts.clear();
        long long batch_total_pairs = 0;

        while (node < num_nodes) {
            const long long node_pairs = static_cast<long long>(h_top_sizes[node]) *
                                         static_cast<long long>(h_bottom_sizes[node]);
            if (node_pairs == 0) {
                ++node;
                continue;
            }

            if (node_pairs > static_cast<long long>(max_pairs_per_batch)) {
                return set_reason(reason, "Single coupling node exceeds MULTIOBJ_GPU_COUPLE_MAX_PAIRS capacity");
            }

            if (!batch_nodes.empty() && (batch_total_pairs + node_pairs > static_cast<long long>(max_pairs_per_batch))) {
                break;
            }

            batch_nodes.push_back(node);
            batch_pair_counts.push_back(static_cast<int>(node_pairs));
            batch_total_pairs += node_pairs;
            ++node;
        }

        if (batch_nodes.empty()) {
            continue;
        }

        const int num_batch_nodes = static_cast<int>(batch_nodes.size());
        const int total_pairs = static_cast<int>(batch_total_pairs);

        h_pair_offsets.assign(num_batch_nodes + 1, 0);
        for (int i = 0; i < num_batch_nodes; ++i) {
            h_pair_offsets[i + 1] = h_pair_offsets[i] + batch_pair_counts[i];
        }

        if (num_batch_nodes > batch_node_capacity) {
            d_node_ids.reserve(num_batch_nodes);
            d_pair_counts.reserve(num_batch_nodes);
            d_pair_offsets.reserve(num_batch_nodes + 1);
            d_live_counts.reserve(num_batch_nodes);
            batch_node_capacity = num_batch_nodes;
        }
        d_node_ids.resize(num_batch_nodes);
        d_pair_counts.resize(num_batch_nodes);
        d_pair_offsets.resize(num_batch_nodes + 1);
        d_live_counts.resize(num_batch_nodes);
        thrust::copy(batch_nodes.begin(), batch_nodes.end(), d_node_ids.begin());
        thrust::copy(batch_pair_counts.begin(), batch_pair_counts.end(), d_pair_counts.begin());
        thrust::copy(h_pair_offsets.begin(), h_pair_offsets.end(), d_pair_offsets.begin());

        if (static_cast<size_t>(total_pairs) > pair_capacity) {
            d_cand_points.reserve(static_cast<size_t>(total_pairs) * NOBJS);
            d_alive.reserve(total_pairs);
            d_alive_prefix.reserve(total_pairs);
            d_live_points.reserve(static_cast<size_t>(total_pairs) * NOBJS);
            pair_capacity = static_cast<size_t>(total_pairs);
        }
        d_cand_points.resize(static_cast<size_t>(total_pairs) * NOBJS);
        d_alive.resize(total_pairs);
        d_alive_prefix.resize(total_pairs);

        build_coupling_candidates_kernel<<<num_batch_nodes, kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_node_ids.data()),
            thrust::raw_pointer_cast(top.offsets.data()),
            thrust::raw_pointer_cast(top.sizes.data()),
            thrust::raw_pointer_cast(top.points.data()),
            thrust::raw_pointer_cast(bottom.offsets.data()),
            thrust::raw_pointer_cast(bottom.sizes.data()),
            thrust::raw_pointer_cast(bottom.points.data()),
            thrust::raw_pointer_cast(d_pair_offsets.data()),
            num_batch_nodes,
            thrust::raw_pointer_cast(d_cand_points.data()));
        if (!launch_ok("build_coupling_candidates_kernel", reason)) {
            return false;
        }

        mark_dominated_segments_kernel<<<num_batch_nodes, kThreadsPerBlock, static_cast<size_t>(kThreadsPerBlock) * NOBJS * sizeof(ObjType)>>>(
            thrust::raw_pointer_cast(d_cand_points.data()),
            thrust::raw_pointer_cast(d_pair_offsets.data()),
            thrust::raw_pointer_cast(d_pair_counts.data()),
            num_batch_nodes,
            thrust::raw_pointer_cast(d_alive.data()));
        if (!launch_ok("mark_dominated_segments_kernel_coupling", reason)) {
            return false;
        }

        thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
        const int total_live = thrust::reduce(d_alive.begin(), d_alive.end(), 0);

        write_segment_live_counts_kernel<<<ceil_div(num_batch_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_pair_offsets.data()),
            thrust::raw_pointer_cast(d_pair_counts.data()),
            thrust::raw_pointer_cast(d_alive.data()),
            num_batch_nodes,
            thrust::raw_pointer_cast(d_live_counts.data()));
        if (!launch_ok("write_segment_live_counts_kernel", reason)) {
            return false;
        }

        scatter_node_live_counts_kernel<<<ceil_div(num_batch_nodes, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_node_ids.data()),
            thrust::raw_pointer_cast(d_live_counts.data()),
            num_batch_nodes,
            thrust::raw_pointer_cast(coupled->sizes.data()));
        if (!launch_ok("scatter_node_live_counts_kernel", reason)) {
            return false;
        }

        d_live_points.resize(static_cast<size_t>(total_live) * NOBJS);
        if (total_live > 0) {
            scatter_alive_points_kernel<<<ceil_div(total_pairs, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_alive.data()),
                thrust::raw_pointer_cast(d_alive_prefix.data()),
                thrust::raw_pointer_cast(d_cand_points.data()),
                thrust::raw_pointer_cast(d_live_points.data()),
                total_pairs);
            if (!launch_ok("scatter_alive_points_kernel_coupling", reason)) {
                return false;
            }
        }

        if (total_live > 0) {
            const size_t live_elems = static_cast<size_t>(total_live) * NOBJS;
            const size_t old_elems = d_coupled_points.size();
            const size_t required_elems = old_elems + live_elems;

            if (required_elems > coupled_capacity_elems) {
                size_t new_capacity = (coupled_capacity_elems == 0 ? required_elems : coupled_capacity_elems);
                while (new_capacity < required_elems) {
                    const size_t grow = std::max<size_t>(new_capacity / 2, static_cast<size_t>(NOBJS) * 1024);
                    new_capacity += grow;
                }
                d_coupled_points.reserve(new_capacity);
                coupled_capacity_elems = d_coupled_points.capacity();
            }

            d_coupled_points.resize(required_elems);
            running_total += total_live;
            thrust::copy(d_live_points.begin(),
                         d_live_points.end(),
                         d_coupled_points.begin() + old_elems);
        }

        if (!sync_stream0("couple_layer_batch_sync", reason)) {
            return false;
        }
    }

    int total_points = 0;
    if (!compute_offsets_from_sizes(&(coupled->sizes), &(coupled->offsets), &total_points, reason)) {
        return false;
    }

    if (total_points != running_total) {
        return set_reason(reason, "Coupling size mismatch after batching");
    }

    coupled->points.swap(d_coupled_points);
    return true;
}

inline void swap_layer_frontier(DeviceLayerFrontier* a,
                                DeviceLayerFrontier* b) {
    std::swap(a->num_nodes, b->num_nodes);
    a->sizes.swap(b->sizes);
    a->offsets.swap(b->offsets);
    a->points.swap(b->points);
}

inline bool tree_reduce_to_single(DeviceLayerFrontier* frontier,
                                  std::string* reason) {
    if (frontier == NULL) {
        return set_reason(reason, "Frontier is NULL in tree reduction");
    }
    if (frontier->num_nodes <= 0) {
        return set_reason(reason, "No coupled nodes available for tree reduction");
    }

    DeviceLayerFrontier current = *frontier;
    DeviceLayerFrontier next;

    int pair_capacity = 0;
    size_t candidate_capacity = 0;
    thrust::device_vector<int> d_pair_counts;
    thrust::device_vector<int> d_pair_offsets;
    thrust::device_vector<int> d_alive;
    thrust::device_vector<int> d_alive_prefix;
    thrust::device_vector<ObjType> d_candidates;

    while (current.num_nodes > 1) {
        const int num_pairs = (current.num_nodes + 1) / 2;

        next.num_nodes = num_pairs;
        next.sizes.assign(num_pairs, 0);
        next.offsets.assign(num_pairs + 1, 0);

        if (num_pairs > pair_capacity) {
            d_pair_counts.reserve(num_pairs);
            d_pair_offsets.reserve(num_pairs + 1);
            pair_capacity = num_pairs;
        }
        d_pair_counts.resize(num_pairs);
        d_pair_offsets.resize(num_pairs + 1);

        compute_union_pair_counts_kernel<<<ceil_div(num_pairs, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(current.sizes.data()),
            current.num_nodes,
            thrust::raw_pointer_cast(d_pair_counts.data()));
        if (!launch_ok("compute_union_pair_counts_kernel", reason)) {
            return false;
        }

        thrust::exclusive_scan(d_pair_counts.begin(), d_pair_counts.end(), d_pair_offsets.begin());
        const int total_candidates = thrust::reduce(d_pair_counts.begin(), d_pair_counts.end(), 0);
        d_pair_offsets[num_pairs] = total_candidates;

        if (total_candidates <= 0) {
            next.points.clear();
            if (!sync_stream0("tree_reduce_round_sync_empty", reason)) {
                return false;
            }
            swap_layer_frontier(&current, &next);
            continue;
        }

        if (static_cast<size_t>(total_candidates) > candidate_capacity) {
            d_candidates.reserve(static_cast<size_t>(total_candidates) * NOBJS);
            d_alive.reserve(total_candidates);
            d_alive_prefix.reserve(total_candidates);
            candidate_capacity = static_cast<size_t>(total_candidates);
        }
        d_candidates.resize(static_cast<size_t>(total_candidates) * NOBJS);
        d_alive.resize(total_candidates);
        d_alive_prefix.resize(total_candidates);

        build_union_candidates_kernel<<<num_pairs, kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(current.offsets.data()),
            thrust::raw_pointer_cast(current.sizes.data()),
            current.num_nodes,
            thrust::raw_pointer_cast(d_pair_offsets.data()),
            num_pairs,
            thrust::raw_pointer_cast(current.points.data()),
            thrust::raw_pointer_cast(d_candidates.data()));
        if (!launch_ok("build_union_candidates_kernel", reason)) {
            return false;
        }

        mark_dominated_segments_kernel<<<num_pairs, kThreadsPerBlock, static_cast<size_t>(kThreadsPerBlock) * NOBJS * sizeof(ObjType)>>>(
            thrust::raw_pointer_cast(d_candidates.data()),
            thrust::raw_pointer_cast(d_pair_offsets.data()),
            thrust::raw_pointer_cast(d_pair_counts.data()),
            num_pairs,
            thrust::raw_pointer_cast(d_alive.data()));
        if (!launch_ok("mark_dominated_segments_kernel_tree", reason)) {
            return false;
        }

        thrust::exclusive_scan(d_alive.begin(), d_alive.end(), d_alive_prefix.begin());
        const int total_next = thrust::reduce(d_alive.begin(), d_alive.end(), 0);

        write_segment_live_counts_kernel<<<ceil_div(num_pairs, kThreadsPerBlock), kThreadsPerBlock>>>(
            thrust::raw_pointer_cast(d_pair_offsets.data()),
            thrust::raw_pointer_cast(d_pair_counts.data()),
            thrust::raw_pointer_cast(d_alive.data()),
            num_pairs,
            thrust::raw_pointer_cast(next.sizes.data()));
        if (!launch_ok("write_segment_live_counts_kernel_tree", reason)) {
            return false;
        }

        thrust::exclusive_scan(next.sizes.begin(), next.sizes.end(), next.offsets.begin());
        next.offsets[num_pairs] = total_next;

        next.points.resize(static_cast<size_t>(total_next) * NOBJS);
        if (total_next > 0) {
            scatter_alive_points_kernel<<<ceil_div(total_candidates, kThreadsPerBlock), kThreadsPerBlock>>>(
                thrust::raw_pointer_cast(d_alive.data()),
                thrust::raw_pointer_cast(d_alive_prefix.data()),
                thrust::raw_pointer_cast(d_candidates.data()),
                thrust::raw_pointer_cast(next.points.data()),
                total_candidates);
            if (!launch_ok("scatter_alive_points_kernel_tree", reason)) {
                return false;
            }
        }

        if (!sync_stream0("tree_reduce_round_sync", reason)) {
            return false;
        }

        swap_layer_frontier(&current, &next);
    }

    *frontier = current;
    return true;
}

} // namespace


bool coupled_cuda_available(std::string* reason) {
    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess) {
        return set_reason(reason, std::string("cudaGetDeviceCount failed: ") + cudaGetErrorString(err));
    }
    if (device_count <= 0) {
        return set_reason(reason, "No CUDA device found");
    }
    return true;
}


ParetoFrontier* coupled_cuda_enumerate(BDD* bdd,
                                       bool maximization,
                                       const int problem_type,
                                       const int dominance_strategy,
                                       MultiObjectiveStats* stats,
                                       std::string* reason) {
    (void)problem_type;
    if (bdd == NULL) {
        set_reason(reason, "BDD pointer is NULL");
        return NULL;
    }
    if (bdd->num_layers <= 0) {
        set_reason(reason, "BDD has zero layers");
        return NULL;
    }
    if (!coupled_cuda_available(reason)) {
        return NULL;
    }
    if (!cuda_ok(cudaSetDevice(0), "cudaSetDevice", reason)) {
        return NULL;
    }

    if (stats != NULL) {
        stats->pareto_dominance_time = 0;
        stats->pareto_dominance_filtered = 0;
        stats->layer_coupling = 0;
    }

    if (dominance_strategy > 0) {
        std::cerr << "Warning: CUDA dynamic-cutset currently ignores state dominance filtering; proceeding without dominance pruning." << std::endl;
    }

    const int root_nodes = static_cast<int>(bdd->layers[0].size());
    const int terminal_layer = bdd->num_layers - 1;
    const int terminal_nodes = static_cast<int>(bdd->layers[terminal_layer].size());

    const int root_idx = bdd->get_root()->index;
    const int terminal_idx = bdd->get_terminal()->index;

    if (root_idx < 0 || root_idx >= root_nodes) {
        set_reason(reason, "Invalid root index for layer 0");
        return NULL;
    }
    if (terminal_idx < 0 || terminal_idx >= terminal_nodes) {
        set_reason(reason, "Invalid terminal index for last layer");
        return NULL;
    }

    DeviceLayerFrontier top;
    top.num_nodes = root_nodes;
    top.sizes.assign(root_nodes, 0);
    thrust::host_vector<int> h_top_root_sizes(root_nodes, 0);
    h_top_root_sizes[root_idx] = 1;
    top.sizes = h_top_root_sizes;
    top.offsets.assign(root_nodes + 1, 0);
    top.offsets[root_nodes] = 1;
    top.points.assign(NOBJS, 0);

    DeviceLayerFrontier bottom;
    bottom.num_nodes = terminal_nodes;
    bottom.sizes.assign(terminal_nodes, 0);
    thrust::host_vector<int> h_bottom_terminal_sizes(terminal_nodes, 0);
    h_bottom_terminal_sizes[terminal_idx] = 1;
    bottom.sizes = h_bottom_terminal_sizes;
    bottom.offsets.assign(terminal_nodes + 1, 0);
    bottom.offsets[terminal_nodes] = 1;
    bottom.points.assign(NOBJS, 0);

    int layer_topdown = 0;
    int layer_bottomup = terminal_layer;
    int val_topdown = 0;
    int val_bottomup = 0;

    std::vector<int> h_edge_src;
    std::vector<int> h_edge_dst;
    std::vector<ObjType> h_edge_weights;

    while (layer_topdown != layer_bottomup) {
        if (val_topdown <= val_bottomup) {
            ++layer_topdown;
            build_topdown_edges(bdd, layer_topdown, maximization, &h_edge_src, &h_edge_dst, &h_edge_weights);

            DeviceLayerFrontier next_top;
            if (!expand_from_edges(top,
                                   static_cast<int>(bdd->layers[layer_topdown].size()),
                                   h_edge_src,
                                   h_edge_dst,
                                   h_edge_weights,
                                   &next_top,
                                   reason))
            {
                return NULL;
            }

            top = next_top;
            thrust::host_vector<int> h_sizes = top.sizes;
            val_topdown = compute_top_layer_value(bdd, layer_topdown, h_sizes);
        } else {
            --layer_bottomup;
            build_bottomup_edges(bdd, layer_bottomup, maximization, &h_edge_src, &h_edge_dst, &h_edge_weights);

            DeviceLayerFrontier next_bottom;
            if (!expand_from_edges(bottom,
                                   static_cast<int>(bdd->layers[layer_bottomup].size()),
                                   h_edge_src,
                                   h_edge_dst,
                                   h_edge_weights,
                                   &next_bottom,
                                   reason))
            {
                return NULL;
            }

            bottom = next_bottom;
            thrust::host_vector<int> h_sizes = bottom.sizes;
            val_bottomup = compute_bottom_layer_value(bdd, layer_bottomup, h_sizes);
        }
    }

    if (stats != NULL) {
        stats->layer_coupling = layer_topdown;
    }

    size_t max_pairs_per_batch = 0;
    if (!determine_max_pairs(reason, &max_pairs_per_batch)) {
        return NULL;
    }

    DeviceLayerFrontier coupled;
    if (!couple_layer_batched(top, bottom, max_pairs_per_batch, &coupled, reason)) {
        return NULL;
    }

    if (!tree_reduce_to_single(&coupled, reason)) {
        return NULL;
    }

    const int num_points = static_cast<int>(coupled.points.size() / NOBJS);
    ParetoFrontier* frontier = new ParetoFrontier;
    frontier->sols.resize(static_cast<size_t>(num_points) * NOBJS, 0);
    if (num_points > 0) {
        thrust::host_vector<ObjType> h_points = coupled.points;
        std::copy(h_points.begin(), h_points.end(), frontier->sols.begin());
    }

    if (reason != NULL) {
        reason->clear();
    }
    return frontier;
}
