// --------------------------------------------------
// Multiobjective
// --------------------------------------------------

// General includes
#include <algorithm>
#include <iostream>
#include <numeric>
#include <sstream>
#include <string>
#include <vector>
#include <cstdlib>

#include "bdd/bdd.hpp"
#include "bdd/bdd_alg.hpp"
#include "bdd/bdd_multiobj.hpp"
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

// Portfolio optimization includes
#include "instances/portfolio_opt_instance.hpp"
#include "bdd/portfolio_opt_bdd.hpp"

// Absolute value instance
#include "instances/absval_instance.hpp"
#include "bdd/absval_bdd.hpp"

// TSP instance
#include "instances/tsp_instance.hpp"
#include "mdd/tsp_mdd.hpp"


using namespace std;

static std::vector<ObjType> canonicalize_frontier(const ParetoFrontier* frontier) {
    if (frontier == NULL || frontier->sols.empty()) {
        return std::vector<ObjType>();
    }

    const int num_sols = frontier->get_num_sols();
    std::vector<int> order(num_sols, 0);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [frontier](int lhs, int rhs) {
        const int lhs_base = lhs * NOBJS;
        const int rhs_base = rhs * NOBJS;
        for (int o = 0; o < NOBJS; ++o) {
            if (frontier->sols[lhs_base + o] != frontier->sols[rhs_base + o]) {
                return frontier->sols[lhs_base + o] < frontier->sols[rhs_base + o];
            }
        }
        return false;
    });

    std::vector<ObjType> canonical;
    canonical.reserve(frontier->sols.size());
    for (int idx : order) {
        const int base = idx * NOBJS;
        for (int o = 0; o < NOBJS; ++o) {
            canonical.push_back(frontier->sols[base + o]);
        }
    }
    return canonical;
}

static bool compare_frontiers_exact(const ParetoFrontier* gpu_frontier,
                                    const ParetoFrontier* cpu_frontier,
                                    std::string* reason) {
    const std::vector<ObjType> gpu = canonicalize_frontier(gpu_frontier);
    const std::vector<ObjType> cpu = canonicalize_frontier(cpu_frontier);

    if (gpu.size() != cpu.size()) {
        if (reason != NULL) {
            std::ostringstream oss;
            oss << "size mismatch (gpu=" << (gpu.size() / NOBJS) << ", cpu=" << (cpu.size() / NOBJS) << ")";
            *reason = oss.str();
        }
        return false;
    }

    for (size_t i = 0; i < gpu.size(); ++i) {
        if (gpu[i] != cpu[i]) {
            if (reason != NULL) {
                std::ostringstream oss;
                oss << "first mismatch at flattened index " << i
                    << " (gpu=" << gpu[i] << ", cpu=" << cpu[i] << ")";
                *reason = oss.str();
            }
            return false;
        }
    }

    if (reason != NULL) {
        reason->clear();
    }
    return true;
}

//
// Main function
//
int main(int argc, char* argv[]) {
    if (argc != 8) {
        cout << '\n';
        cout << "Usage: multiobj [input file] [problem type] [preprocess?] [method] [appr-S] [appr-T] [dominance]\n";
        
        cout << "\n\twhere:";
        
        cout << "\n";
        cout << "\t\tproblem_type = 1: knapsack\n";
        cout << "\t\tproblem_type = 2: set packing\n";
        cout << "\t\tproblem_type = 3: set covering\n";
        cout << "\t\tproblem_type = 4: portfolio optimization\n";
        cout << "\t\tproblem_type = 5: absolute value\n";
        cout << "\t\tproblem_type = 6: TSP\n";

        cout << "\n";
        cout << "\t\tpreprocess = 0: do not preprocess instance\n";
        cout << "\t\tpreprocess = 1: preprocess input to minimize BDD size\n";
        
        cout << "\n";
        cout << "\t\tmethod = 1: top-down BFS\n";
        cout << "\t\tmethod = 2: bottom-up BFS\n";
        cout << "\t\tmethod = 3: dynamic layer cutset\n";
        
        cout << "\n";
        cout << "\t\tapprox = n m: approximate n-sized S set and m-sized T set (n=0 if disabled)\n";

        cout << "\n";
        cout << "\t\tdominance = 0:  disable state dominance\n";
        cout << "\t\tdominance = 1:  state dominance strategy 1\n";

        cout << endl;
        exit(1);
    }
    
    // Read input
    int problem_type = atoi(argv[2]);
    bool preprocess = (argv[3][0] == '1');
    int method = atoi(argv[4]);
    bool maximization = true;
    int approx_S = atoi(argv[5]);
    int approx_T = atoi(argv[6]);
    int dominance = atoi(argv[7]);
    
    
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
    BDD* bdd = NULL;
    vector< vector<int> > obj_coeffs;
    timers.start_timer(bdd_compilation_time);
    
    // --- Knapsack ---
    if (problem_type == 1) {
        
        // Read instance
        KnapsackInstance inst;
        inst.read(argv[1]);
        
        // if (preprocess) {
        //     // Reorder variables
        //     inst.reorder_coefficients();
        // }
        
        // Construct BDD
        KnapsackBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();
        //obj_coeffs = inst.obj_coeffs;
        
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
        if (approx_S != 0) {
            timers.start_timer(approx_time);
            //BDDMultiObj::approximate_pareto_frontier_bottomup(bdd, approx_S, approx_T);
            //BDDMultiObj::approximate_pareto_frontier_topdown(bdd, approx_S, approx_T);
            //BDDMultiObj::approximate_pareto_frontier_topdown_dominance(bdd, approx_S, approx_T);
            timers.end_timer(approx_time);
        }


        // Reduce BDD
        BDDAlg::reduce(bdd);
        
        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
        
        //cout << "Reduced-2 width: " << reduced_width << " - number of nodes: " << reduced_num_nodes << endl;

        // Update node weights
        bddCons.update_node_weights(bdd);    

//        bdd->print();
    }
    
    
    // --- Set Packing ---
    else if (problem_type == 2) {
        
        // read instance
        SetPackingInstance setpack(argv[1]);
        
        // create associated independent set instance
        IndepSetInst* inst = setpack.create_indepset_instance();
        
        // generate independent set BDD
        IndepSetBDDConstructor bddConstructor(inst, setpack.objs);
        bdd = bddConstructor.generate_exact();
        
        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();
        
        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
    }
    
    
    // --- Set Covering ---
    else if (problem_type == 3) {
        // set objective sense
        maximization = false;
        
        // read instance
        SetCoveringInstance setcover(argv[1]);
        
        // preprocess
        if (preprocess) {
            setcover.minimize_bandwidth();
        }
        
        // create BDD
        SetCoveringBDDConstructor bddConstructor(&setcover, setcover.objs);
        bdd = bddConstructor.generate_exact();
        
        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();
        
        // Reduce BDD
        //BDDAlg::reduce(bdd);
        
        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
        
    
    // --- Portfolio Optimization ---
    } else if (problem_type == 4) {
        
        // Read instance
        PortfolioInstance inst;
        inst.read_BDD(argv[1]);
        
        // Construct BDD
        PortfolioOptBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();
        
        assert( bdd != NULL );
        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();
        
        // Reduce BDD
        //BDDAlg::reduce(bdd);
        
        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();
        

    // --- AbsVal ---
    } else if (problem_type == 5) {
        // Set objective sense
        maximization = false;

        AbsValInstance inst;
        inst.read_BDD(argv[1]);

        AbsValBDDConstructor bddCons(&inst);
        bdd = bddCons.generate_exact();

        assert( bdd != NULL );
        original_width = bdd->get_width();
        original_num_nodes = bdd->get_num_nodes();
        
        // Reduce BDD
        //BDDAlg::reduce(bdd);
        
        reduced_width = bdd->get_width();
        reduced_num_nodes = bdd->get_num_nodes();


    // --- TSP ---
    } else if (problem_type == 6) {

        clock_t init_tsp = clock();

        // Read instance
        TSPInstance inst;
        inst.read(argv[1]);

        // Construct MDD
        clock_t compilation_tsp = clock();

        MDDTSPConstructor mddCons(&inst);
        MDD* mdd = mddCons.generate_exact();
        assert( mdd != NULL );

        compilation_tsp = clock() - compilation_tsp;

        // Generate frontier
        clock_t frontier_tsp = clock();

        // cout << "\nGenerating frontier..." << endl;
        MultiObjectiveStats* statsMultiObj = new MultiObjectiveStats;
        ParetoFrontier* pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(mdd, statsMultiObj);
        assert( pareto_frontier != NULL );

        frontier_tsp = clock() - frontier_tsp;

        cout << pareto_frontier->get_num_sols() << endl;
        cout << (double)(compilation_tsp + frontier_tsp)/CLOCKS_PER_SEC << endl;
        cout << (double)compilation_tsp/CLOCKS_PER_SEC;
        cout << "\t" << frontier_tsp/CLOCKS_PER_SEC;
        cout << endl;
    
        return 0;

    } else {
        cout << "Error - problem type not recognized" << endl;
        exit(1);
    }
    
    timers.end_timer(bdd_compilation_time);
    
    
    // cout << "\nBDD Info:\n";
    // cout << "\tOriginal width: " << original_width << endl;
    // cout << "\tOriginal number of nodes: " << original_num_nodes << endl;
    // cout << "\n\tReduced width: " << reduced_width << endl;
    // cout << "\tReduced number of nodes: " << reduced_num_nodes << endl;
    // cout << "\n\tBDD compilation total time: " << timers.get_time(bdd_compilation_time) << endl;
        
    // Initialize multiobjective stats
    MultiObjectiveStats* statsMultiObj = new MultiObjectiveStats;

    // Compute pareto frontier based on methodology
    //cout << "\n\nComputing pareto frontier..." << endl;
    ParetoFrontier* pareto_frontier = NULL;
    bool used_cuda_method3 = false;
    timers.start_timer(pareto_time);
    
    if (method == 1) {
        // -- Optimal BFS algorithm: top-down --
#ifdef USE_CUDA
        pareto_frontier = BDDMultiObj::pareto_frontier_topdown_cuda(bdd, maximization, problem_type, dominance, statsMultiObj);
        if (pareto_frontier == NULL) {
            cout << "Error: CUDA enumeration failed in GPU build." << endl;
            exit(1);
        }
#else
        pareto_frontier = BDDMultiObj::pareto_frontier_topdown(bdd, maximization, problem_type, dominance, statsMultiObj);
#endif
        
    } else if (method == 2) {
        // -- Optimal BFS algorithm: bottom-up --
        pareto_frontier = BDDMultiObj::pareto_frontier_bottomup(bdd, maximization, problem_type, dominance, statsMultiObj);
        
    } else if (method == 3) {
        // -- Dynamic layer cutset --
#ifdef USE_CUDA
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset_cuda(bdd, maximization, problem_type, dominance, statsMultiObj);
        used_cuda_method3 = true;
#else
        pareto_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, statsMultiObj);
#endif
    }

	if (pareto_frontier == NULL) {
		cout << "\nError - pareto frontier not computed" << endl;
		exit(1);
	}

    timers.end_timer(pareto_time);

#ifdef USE_CUDA
    if (used_cuda_method3) {
        const char* verify_env = getenv("MULTIOBJ_VERIFY_GPU_METHOD3");
        const bool verify = (verify_env != NULL && verify_env[0] == '1');
        if (verify) {
            MultiObjectiveStats cpu_stats;
            ParetoFrontier* cpu_frontier = BDDMultiObj::pareto_frontier_dynamic_layer_cutset(bdd, maximization, problem_type, dominance, &cpu_stats);
            if (cpu_frontier == NULL) {
                cout << "Error: CPU parity check failed because CPU dynamic-cutset returned NULL." << endl;
                exit(1);
            }

            std::string mismatch_reason;
            if (!compare_frontiers_exact(pareto_frontier, cpu_frontier, &mismatch_reason)) {
                cout << "Error: GPU method=3 parity check failed (" << mismatch_reason << ")." << endl;
                exit(1);
            }
            if (dominance == 0 && statsMultiObj->layer_coupling != cpu_stats.layer_coupling) {
                cout << "Error: GPU method=3 parity check failed (layer_coupling mismatch: gpu="
                     << statsMultiObj->layer_coupling << ", cpu=" << cpu_stats.layer_coupling << ")." << endl;
                exit(1);
            }
        }
    }
#endif

    double total_time = (timers.get_time(bdd_compilation_time) + timers.get_time(approx_time) + timers.get_time(pareto_time) );

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
    cout << "\t" << ((double)statsMultiObj->pareto_dominance_time)/CLOCKS_PER_SEC;    
    cout << endl;    

	return 0;
}
