// ----------------------------------------------------------
// Knapsack Instance
// ----------------------------------------------------------

#ifndef KNAPSACK_INSTANCE_HPP_
#define KNAPSACK_INSTANCE_HPP_

#include <vector>

using namespace std;

//
// Multiobjective knapsack problem
//
struct KnapsackInstance
{
	// Number of variables
	int n_vars;
	// Number of constraints
	int n_cons;
	// Number of objective functions
	int num_objs;
	// Objective function coefficients (indexed by variable/objective)
	vector<vector<int>> obj_coeffs;
	// Constraint coefficients (indexed by constraint/variable)
	vector<vector<int>> coeffs;
	// Right-hand sides
	vector<int> rhs;

	// Canonical objective function coefficients (indexed by variable/objective)
	vector<vector<int>> obj_coeffs_canonical;
	// Canonical constraint coefficients (indexed by constraint/variable)
	vector<vector<int>> coeffs_canonical;

	// Empty Constructor
	KnapsackInstance() {}

	// Read instance based on our format
	void read(char *filename);

	// Print Instance
	void print();

	// Reorder variables based on constraint coefficients
	void reorder_coefficients();

	void reset_order(vector<int> new_order);
};

#endif
