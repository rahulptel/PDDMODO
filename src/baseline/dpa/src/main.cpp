//#define PRG_NAME "A Simple,  Efficient and Versatile Objective Space Algorithm for Multiobjective Integer Programming" 
//#define CPY_RGHT "Copyright © 2019-2021" 
//#define AUTHOR "Kerstin Daechert and Tino Fleuren and Kathrin Klamroth" 

#include "DefiningPoint.h"
#include <string>
#include <numeric>
#include <deque> 
#include <iostream>
#include <cerrno>
#include <cstring>
#include <stdexcept>
#include <sys/stat.h>
#include <sys/types.h>
#include "main.h"

using namespace std;

char* GetFileName(const char* argument) {
	char *fileName = new char[150];
	strcpy(fileName, argument);
	return fileName;
}

string GetBaseName(const string& path) {
	size_t slash = path.find_last_of("/\\");
	if (slash == string::npos)
		return path;
	return path.substr(slash + 1);
}

string RemoveExtension(const string& fileName) {
	size_t dot = fileName.find_last_of('.');
	if (dot == string::npos)
		return fileName;
	return fileName.substr(0, dot);
}

void EnsureDirectoryExists(const string& path) {
	if (path.empty())
		return;

	string current;
	for (size_t i = 0; i < path.size(); i++) {
		current += path[i];
		if (path[i] != '/' && i + 1 != path.size())
			continue;
		if (current == "/")
			continue;

		if (mkdir(current.c_str(), 0755) != 0 && errno != EEXIST)
			throw runtime_error("Could not create directory: " + current);
	}
}

string GetMethodPrefix(bool augmented, bool verbose) {
	string method = "dpa";
	if (augmented)
		method += "-a";
	if (verbose)
		method += "-v";
	return method;
}

string BuildResultFileName(const string& inputFileName, bool augmented, bool verbose) {
	string marker = "TestInstances/";
	size_t markerPosition = inputFileName.find(marker);
	string rootPrefix;
	string instancePrefix;
	string fileName = GetBaseName(inputFileName);

	if (markerPosition != string::npos) {
		rootPrefix = inputFileName.substr(0, markerPosition);
		string relativeInstancePath = inputFileName.substr(markerPosition + marker.size());
		size_t slash = relativeInstancePath.find_last_of("/\\");
		if (slash != string::npos) {
			instancePrefix = relativeInstancePath.substr(0, slash);
			fileName = relativeInstancePath.substr(slash + 1);
		}
	}

	string instanceName = RemoveExtension(fileName);
	string directory = rootPrefix + "Solutions";
	if (!instancePrefix.empty())
		directory += "/" + instancePrefix;
	directory += "/" + GetMethodPrefix(augmented, verbose);

	EnsureDirectoryExists(directory);
	return directory + "/" + instanceName + ".sol";
}

int main(int argc, char** argv) {

	if (argc < 2) {
		std::cerr << "Usage: " << argv[0] << " <file name> [-v]" << std::endl;
		std::cerr << "       " << " -v .. verbose" << std::endl;
		std::cerr << "       " << " -a .. augmented" << std::endl;
		return 1;
	}

	std::cout << "Processing: " << argv[1] << std::endl;

	const char* fileName = GetFileName(argv[1]);
	bool augmented = false;
	bool verbose = false;

	if (argc > 2 && strcmp(argv[2], "-v") == 0)
		verbose = true;
	if (argc > 2 && strcmp(argv[2], "-a") == 0)
		augmented = true;
	if (argc > 3 && strcmp(argv[3], "-v") == 0)
		verbose = true;
	if (argc > 3 && strcmp(argv[3], "-a") == 0)
		augmented = true;


	DefiningPoint df = DefiningPoint(verbose);
	try {
		// input
		df.ImportProblemSpecification(fileName);

		// arguments:
		// scalarization method (currently only econstraint-method implemented)
		// scalarization variant (augmented or not)
		// selected index for econstraint-method
		df.Compute(true, augmented, df.numObjectives - 1);


		// output
		std::string resultFileName = BuildResultFileName(fileName, augmented, verbose);
		df.ExportNonDominatedPointsToFile(resultFileName);
	}
	catch (std::exception& e) {
		std::cerr << "Error: " << e.what() << std::endl;
		return -1;
	}

	return 0;
}


