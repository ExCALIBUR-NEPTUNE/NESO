///////////////////////////////////////////////////////////////////////////////
//
// Description: Entrypoint for the evaluation benchmark.
//
///////////////////////////////////////////////////////////////////////////////
#include <iostream>
#include <mpi.h>
#include <LibUtilities/BasicUtils/SessionReader.h>
#include <string>
#include <map>
using namespace Nektar;

int main_evaluation(int argc, char *argv[], LibUtilities::SessionReaderSharedPtr session);
