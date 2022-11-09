#!/bin/env bash

#--------------------------------------------------------------------------------------------------
# Helper functions
echo_usage() {
    echo "Usage:"
    echo "    $0 [solver_name] [example_name] <-n num_MPI>"
}

execute() {
    local run_cmd=$1
    local run_dir=$2
    echo "----------------------------------------------------------------------------------------------------"
    echo "Executing [$run_cmd] in [$run_dir]"
    cd "$run_dir" 
    eval "$run_cmd"
    cd -
    echo "----------------------------------------------------------------------------------------------------"
}

generate_run_dir() {
    local eg_dir="$1"
    local run_dir="$2"
    run_dir="$REPO_ROOT/example-runs/$solver_name/$eg_name"
    if [ -e "$run_dir" ]; then
        read -p "Overwrite existing run directory at $run_dir? (Y/N): " choice && [[ $choice == [yY] || $choice == [yY][eE][sS] ]] || exit 5
        \rm -rf "$run_dir"
    fi
    mkdir -p "$(dirname $run_dir)"
    cp -r "$eg_dir" "$run_dir"
}

parse_args() {
    if [ $# -lt 2 ]; then
        echo_usage
        exit 1
    fi
    POSITIONAL_ARGS=()
    while [[ $# -gt 0 ]]; do
    case $1 in
        -n|--num_mpi)
        nmpi="$2"
        shift 2
        ;;
        -*|--*)
        echo "Unknown option $1"
        exit 2
        ;;
        *)
        # Save positional args in an array
        POSITIONAL_ARGS+=("$1")
        shift
        ;;
    esac
    done

    # Restore and extract positional args
    set -- "${POSITIONAL_ARGS[@]}"
    solver_name=$1
    eg_name=$2
}

report_options() {
    #echo "--------------------------------------------------"
    echo "Options:"
    echo "    Solver : $solver_name"
    echo "      e.g. : $eg_name"
    echo "     n MPI : $nmpi"
    echo ""
}

set_run_cmd() {
    local run_dir=$1
    local solver_exec="$2"
    local nmpi="$3"
    run_cmd_file="$run_dir/run_cmd_template.txt"
    if [ -f "$run_cmd_file" ]; then
        run_cmd=$(sed -e 's|<SOLVER_EXEC>|'"$solver_exec"'|g' -e 's|<NMPI>|'"$nmpi"'|g'< "$run_cmd_file")
        \rm "$run_cmd_file"
    else
        echo "Can't read template run command from $run_cmd_file."
        exit 6
    fi
}

validate_paths() {
    local solver_exec=$1
    local eg_dir=$2
    if [ ! -f "$solver_exec" ]; then
        echo "No solver found at $solver_exec"
        exit 3
    fi
    if [ ! -d "$eg_dir" ]; then
        echo "No example directory found at $eg_dir"
        exit 4
    fi
}
#--------------------------------------------------------------------------------------------------

# Default options
solver_name='Not set'
eg_name='Not set'
nmpi='4'

# Parse command line args and report resulting options
parse_args $*
report_options

# Set paths to the solver executable and example directory
REPO_ROOT=$( cd -- "$(realpath $( dirname -- "${BASH_SOURCE[0]}" )/..)" &> /dev/null && pwd )
solver_exec="$REPO_ROOT/solvers/$solver_name/build/dist/$solver_name"
eg_dir="$REPO_ROOT/examples/$solver_name/$eg_name"
# Validate exec, examples paths
validate_paths "$solver_exec" "$eg_dir"

# Set up run directory, confirming overwrite if it already exists
run_dir="$REPO_ROOT/example-runs/$solver_name/$eg_name"
generate_run_dir "$eg_dir" "$run_dir"

# Read run command template and populate it
run_cmd="Not set"
set_run_cmd "$run_dir" "$solver_exec" "$nmpi"

# Execute run_cmd in run_dir
execute "$run_cmd" "$run_dir"
