#!/bin/env bash

#--------------------------------------------------------------------------------------------------
# Helper functions
echo_usage() {
    echo "Usage:"
    echo "    $0 [solver_name] [example_name] <-n num_MPI> <-b build_dir/install_dir>"
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
    run_dir="$REPO_ROOT/runs/$solver_name/$eg_name"
    if [ -e "$run_dir" ]; then
        read -p "Overwrite existing run directory at $run_dir? (Y/N): " choice && [[ $choice == [yY] || $choice == [yY][eE][sS] ]] || exit 5
        \rm -rf "$run_dir"
    fi
    mkdir -p "$(dirname $run_dir)"
    cp -r "$eg_dir" "$run_dir"
}

set_default_exec_loc() {
    solver_exec=$(find -L "$REPO_ROOT/views" -maxdepth 3 -name "$solver_name" -printf "%TY-%Tm-%Td %TT %p\n" | sort -n|tail -1|cut -d " " -f 3)
}

parse_args() {
    if [ $# -lt 2 ]; then
        echo_usage
        exit 1
    fi
    POSITIONAL_ARGS=()
    while [[ $# -gt 0 ]]; do
    case $1 in
        -b|--build-dir)
        exec_loc=$(realpath "$2")
        shift 2
        ;;
        -h|--help)
        echo_usage
        exit 0
        ;;
        -n|--num_mpi)
        nmpi="$2"
        shift 2
        ;;
        -*)
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

set_exec_paths_and_validate() {
    local exec_loc=$1
    local eg_dir=$2

    
    if [ -z "$exec_loc" ]; then
        # If no executable location was specified, set solver_exec to the most
        # recently modified executable in the views
        set_default_exec_loc
        if [ -z "$solver_exec" ]; then
            echo "No installed solver found in ./views; run spack install or pass -b <build_directory>"
            exit 3
        fi
    else
        # Else check build and install locations relative to the specified $exec_loc
        solver_build_exec="$exec_loc/solvers/$solver_name/$solver_name"
        solver_install_exec="$exec_loc/bin/$solver_name"
        if [ -f "$solver_build_exec" ]; then
            solver_exec="$solver_build_exec"
        elif [ -f "$solver_install_exec" ]; then
            solver_exec="$solver_install_exec"
        else
            echo "No solver found at [$solver_build_exec] or [$solver_install_exec]"
            exit 3
        fi
    fi

    if [ ! -d "$eg_dir" ]; then
        echo "No example directory found at $eg_dir"
        exit 4
    fi
}
#--------------------------------------------------------------------------------------------------

REPO_ROOT=$( cd -- "$(realpath $( dirname -- "${BASH_SOURCE[0]}" )/..)" &> /dev/null && pwd )

# Default options
solver_name='Not set'
eg_name='Not set'
nmpi='4'
exec_loc=''

# Parse command line args and report resulting options
parse_args $*
report_options

solver_exec='Not set'
eg_dir="$REPO_ROOT/examples/$solver_name/$eg_name"
# Find the executable inside $exec_loc and validate the examples paths
set_exec_paths_and_validate "$exec_loc" "$eg_dir"

# Set up run directory, confirming overwrite if it already exists
run_dir="$REPO_ROOT/runs/$solver_name/$eg_name"
generate_run_dir "$eg_dir" "$run_dir"

# Read run command template and populate it
run_cmd="Not set"
set_run_cmd "$run_dir" "$solver_exec" "$nmpi"

# Execute run_cmd in run_dir
execute "$run_cmd" "$run_dir"
