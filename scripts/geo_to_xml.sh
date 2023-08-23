#!/bin/env bash

#------------------------------------------------------------------------------
# Helper functions
check_exec() {
    local cmd="$1"
    local flag="$2"
    if ! command -v $cmd &> /dev/null
    then
        echo "[$cmd] doesn't seem to be a valid executable"
        if [ -n "$flag" ]; then
            echo "Override location with -$flag <path>"
        fi
        exit
    fi
}

echo_usage() {
    echo "Usage:"
    echo "    $0 [path_to_geo_file] <-g gmsh_path> <-m NekMesh_path> <-o output_filename>"
}

parse_args() {
    POSITIONAL_ARGS=()
    while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gmsh)
        gm_exec="$2"
        shift 2
        ;;
        -m|--nekmesh)
        nm_exec="$2"
        shift 2
        ;;
        -n|--ndims)
        ndims="$2"
        case $ndims in
            1|2|3);;
            *)
            echo "Invalid number of dimensions [$ndims]; must be 1, 2 or 3"
            exit 3
            ;;
        esac
        shift 2
        ;;
        -o|--output)
        output_fname="$2"
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
    if [ $# -lt 1 ]; then
        echo_usage
        exit 1
    fi
    geo_path=$1
}

report_options() {
    echo "Options:"
    echo "    path to .geo : $geo_path"
    echo "     output path : $xml_path"
    output_fname
    echo "          n dims : $ndims"
    echo "       gmsh exec : $gm_exec"
    echo "    NekMesh exec : $nm_exec"
    echo ""
}
#------------------------------------------------------------------------------

# Default options
geo_path="Not set"
gm_exec="gmsh"
ndims="3"
nm_exec="NekMesh"

# Parse command line args and report resulting options
parse_args $*
report_options

# Check gmsh, NekMesh can be found
check_exec "$gm_exec" "g"
check_exec "$nm_exec" "m"

# Run gmsh
gmsh_cmd="$gm_exec -$ndims $geo_path"
echo "Running [$gmsh_cmd]"
gmsh_output=$($gmsh_cmd)
gmsh_ret_code=$? 
if [ $gmsh_ret_code -ne 0 ] 
then
    echo "gmsh returned $gmsh_ret_code. Output was: "
    echo $gmsh_output
    exit 4
fi
echo Done
echo

msh_path="${geo_path%.geo}.msh"
if [ -n "$output_fname" ]; then
    xml_path="$(dirname $geo_path)/${output_fname%.xml}.xml"
else
    xml_path="${geo_path%.geo}.xml"
fi
echo $xml_path

# Remove any existing .xml file
\rm -f "$xml_path"

# Run NekMesh
nm_cmd="$nm_exec $msh_path $xml_path:xml:uncompress"
echo "Running [$nm_cmd]"
nm_output=$($nm_cmd)
nm_ret_code=$? 
if [ $nm_ret_code -ne 0 ] 
then
    echo "NekMesh returned $nm_ret_code. Output was: "
    echo $nm_output
    exit 5
fi
echo Done
echo

# Remove intermediate .msh file
\rm -f "$msh_path"

# Remove EXPANSIONS node from mesh xml
sed -i '/<EXPANSIONS>/,/<\/EXPANSIONS>/d' "$xml_path"

echo "Generated Nektar xml mesh at $xml_path"