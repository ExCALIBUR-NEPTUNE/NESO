#!/bin/env bash
#
# Script to convert a gmsh .geo file to a Nektar++ xml mesh.
# Requires gmsh and NekMesh.
#
# Usage:
#    ./scripts/geo_to_xml.sh [path to .geo file] <-g gmsh_path> <-m NekMesh_path> <-o output_filename> <-x x_bndry_compIDs> <-y y_bndry_compIDs> <-z z_bndry_compIDs>
# 
# If 'gmsh' or 'NekMesh' ('NekMesh-rg' also valid) aren't on your path, locations can be supplied with '-g' and '-m' respectively.
# Output is generated next to the input file being converted.
# By default, output_filename = input_filename[.geo => .xml]. Use '-o' to specify something else.
# If periodic BCs will be used, provide boundary composite IDs via the -x, -y and -z args; e.g. -x 1,2 -y 3,4 -z 5,6
#
# Example:
#   ./scripts/geo_to_xml.sh examples/DriftReduced/hw/cuboid_periodic_8x8x16.geo -o cuboid.xml --xbids 1,2 --ybids 3,4 --zbids 5,6
#
# Converts .geo file, ensuring Nektar composites are correctly aligned for periodic BCs in x,y and z directions. Output to 'cuboid.xml'.

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
    echo "    $0 [path_to_geo_file] <options>"
    echo "       Options include"
    echo '         - Specifying a path and/or arguments for gmsh: <-g gmsh_path> <--gmsh_args "arg1 arg2...">'
    echo '         - Specifying a path and/or XML file properties for NekMesh: <-m NekMesh_path> <-p "prop1 prop2...">'
    echo '           (e.g. -p "uncompress" to generate a human-readable, uncompressed XML file)'
    echo "         - Setting an output filename <-o output_filename>"
    echo "         - Specifying composite IDs for periodic boundary conditions <-x x_bndry_compIDs> <-y y_bndry_compIDs> <-z z_bndry_compIDs>"
    echo "           (e.g. -x 1,2 -y 3,4 -z 5,6)"
}

parse_args() {
    POSITIONAL_ARGS=()
    while [[ $# -gt 0 ]]; do
    case $1 in
        -g|--gmsh)
        gm_exec="$2"
        shift 2
        ;;
        --gmsh_args)
        gm_args="$gm_args $2"
        shift 2
        ;;
        -h|--help)
        echo_usage
        exit 0
        ;;
        -m|--nekmesh)
        nm_exec="$2"
        shift 2
        ;;
        -p|--xml_props)
        nm_xml_props="$2"
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
        -x|--xbids)
        xbids="$2"
        shift 2
        ;;
        -y|--ybids)
        ybids="$2"
        shift 2
        ;;
        -z|--zbids)
        zbids="$2"
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
    if [ $# -lt 1 ]; then
        echo_usage
        exit 1
    fi
    geo_path=$1
    msh_path="${geo_path%.geo}.msh"
    if [ -n "$output_fname" ]; then
        xml_path=$(dirname "$geo_path")/${output_fname%.xml}.xml
    else
        xml_path="${geo_path%.geo}.xml"
    fi
}

# Try release version of NekMesh first; fallback to RelWithDebInfo version
set_nm_default_exec() {
    nm_exec="NekMesh"
    if ! command -v $nm_exec &> /dev/null; then
        nm_exec="NekMesh-rg"
        if ! command -v $nm_exec &> /dev/null; then
            # Neither version found; reset nm_exec and leave it to check_exec to inform the user
            nm_exec="NekMesh"
        fi
    fi
}

# if physical surface/composite IDs have been passed, assemble appropriate 'peralign' argument strings for NekMesh
set_peralign_opts() {
    for dim in x y z; do
        vn="${dim}bids"
        if [ -n "${!vn}" ]; then
            arr=(${!vn//,/ })
            if [ ${#arr[@]} -ne 2 ]; then
                echo "Failed to parse boundary IDs string [${!vn}] for dimension ${dim}"
                echo "Strings should be two integers separated by a comma, with no spaces "
                exit 6
            fi
            nm_args="$nm_args -m peralign:surf1=${arr[0]}:surf2=${arr[1]}:dir=${dim}:orient"
        fi 
    done
}

set_xml_opts(){
    xml_opts="xml"
    for prop in $nm_xml_props; do
        xml_opts="$xml_opts:$prop"
    done
}

report_options() {
    echo "Using options"
    echo "      path to .geo : $geo_path"
    echo "       output path : $xml_path"
    echo "            n dims : $ndims"
    echo "         gmsh exec : $gm_exec"
    if [ -n "$gm_args" ]; then
        echo "         gmsh args : [$gm_args]"
    fi
    echo "      NekMesh exec : $nm_exec"
    if [ -n "$nm_xml_props" ]; then
        echo " NekMesh xml props : [$nm_xml_props]"
    fi
    echo ""
}
#------------------------------------------------------------------------------

# Default options
geo_path="Not set"
gm_exec="gmsh"
gm_args=""
ndims="3"
set_nm_default_exec
nm_args="-v"
nm_xml_props=""
# Parse command line args and report resulting options
parse_args "$@"
report_options

set_peralign_opts
set_xml_opts

# Check gmsh, NekMesh can be found
check_exec "$gm_exec" "g"
check_exec "$nm_exec" "m"

# Run gmsh
gm_args="-$ndims $gm_args"
gmsh_cmd="$gm_exec $gm_args $geo_path"
echo "Running [$gmsh_cmd]"
gmsh_output=$($gmsh_cmd)
gmsh_ret_code=$?
if [ $gmsh_ret_code -ne 0 ] 
then
    echo "gmsh returned $gmsh_ret_code. Output was: "
    echo "$gmsh_output"
    exit 4
fi
echo Done
echo



# Remove any existing .xml file
\rm -f "$xml_path"
# Run NekMesh
nm_cmd="$nm_exec $nm_args $msh_path $xml_path:$xml_opts"
echo "Running [$nm_cmd]"
nm_output=$($nm_cmd)
nm_ret_code=$?
if [ $nm_ret_code -ne 0 ] 
then
    echo "NekMesh returned $nm_ret_code. Output was: "
    echo "$nm_output"
    exit 5
fi
echo Done
echo

# Remove intermediate .msh file
\rm -f "$msh_path"

# Remove EXPANSIONS node from mesh xml
sed -i '/<EXPANSIONS>/,/<\/EXPANSIONS>/d' "$xml_path"

echo "Generated Nektar xml mesh at $xml_path"
echo "(remember to add an <EXPANSIONS> node to one of the xml files passed to nektar)"