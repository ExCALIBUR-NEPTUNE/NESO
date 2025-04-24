#!/bin/bash
#
# Inject script usage messages into an .rst file so they can be built into the sphinx docs
#

REPO_ROOT=$( cd -- "$(realpath $( dirname -- "${BASH_SOURCE[0]}" )/..)" &> /dev/null && pwd )

SCRIPTS_DIR="$REPO_ROOT/scripts"
SPHINX_SRC_DIR="$REPO_ROOT/docs/sphinx/source"
INPUT_RST="$SPHINX_SRC_DIR/solvers.rst.in"
TMP_RST="$SPHINX_SRC_DIR/solvers.rst.tmp"
OUTPUT_RST="$SPHINX_SRC_DIR/solvers.rst"

\rm -rf "$OUTPUT_RST"
\cp "$INPUT_RST" "$TMP_RST"
for script_fname in $(find "$SCRIPTS_DIR" -executable -name "*.sh" -printf '%f\n'); do
    # Construct tag
    lbl="${script_fname%.*}"
    tag="<${lbl@U}_USAGE>"

    # Execute script from repo root with --help get usage msg
    usage_msg=$(cd "$REPO_ROOT" && "./scripts/$script_fname" --help)
    escaped_usage_msg=$(printf '%s\n' "$usage_msg" | sed -e 's:[\\/&]:\\&:g; $!s/$/\\/' )
    # Sub into rst file
    sed -i -e "s|$tag|$escaped_usage_msg|" "$TMP_RST"
done
\mv "$TMP_RST" "$OUTPUT_RST"
