#!/bin/bash

# create the output dir where we collect doc versions
OUTPUT_DIR=$(pwd)/builds
mkdir -p "${OUTPUT_DIR}"

echo "Current branch: $(git rev-parse --abbrev-ref HEAD)"

# ensure the tags are up to date and echo them
git fetch --tags
echo "Available tags are: [ $(git tag -l |tr "\n" " ")]"

# create the switcher.json
python3 generate_versions_json.py
# copy the generated switcher.json to the output dir so that it is included in the deployed website
cp switcher.json "${OUTPUT_DIR}"

# determine the branches/tags from the switcher json
BRANCHES=$(python3 -c "import json; print(' '.join([fx['version'] for fx in json.loads(open('./switcher.json').read())]))")
echo "Branches/tags in switcher are: $BRANCHES"

# copy the redirecting index to the output directory
cp ./redirect_index.html "${OUTPUT_DIR}/index.html"

# clone the repo into a temporary place
REPO=https://github.com/ExCALIBUR-NEPTUNE/NESO.git
mkdir -p /tmp/repo-checkout
rm -rf "/tmp/repo-checkout/NESO"
cd /tmp/repo-checkout
git clone $REPO
cd NESO/docs


# checkout each branch to build and build the docs for that tag/branch/version in tmp
for BX in $BRANCHES
do
    echo "Building docs for branch/tag $BX"
    echo "Working dir is $PWD"

    # checkout a version and build the docs for it
    git checkout $BX
    echo "$BX" > ./sphinx/source/docs_version
    cat ./sphinx/docs_version

    # Build docs in-place if Makefile allows, else do it the old way
    if grep -q DOCS_OUTDIR Makefile; then
        make DOCS_OUTDIR="${OUTPUT_DIR}/$BX"
    else
        make
        # create a directory for this version in the global output directory
        BRANCH_OUTPUT=${OUTPUT_DIR}/$BX
        mkdir -p "${BRANCH_OUTPUT}"
        # copy the docs for this version to the global output directory
        mv build/* "${BRANCH_OUTPUT}"
    fi
done



