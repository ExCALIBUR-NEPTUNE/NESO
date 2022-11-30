#!/bin/bash

# create the output dir where we collect doc versions
OUTPUT_DIR=$(pwd)/build
mkdir -p ${OUTPUT_DIR}

echo $(pwd)
ls
git rev-parse --abbrev-ref HEAD

# ensure the tags are up to date
git fetch
git fetch --tags

# list tags
git tag -l

# create the switcher.json
python3 generate_versions_json.py
# copy the generated switcher.json to the output dir so that it is included in the deployed website
cp switcher.json ${OUTPUT_DIR}/.

# determine the branches from the switcher json (could also list tags instead)
BRANCHES=$(python3 -c "import json; print(' '.join([fx['version'] for fx in json.loads(open('./switcher.json').read())]))")
echo $BRANCHES

# copy the redirecting index to the output directory
cp ./redirect_index.html ${OUTPUT_DIR}/index.html

# clone the repo into a temporary place
REPO=https://github.com/ExCALIBUR-NEPTUNE/NESO.git
mkdir /tmp/repo-checkout
cd /tmp/repo-checkout
git clone $REPO
cd NESO/docs


# checkout each version to build and build the docs for that version in tmp
for BX in $BRANCHES
do
    echo $BX
    echo $(pwd)

    # checkout a version and build the docs for it
    git checkout $BX
    echo "$BX" > ./sphinx/source/docs_version
    cat ./sphinx/docs_version
    make

    # create a directory for this version in the global output directory
    BRANCH_OUTPUT=${OUTPUT_DIR}/$BX
    mkdir -p ${BRANCH_OUTPUT}
    # copy the docs for this version to the global output directory
    mv build/* ${BRANCH_OUTPUT}
done



