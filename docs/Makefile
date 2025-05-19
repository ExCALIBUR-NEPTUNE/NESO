# Makefile for the documentation
#
# The default build location is <NESO>/docs/builds/current
# To build at a different location, pass either DOCS_BUILD_LBL=<LABEL> or DOCS_OUTDIR=<PATH>.

DOCS_BUILD_LBL := current
REPO_ROOT := $(realpath $(dir $(lastword $(MAKEFILE_LIST)))/..)
DOCS_ROOT := $(REPO_ROOT)/docs
EGS_DIR   := $(REPO_ROOT)/examples

# Default build location
DOCS_OUTDIR := $(DOCS_ROOT)/builds/${DOCS_BUILD_LBL}
DOCS_SPHINX_OUTDIR := $(DOCS_OUTDIR)/sphinx
DOCS_SPHINX_HTML_OUTDIR := $(DOCS_SPHINX_OUTDIR)/html
DOCS_DOXYGEN_OUTDIR := $(DOCS_SPHINX_HTML_OUTDIR)/doxygen

$(info Building docs in $(DOCS_OUTDIR))

# Default target
all: sanitize_readmes build revert_readmes

.PHONY: sanitize_readmes build revert_readmes all

# Target that temporarily edits the example readmes to workaround GitHub Markdown rendering bugs
sanitize_readmes:
	$(shell find $(EGS_DIR) -name "*.md" -exec sed -i {} -e 's/\\left\\\\{/\\left\\{/' -e 's/\\right\\\\}/\\right\\}/' \;)

# Target for doxygen+sphinx build
build:
	rm -rf $(DOCS_SPHINX_OUTDIR)
	mkdir -p $(DOCS_SPHINX_HTML_OUTDIR)
	$(DOCS_ROOT)/populate_script_usage_strings.sh
	cd doxygen && make DOXYGEN_OUTPUT_DIR="${DOCS_DOXYGEN_OUTDIR}"
	cd sphinx && make BUILDDIR="$(DOCS_SPHINX_OUTDIR)" html	

# Target that reverts temporary changes to the example readmes
revert_readmes:
	$(shell find $(EGS_DIR) -name "*.md" -exec git checkout -- {} \;)