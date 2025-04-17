DOCS_BUILD_LBL=current
DOCS_OUTDIR="$(realpath .)/builds/${DOCS_BUILD_LBL}"
DOCS_SPHINX_OUTDIR=${DOCS_OUTDIR}/sphinx
DOCS_SPHINX_HTML_OUTDIR=${DOCS_SPHINX_OUTDIR}/html
DOCS_DOXYGEN_OUTDIR=${DOCS_SPHINX_HTML_OUTDIR}/doxygen

$(info "Building in ${DOCS_OUTDIR}")

all:
	rm -rf "${DOCS_SPHINX_OUTDIR}"
	mkdir -p "${DOCS_SPHINX_HTML_OUTDIR}"
	./populate_script_usage_strings.sh
	cd doxygen && make DOXYGEN_OUTPUT_DIR="${DOCS_DOXYGEN_OUTDIR}"
	cd sphinx && make BUILDDIR="${DOCS_SPHINX_OUTDIR}" html
