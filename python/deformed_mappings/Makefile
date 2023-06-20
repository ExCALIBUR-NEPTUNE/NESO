OUTPUT_DEV:=./generated_linear
OUTPUT_NESO:=../../include/nektar_interface/particle_cell_mapping/generated_linear

PYTHON:=python
SCRIPT:=generate_linear_source.py
SCRIPT_TEST:=test_all.py

neso:
	${PYTHON} ${SCRIPT} ${OUTPUT_NESO}

dev:
	${PYTHON} ${SCRIPT} ${OUTPUT_DEV}

test:
	${PYTHON} ${SCRIPT_TEST}

