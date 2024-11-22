#!/bin/bash
# This script generates the Newton implementations for linear sided elements in NESO
PYTHON=python
SCRIPT=../../python/deformed_mappings/generate_linear_source.py
OUTPUT_DIR=../../include/nektar_interface/particle_cell_mapping/generated_linear
$PYTHON $SCRIPT $OUTPUT_DIR
