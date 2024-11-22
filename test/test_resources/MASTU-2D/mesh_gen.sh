#!/bin/bash
GEO_FILE="mastu_cd.geo"
BASENAME=$(basename $GEO_FILE .geo)
gmsh -2 $GEO_FILE
NEKMESH=NekMesh
if hash NekMesh-rg 2>/dev/null; then
    NEKMESH=NekMesh-rg
fi

$NEKMESH -v "$BASENAME.msh" "$BASENAME.tmp.xml":xml:uncompress

awk '!/EXPANSIONS/' "$BASENAME.tmp.xml" > "$BASENAME.tmp2.xml"
rm "$BASENAME.tmp.xml"
awk '!/NUMMODES/' "$BASENAME.tmp2.xml" > "$BASENAME.xml"
rm "$BASENAME.tmp2.xml"
