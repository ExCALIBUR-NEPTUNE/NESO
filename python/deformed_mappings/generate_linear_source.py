"""
This is a wrapper script that generates the implementations to compute Newton
steps and residuals for 3D linear Nektar++ geometry objects (and 2D
linear quadrilaterals).
"""

import sys
import os
import prism
import pyramid
import tetrahedron
import hexahedron
import quadrilateral
import quadrilateral_embed_3d
from newton_generation import *
from utility import get_generation_header

if len(sys.argv) < 2 or "-h" in sys.argv:
    print(
        """
This is a wrapper script that generates the implementations to compute Newton
steps and residuals for 3D linear Nektar++ geometry objects (and 2D
linear quadrilaterals). The script is called with a single argument like

python generate_linear_source.py <output_dir>

where <output_dir> is the output directory where generated code will be placed.
"""
    )
    quit()

# The geometry types to generate code for.
geom_types = (
    prism.get_geom_type(),
    pyramid.get_geom_type(),
    tetrahedron.get_geom_type(),
    hexahedron.get_geom_type(),
    quadrilateral.get_geom_type(),
    quadrilateral_embed_3d.get_geom_type(),
)
# Create an instance of each type
geom_objects = [gx() for gx in geom_types]
# Create a Newton instance for each type.
geom_newton = [Newton(gx) for gx in geom_objects]
# Create the C code for each type.
geom_ccode = [NewtonLinearCCode(gx) for gx in geom_newton]

file_header = get_generation_header()

# Place the generated code for each geometry type in its own header file.
output = {}
for gx in geom_ccode:
    filename = "{}.hpp".format(gx.newton.geom.namespace.lower())
    source = """/**
{FILE_HEADER}
*/
#ifndef __GENERATED_{NAMESPACE_UPPER}_LINEAR_NEWTON_H__
#define __GENERATED_{NAMESPACE_UPPER}_LINEAR_NEWTON_H__

#include <neso_particles.hpp>
using namespace NESO;
using namespace NESO::Particles;

namespace NESO {{
namespace {NAMESPACE} {{

{STEP}

{RESIDUAL}

}}
}}

#endif
""".format(
        FILE_HEADER=file_header,
        NAMESPACE_UPPER=gx.newton.geom.namespace.upper(),
        NAMESPACE=gx.newton.geom.namespace,
        STEP=gx.step(),
        RESIDUAL=gx.residual(),
    )

    output[filename] = source


output_dir = sys.argv[1]
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
    assert os.path.exists(output_dir)

for filename in output.keys():
    p = os.path.join(output_dir, filename)
    with open(p, "w") as fh:
        fh.write(output[filename])

# Create an overarching header file that includes all the generated header
# files for the individual geometry objects.
includes = "\n".join([f'#include "{s}"' for s in output.keys()])

with open(os.path.join(output_dir, "linear_newton_implementation.hpp"), "w") as fh:
    fh.write(
        f"""#ifndef __GENERATED_LINEAR_NEWTON_IMPLEMENTATIONS_H__
#define __GENERATED_LINEAR_NEWTON_IMPLEMENTATIONS_H__
/*
{file_header}
*/

{includes}

#endif
"""
    )
