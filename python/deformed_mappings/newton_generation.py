"""
This module contains the implementation to symbolically compute the Newton
update step functional forms and the functional forms that evaluate the
residual.
"""

from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization
import numpy as np


def make_vector(*args):
    """
    Helper function to convert all arguments to elements of a vector.

    :param args: Entries of output vector.
    """
    return Matrix([symbols(ax) for ax in args])


class LinearBase:
    """
    Base class to describe X maps for linear sided geometry objects. These X
    maps are defined in terms of a fixed number of vertices.
    """

    def __init__(
        self,
        num_vertices: int,
        ndim: int,
        name: str,
        namespace: str,
        x_description: str,
    ):
        """
        :param num_vertices: Number of vertices in the geometry object.
        :param ndim: Number dimensions of the space the X map and vertices exist in.
        :param name: Name of the type of X map that is described, e.g. linear_3d.
        :param namespace: Name of output namespace and geometry type, e.g. Hexahedron.
        :param x_description: Description of the X map, e.g. functional form.
        This will be included in the generated C++ output.
        """
        self.num_vertices = num_vertices
        """Number of vertices that define the linear sided geometry object."""
        self.ndim = ndim
        """Number of dimensions the vertices, X(xi) and xi exist in."""
        self.vertices = [
            make_vector(*["v{}{}".format(vx, dx) for dx in range(self.ndim)])
            for vx in range(self.num_vertices)
        ]
        """Symbolic representation of the vertices that define the object and X map."""
        self.name = name
        """Name of the type of map, e.g. linear_2d."""
        self.namespace = namespace
        """Output namespace for the generated implementation, e.g. Hexahedron."""
        self.x_description = x_description
        """Docstring that describes the X map, will be included in generated output doxygen."""

    def get_f(self, xi, phys):
        """
        Get the f map for the Newton iteration.

        :param xi: Vector of xi components (symbolic).
        :param phys: Vector of target coordinate in physical space (symbolic).
        """
        x = self.get_x(xi)
        f = x - phys
        return f


class SymbolicCommon:
    """
    Base class for evaluation and generation classes. This class standardises
    the names of common attributes for these derived types.
    """

    def __init__(self, geom):
        """
        :param geom: Instance of type derived from LinearBase that describes an
        X map for a particular geometry type.
        """
        self.geom = geom
        """The symbolic representation of the geometry object."""
        self.ndim = geom.ndim
        """Number of spatial dimensions."""
        self.xi = make_vector(*["xi{}".format(dx) for dx in range(self.ndim)])
        """Symbolic representation of xi."""
        self.phys = make_vector(*["phys{}".format(dx) for dx in range(self.ndim)])
        """Symbolic representation of the target point in global (physical) space."""
        self._x = self.geom.get_x(self.xi)
        self._f = self.geom.get_f(self.xi, self.phys)


class NewtonCommon(SymbolicCommon):
    """
    Base class for Newton implementations, symbolic and evaluation.
    """

    def __init__(self, geom):
        """
        Constructor called from derived classes.

        :param geom: Instance of type derived from LinearBase.
        """
        SymbolicCommon.__init__(self, geom)
        self.xi_next = make_vector(*["xin{}".format(dx) for dx in range(self.ndim)])
        """Symbolic vector for the next Newton iteration output."""
        self.fv = make_vector(*["f{}".format(dx) for dx in range(self.ndim)])
        """Symbolic vector of f(xi) at the current iteration."""


class LinearEvaluateCommon:
    """
    Base class for objects which numerically evaluate the symbolic implementations.
    """

    def __init__(self, geom, vertices):
        """
        Constructor for evaluate class.

        :param geom: Instance of type derived from LinearBase that describes an
        X map for a particular geometry type.
        :param vertices: Container of vertices, shape (num_vertices, ndim).
        """
        assert len(vertices) == geom.num_vertices
        self.vertices = vertices
        """Container of numerical values for the vertices."""
        self.sub_vertices = {}
        """The substitutions Sympy will require to replace the symbolic representation of the vertices with numerical values."""
        for vx in range(geom.num_vertices):
            for dimx in range(self.ndim):
                self.sub_vertices[geom.vertices[vx][dimx]] = self.vertices[vx][dimx]


class LinearGeomEvaluate(SymbolicCommon):
    """
    Class to hold numerical values required to evaluate X maps and Newton iterations.
    """

    def __init__(self, geom, vertices):
        """
        Create new evaluate object.

        :param geom: Instance of type derived from LinearBase that describes an
        X map for a particular geometry type.
        :param vertices: Container shape (geom.num_vertices, geom.ndim) containing vertices (numerical values).
        """
        SymbolicCommon.__init__(self, geom)
        LinearEvaluateCommon.__init__(self, geom, vertices)

    def x(self, xi):
        """
        Evaluate the map X(xi) at a point xi.

        :param float xi: Point to evaluate X map at.
        :returns: List, size ndim of evaluations of X.
        """
        subs = {}
        for dimx in range(self.ndim):
            subs[self.xi[dimx]] = xi[dimx]
        subs.update(self.sub_vertices)
        return [float(fx) for fx in self._x.evalf(subs=subs)]

    def f(self, xi, phys):
        """
        Evaluate F(xi, phys) := phys - F(xi)

        :param xi: Numerical values of evaluation point in reference space.
        :param phys: Numerical values of target point in physical space.
        :returns: List describing F(xi, phys).
        """

        subs = {}
        for dimx in range(self.ndim):
            subs[self.xi[dimx]] = xi[dimx]
            subs[self.phys[dimx]] = phys[dimx]
        subs.update(self.sub_vertices)
        return [float(fx) for fx in self._f.evalf(subs=subs)]


class Newton(NewtonCommon):
    """
    Class that symbolically computes the form of the Newton step and residual
    functions.
    """

    def __init__(self, geom):
        """
        Compute Newton components for an X(xi) map.

        :param geom: Instance of type derived from LinearBase that describes an
        X map for a particular geometry type.
        """
        NewtonCommon.__init__(self, geom)
        self.J = self._f.jacobian(self.xi)
        """Symbolic representation of the Jacobian for f."""

        Jsymbols = []
        ndim = geom.ndim
        for rowx in range(ndim):
            Jsymbols.append([])
            for colx in range(ndim):
                Jsymbols[-1].append(symbols(f"J{rowx}{colx}"))

        self.J_symbolic = Matrix(Jsymbols)
        """Matrix that represents the Jacobian as Jij symbols."""
        self.Jinv_symbolic = self.J_symbolic.inv()
        """Inverse of the Jacobian matrix"""
        self.step = solve(
            self.xi_next - self.xi + (self.Jinv_symbolic) * self.fv,
            self.xi_next,
            dict=True,
        )
        """Symbolic representation of the Newton iteration."""
        self.step_components = [
            self.step[0][self.xi_next[dimx]] for dimx in range(self.ndim)
        ]
        """Vector of symbolic expressions to compute the next xi iteration."""
        self.f = self._f
        """The symbolic function the Newton iteration aims to make 0."""
        self.x = self._x
        """Symbolic representation of X(xi)."""


class NewtonEvaluate:
    """
    Class to numerically evaluate Newton iterations.
    """

    def __init__(self, newton: Newton, evaluate: LinearGeomEvaluate):
        """
        Create new Newton evaluation object.

        :param newton: Instance of Newton class that defines the Newton iteration symbolically.
        :param evaluate: LinearGeomEvaluate instance that defines the numerical values.
        """
        self.newton = newton
        """Newton instance to be numerically evaluated."""
        self.evaluate = evaluate
        """LinearGeomEvaluate instance that holds the numerical values for the X map."""
        assert self.newton.geom == evaluate.geom

    def residual(self, xi, phys):
        """
        Numerically evaluate the residual.

        :param xi: Vector describing xi point in reference space.
        :param phys: Vector describing target point in physical space.
        :returns: Infinity norm of residual and residual.
        """

        subs = {}
        subs.update(self.evaluate.sub_vertices)
        for dimx in range(self.evaluate.geom.ndim):
            subs[self.evaluate.phys[dimx]] = phys[dimx]
            subs[self.evaluate.xi[dimx]] = xi[dimx]

        e = self.newton.f.evalf(subs=subs)
        efloat = [float(ex) for ex in e]

        r = 0.0
        for ex in efloat:
            r = max(r, abs(ex))
        return r, efloat

    def step(self, xi, phys, fv):
        """
        Perform a Newton iteration.

        :param xi: Current iteration xi_n (numerical values).
        :param phys: Target point in physical space (numerical values).
        :param fv: F evaluation at current iteration xi_n (numerical values).
        :returns: New iteration xi_{n+1}.
        """
        subs = {}
        subs.update(self.evaluate.sub_vertices)
        ndim = self.evaluate.geom.ndim
        for dimx in range(ndim):
            subs[self.evaluate.phys[dimx]] = phys[dimx]
            subs[self.evaluate.xi[dimx]] = xi[dimx]
            subs[self.newton.fv[dimx]] = fv[dimx]

        Jeval = self.newton.J.evalf(subs=subs)
        for rowx in range(ndim):
            for colx in range(ndim):
                subs[self.newton.J_symbolic[rowx, colx]] = Jeval[rowx, colx]

        xin = [
            self.newton.step_components[dimx].evalf(subs=subs) for dimx in range(ndim)
        ]
        xinfloat = [float(ex) for ex in xin]
        return xinfloat


class NewtonLinearCCode:
    """
    Generate the C++ code that evaluates a Newton iteration and the residual.
    """

    def __init__(self, newton: Newton):
        """
        Construct new instance.

        :param newton: Newton instance that describes the functional forms.
        """
        self.newton = newton
        """Newton instance that describes the functional forms."""

    def to_c(self, rhs) -> str:
        """
        Convert a Sympy expression to C code.

        :param rhs: Sympy expression to convert to C code.
        :returns: String of C code for input expression.
        """
        expand_opt = create_expand_pow_optimization(99)
        return sympy.printing.c.ccode(expand_opt(rhs), standard="C99")

    def residual(self) -> str:
        """
        Generate the C code that evaluates the residual for given xi and target vectors.

        :returns: String containing docstring and C++ function definition.
        """
        ndim = self.newton.geom.ndim
        component_name = ("x", "y", "z")
        docs_params = []
        args = []
        for dimx in range(ndim):
            n = f"xi{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(
                f"@param[in] {n} Current xi_n point, {component_name[dimx]} component."
            )

        for vi, vx in enumerate(self.newton.geom.vertices):
            for dimx in range(ndim):
                n = f"v{vi}{dimx}"
                args.append(f"const REAL {n}")
                docs_params.append(
                    f"@param[in] {n} Vertex {vi}, {component_name[dimx]} component."
                )

        for dimx in range(ndim):
            n = f"phys{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(
                f"@param[in] {n} Target point in global space, {component_name[dimx]} component."
            )

        for dimx in range(ndim):
            n = f"f{dimx}"
            args.append(f"REAL * {n}")
            docs_params.append(
                f"@param[in, out] {n} Current f evaluation at xi, {component_name[dimx]} component."
            )

        params = "\n * ".join(docs_params)
        x_description = "\n * ".join(self.newton.geom.x_description.split("\n"))

        docstring = f"""
/**
 * Compute and return F evaluation where
 * 
 * F(xi) = X(xi) - X_phys
 * 
 * where X_phys are the global coordinates. X is defined as
 * 
 * {x_description}
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. See top of file.
 *
 * {params}
 */
"""

        args_string = ",\n".join(args)
        name = self.newton.geom.name
        s = f"""inline void newton_f_{name}(
{args_string}
)"""

        instr = ["{"]
        steps = [fx for fx in self.newton.f]
        cse_list = cse(steps, optimizations="basic")
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            rhs = self.to_c(cse_expr[1])
            expr = f"const REAL {lhs} = {rhs};"
            instr.append(expr)

        for dimx in range(ndim):
            rhs = self.to_c(cse_list[1][dimx])
            instr.append(f"const REAL f{dimx}_tmp = {rhs};")
        for dimx in range(ndim):
            instr.append(f"*f{dimx} = f{dimx}_tmp;")

        s += "\n  ".join(instr)
        s += "\n}\n"

        return docstring + s

    def step(self) -> str:
        """
        Generate the C code that evaluates the Newton update for given xi and target vectors.

        :returns: String containing docstring and C++ function definition.
        """

        ndim = self.newton.geom.ndim

        component_name = ("x", "y", "z")
        docs_params = []
        args = []
        for dimx in range(ndim):
            n = f"xi{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(
                f"@param[in] {n} Current xi_n point, {component_name[dimx]} component."
            )

        for vi, vx in enumerate(self.newton.geom.vertices):
            for dimx in range(ndim):
                n = f"v{vi}{dimx}"
                args.append(f"const REAL {n}")
                docs_params.append(
                    f"@param[in] {n} Vertex {vi}, {component_name[dimx]} component."
                )

        for dimx in range(ndim):
            n = f"phys{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(
                f"@param[in] {n} Target point in global space, {component_name[dimx]} component."
            )

        for dimx in range(ndim):
            n = f"f{dimx}"
            args.append(f"const REAL {n}")
            docs_params.append(
                f"@param[in] {n} Current f evaluation at xi, {component_name[dimx]} component."
            )

        for dimx in range(ndim):
            n = f"xin{dimx}"
            args.append(f"REAL * {n}")
            docs_params.append(
                f"@param[in, out] {n} Output local coordinate iteration, {component_name[dimx]} component."
            )

        params = "\n * ".join(docs_params)
        x_description = "\n * ".join(self.newton.geom.x_description.split("\n"))

        docstring = f"""
/**
 * Perform a Newton method update step for a Newton iteration that determines
 * the local coordinates (xi) for a given set of physical coordinates. If
 * v0,v1,v2 and v3 (passed component wise) are the vertices of a linear sided
 * quadrilateral then this function performs the Newton update:
 * 
 * xi_{{n+1}} = xi_n - J^{{-1}}(xi_n) * F(xi_n)
 * 
 * where
 * 
 * F(xi) = X(xi) - X_phys
 * 
 * where X_phys are the global coordinates. 
 * 
 * X is defined as
 * 
 * {x_description}
 *
 * This is a generated function. To modify this function please edit the script
 * that generates this function. See top of file.
 *
 * {params}
 */
"""

        args_string = ",\n".join(args)
        name = self.newton.geom.name
        s = f"""inline void newton_step_{name}(
{args_string}
)"""

        instr = ["{"]

        J_steps = []
        J_lhs = []
        for rowx in range(ndim):
            for colx in range(ndim):
                J_lhs.append(self.newton.J_symbolic[rowx, colx])
                J_steps.append(self.newton.J[rowx, colx])

        cse_list = cse(J_steps, optimizations="basic")
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            rhs = self.to_c(cse_expr[1])
            expr = f"const REAL {lhs} = {rhs};"
            instr.append(expr)

        counter = 0
        for rowx in range(ndim):
            for colx in range(ndim):
                rhs = self.to_c(cse_list[1][counter])
                expr = f"const REAL J{rowx}{colx} = {rhs};"
                counter += 1
                instr.append(expr)

        cse_list = cse(
            self.newton.step_components, numbered_symbols("y"), optimizations="basic"
        )
        for cse_expr in cse_list[0]:
            lhs = cse_expr[0]
            rhs = self.to_c(cse_expr[1])
            expr = f"const REAL {lhs} = {rhs};"
            instr.append(expr)

        for dimx in range(ndim):
            rhs = self.to_c(cse_list[1][dimx])
            instr.append(f"const REAL xin{dimx}_tmp = {rhs};")
        for dimx in range(ndim):
            instr.append(f"*xin{dimx} = xin{dimx}_tmp;")
        s += "\n  ".join(instr)
        s += "\n}\n"

        return docstring + s
