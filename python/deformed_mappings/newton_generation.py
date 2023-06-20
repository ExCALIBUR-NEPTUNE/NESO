from sympy import *
import sympy.printing.c
from sympy.codegen.rewriting import create_expand_pow_optimization
import numpy as np

init_printing(use_unicode=True)


def make_vector(*args):
    return Matrix([symbols(ax) for ax in args])


class SymbolicCommon:
    def __init__(self, geom):
        self.geom = geom
        self.ndim = geom.ndim
        self.xi = make_vector(*["xi{}".format(dx) for dx in range(self.ndim)])
        self.phys = make_vector(*["phys{}".format(dx) for dx in range(self.ndim)])
        self._x = self.geom.get_x(self.xi)
        self._f = self.geom.get_f(self.xi, self.phys)


class NewtonCommon(SymbolicCommon):
    def __init__(self, geom):
        SymbolicCommon.__init__(self, geom)
        self.xi_next = make_vector(*["xin{}".format(dx) for dx in range(self.ndim)])
        self.fv = make_vector(*["f{}".format(dx) for dx in range(self.ndim)])


class LinearEvaluateCommon:
    def __init__(self, geom, vertices):
        assert len(vertices) == geom.num_vertices
        self.vertices = vertices
        self.sub_vertices = {}
        for vx in range(geom.num_vertices):
            for dimx in range(self.ndim):
                self.sub_vertices[geom.vertices[vx][dimx]] = self.vertices[vx][dimx]


class Newton(NewtonCommon):
    def __init__(self, geom):
        NewtonCommon.__init__(self, geom)
        self.J = self._f.jacobian(self.xi)

        Jsymbols = []
        ndim = geom.ndim
        for rowx in range(ndim):
            Jsymbols.append([])
            for colx in range(ndim):
                Jsymbols[-1].append(symbols(f"J{rowx}{colx}"))

        self.J_symbolic = Matrix(Jsymbols)
        self.Jinv_symbolic = self.J_symbolic.inv()
        self.step = solve(
            self.xi_next - self.xi + (self.Jinv_symbolic) * self.fv,
            self.xi_next,
            dict=True,
        )
        self.step_components = [
            self.step[0][self.xi_next[dimx]] for dimx in range(self.ndim)
        ]
        self.f = self._f
        self.x = self._x


class NewtonEvaluate:
    def __init__(self, newton, evaluate):
        self.newton = newton
        self.evaluate = evaluate
        assert self.newton.geom == evaluate.geom

    def residual(self, xi, phys):
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
    def __init__(self, newton):
        self.newton = newton

    def to_c(self, rhs):
        expand_opt = create_expand_pow_optimization(99)
        return sympy.printing.c.ccode(expand_opt(rhs), standard="C99")

    def residual(self):
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
 * that generates this function.
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

    def step(self):
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
 * that generates this function.
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


class LinearGeomEvaluate(SymbolicCommon):
    def __init__(self, geom, vertices):
        SymbolicCommon.__init__(self, geom)
        LinearEvaluateCommon.__init__(self, geom, vertices)

    def x(self, xi):
        subs = {}
        for dimx in range(self.ndim):
            subs[self.xi[dimx]] = xi[dimx]
        subs.update(self.sub_vertices)
        return [float(fx) for fx in self._x.evalf(subs=subs)]

    def f(self, xi, phys):
        subs = {}
        for dimx in range(self.ndim):
            subs[self.xi[dimx]] = xi[dimx]
            subs[self.phys[dimx]] = phys[dimx]
        subs.update(self.sub_vertices)
        return [float(fx) for fx in self._f.evalf(subs=subs)]


class LinearBase:
    def __init__(self, num_vertices, ndim, name, namespace, x_description):
        self.num_vertices = num_vertices
        self.ndim = ndim
        self.vertices = [
            make_vector(*["v{}{}".format(vx, dx) for dx in range(self.ndim)])
            for vx in range(self.num_vertices)
        ]
        self.name = name
        self.namespace = namespace
        self.x_description = x_description

    def get_f(self, xi, phys):
        x = self.get_x(xi)
        f = x - phys
        return f
