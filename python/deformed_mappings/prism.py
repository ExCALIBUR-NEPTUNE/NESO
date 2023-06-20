"""
Defines the X maps for Prisms.
"""

from newton_generation import *


class LinearPrism(LinearBase):
    def __init__(self):

        num_vertices = 6
        ndim = 3
        name = "linear_3d"
        namespace = "Prism"
        x_description = """
X(xi) = c_0 v_0 + c_1 v_1 + c_2 v_2 + c_3 v_3 + c_4 v_4 + c_5 v_5

where

eta_0 = 2 * ((1 + xi_0) / (1 - xi_2)) - 1
c_0 = 0.125 * (1 - eta_0) * (1 - xi_1) * (1 - xi_2)
c_1 = 0.125 * (1 + eta_0) * (1 - xi_1) * (1 - xi_2)
c_2 = 0.125 * (1 + eta_0) * (1 + xi_1) * (1 - xi_2)
c_3 = 0.125 * (1 - eta_0) * (1 + xi_1) * (1 - xi_2)
c_4 = 0.25 * (1 - xi_1) * (1 + xi_2)
c_5 = 0.25 * (1 + xi_1) * (1 + xi_2)
"""
        LinearBase.__init__(self, num_vertices, ndim, name, namespace, x_description)

    def get_x(self, xi):

        v = self.vertices

        # note this collapases along x whilst the textbook collapses along y
        eta0 = 2 * ((1 + xi[0]) / (1 - xi[2])) - 1
        a0 = (1 - eta0) * (1 - xi[2])
        a1 = (1 + eta0) * (1 - xi[2])

        a0_no_singularity = -2 * xi[2] - 2 * xi[0]
        a1_no_singularity = 2 + 2 * xi[0]
        assert simplify(a0_no_singularity - a0) == 0
        assert simplify(a1_no_singularity - a1) == 0

        c0 = 0.125 * (1 - xi[1]) * a0_no_singularity
        c1 = 0.125 * (1 - xi[1]) * a1
        c2 = 0.125 * (1 + xi[1]) * a1
        c3 = 0.125 * (1 + xi[1]) * a0_no_singularity
        c4 = 0.25 * (1 - xi[1]) * (1 + xi[2])
        c5 = 0.25 * (1 + xi[1]) * (1 + xi[2])

        x = c0 * v[0] + c1 * v[1] + c2 * v[2] + c3 * v[3] + c4 * v[4] + c5 * v[5]

        return x


def self_test():

    geom_x = LinearPrism()

    vertices_ref = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (1.0, 1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
        (-1.0, 1.0, 1.0),
    )
    geom_ref = LinearGeomEvaluate(geom_x, vertices_ref)

    for vx in vertices_ref:
        to_test = geom_ref.x(vx)
        correct = vx
        assert (
            np.linalg.norm(
                np.array(correct).ravel() - np.array(to_test).ravel(), np.inf
            )
            < 1.0e-15
        )
        to_test = geom_ref.f(vx, vx)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    vertices_test = (
        (-3.0, -2.0, 2.0),
        (1.0, -1.0, 2.0),
        (2.0, 2.0, 2.5),
        (-1.0, 4.0, 4.5),
        (-3.0, -2.0, -0.1),
        (-1.0, 4.0, -1.5),
    )
    geom_test = LinearGeomEvaluate(geom_x, vertices_test)
    geom_newton = Newton(geom_x)

    geom_newton_evaluate = NewtonEvaluate(geom_newton, geom_test)

    xi_correct0 = -0.9
    xi_correct1 = 0.8
    xi_correct2 = 0.2
    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = geom_test.x(xi_correct)
    residual, fv = geom_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = geom_newton_evaluate.residual(xi, phys)
        xin = geom_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
        xi[2] = xin[2]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14

    vertices_test = (
        (-1, -1, 0.0),
        (-0.8, -1, 0.0),
        (-0.8, -1, -0.2),
        (-1, -1, -0.2),
        (-1, -0.8, 0.0),
        (-1, -0.8, -0.2),
    )
    geom_test = LinearGeomEvaluate(geom_x, vertices_test)
    geom_newton = Newton(geom_x)

    geom_newton_evaluate = NewtonEvaluate(geom_newton, geom_test)

    for vi, vx in enumerate(vertices_ref):
        to_test = geom_test.x(vx)
        correct = vertices_test[vi]
        assert (
            np.linalg.norm(
                np.array(correct).ravel() - np.array(to_test).ravel(), np.inf
            )
            < 1.0e-15
        )
        to_test = geom_test.f(vx, correct)
        assert np.linalg.norm(np.array(to_test).ravel(), np.inf) < 1.0e-15

    xi_correct0 = -0.6
    xi_correct1 = -0.5
    xi_correct2 = -0.2

    xi_correct = (xi_correct0, xi_correct1, xi_correct2)
    phys = geom_test.x(xi_correct)
    residual, fv = geom_newton_evaluate.residual(xi_correct, phys)
    assert residual < 1.0e-15

    # phys_nektar = ( -0.960000, -0.990000, -0.030000)
    phys_nektar = (-0.960000, -0.920000, -0.050000)

    assert abs(phys_nektar[0] - phys[0]) < 1.0e-8
    assert abs(phys_nektar[1] - phys[1]) < 1.0e-8
    assert abs(phys_nektar[2] - phys[2]) < 1.0e-8

    xi = [0.0, 0.0, 0.0]
    for stepx in range(5):
        residual, fv = geom_newton_evaluate.residual(xi, phys)
        xin = geom_newton_evaluate.step(xi, phys, fv)
        xi[0] = xin[0]
        xi[1] = xin[1]
        xi[2] = xin[2]

    assert abs(xi[0] - xi_correct[0]) < 1.0e-14
    assert abs(xi[1] - xi_correct[1]) < 1.0e-14
    assert abs(xi[2] - xi_correct[2]) < 1.0e-14


def get_geom_type():

    self_test()

    # geom_x = LinearPrism()

    # geom_newton = Newton(geom_x)
    # geom_newton_ccode = NewtonLinearCCode(geom_newton)
    # return geom_newton_ccode

    return LinearPrism


if __name__ == "__main__":
    self_test()
