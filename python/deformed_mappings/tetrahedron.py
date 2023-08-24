"""
Defines the X maps for Tetrahedrons.
"""

from newton_generation import *


class LinearTetrahedron(LinearBase):
    def __init__(self):

        num_vertices = 4
        ndim = 3
        name = "linear_3d"
        namespace = "Tetrahedron"
        x_description = """
X(xi) = (1/2)[v1-v0, v2-v0, v3-v0] (xi - [-1,-1,-1]^T) + v0

where v*-v0 form the columns of the matrix.
"""
        LinearBase.__init__(self, num_vertices, ndim, name, namespace, x_description)

    def get_x(self, xi):
        v = self.vertices
        A = 0.5 * Matrix(
            [
                [v[1][0] - v[0][0], v[2][0] - v[0][0], v[3][0] - v[0][0]],
                [v[1][1] - v[0][1], v[2][1] - v[0][1], v[3][1] - v[0][1]],
                [v[1][2] - v[0][2], v[2][2] - v[0][2], v[3][2] - v[0][2]],
            ]
        )

        s = Matrix(
            [-1.0, -1.0, -1.0],
        )
        x = A @ (xi - s) + v[0]

        return x


def self_test():

    geom_x = LinearTetrahedron()

    vertices_ref = (
        (-1.0, -1.0, -1.0),
        (1.0, -1.0, -1.0),
        (-1.0, 1.0, -1.0),
        (-1.0, -1.0, 1.0),
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
        (-0.2, -0.8, -0.6),
        (-2.20179e-12, -0.8, -0.6),
        (0.00913424, -0.730923, -0.420212),
        (-0.1, -0.9, -0.54),
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

    phys_nektar = (-0.0677164398704403, -0.8227307060000001, -0.5310530700000000)

    assert abs(phys_nektar[0] - phys[0]) < 1.0e-7
    assert abs(phys_nektar[1] - phys[1]) < 1.0e-7
    assert abs(phys_nektar[2] - phys[2]) < 1.0e-7

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
    return LinearTetrahedron


if __name__ == "__main__":
    self_test()
