import unittest

import numpy
import numpy as np
from numpy import array
from numpy.testing import assert_almost_equal

from arzt.math.grad import Identity, Const, Mul, Chain, Concat, Sin, MatMul, ReLU


class TestGrad(unittest.TestCase):

    def test_identity(self):
        id = Identity(2)
        x = array([1.0, 2.0], dtype=numpy.float32)
        y = id(x)
        assert_almost_equal(x, y)
        id.assert_grad(x)

    def test_const(self):
        val = array([1, 2, 3, 4])
        const = Const(val=val)
        x = array([])
        y = const(x)
        assert_almost_equal(y, val)
        const.assert_grad(x)

    def test_mul(self):
        mul = Mul(3)
        x = array([1, 2, 3, 4, 5, 6])
        y = mul(x)
        assert_almost_equal(y, array([4, 10, 18]))
        mul.assert_grad(x)

    def test_concat(self):
        conc = Concat(f=Identity(2), g=Mul(1))
        x = array([1, 2, 3, 4])
        y = conc(x)
        assert_almost_equal(y, [1, 2, 12])
        conc.assert_grad(x)

    def test_sin(self):
        x = numpy.random.randn(10)
        sin = Sin(10)
        y = sin(x)
        y_exp = np.sin(x)
        assert_almost_equal(y, y_exp)
        sin.assert_grad(x)

    def test_chain(self):
        inner = Sin(4)
        outer = Identity(4)
        chain = Chain(inner, outer)
        x = array([1, 2, 3, 4])
        y = chain(x)
        assert_almost_equal(y, np.sin(array([1, 2, 3, 4])))
        chain.assert_grad(x)

    def test_chain_2(self):
        inner = Sin(4)
        outer = Mul(2)
        chain = Chain(inner, outer)
        x = array([1, 2, 3, 4])
        y = chain(x)
        y_exp = outer(inner(x))
        assert_almost_equal(y, y_exp)
        chain.assert_grad(x)

    def test_mat_mul(self):
        a = np.random.randn(4, 3)
        b = np.random.randn(3, 5)
        y_exp = (a @ b).ravel()

        mul = MatMul(4, 3, 5)
        x = np.hstack([a.ravel(), b.ravel()])
        y = mul.forward(x)

        assert_almost_equal(y, y_exp)
        mul.assert_grad(x)

    def test_relu(self):
        x = array([4, -4, 5, -0.01, -1])
        relu = ReLU(5)
        y = relu.forward(x)
        y_exp = array([4, 0, 5, 0, 0])
        assert_almost_equal(y, y_exp)
        relu.assert_grad(x)

    def test_net(self):
        x = np.random.randn(1, 5)
        w1 = np.random.randn(5, 4)
        y_exp = (x @ w1).ravel()
        concat1 = Concat(Const(x.ravel()), Identity(20))
        mul_1 = Chain(concat1, MatMul(1, 5, 4))
        y = mul_1(w1.ravel())
        assert_almost_equal(y_exp, y)


if __name__ == "__main__":
    unittest.main()
