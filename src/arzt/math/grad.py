import numpy
import numpy as np
from numpy import array, ndarray, hstack, diagflat
from numpy.testing import assert_allclose


class Func:

    def __init__(self, shape: ndarray):
        self.shape: ndarray = shape

    def __call__(self, *args, **kwargs):
        x = args[0]
        return self.forward(x)

    def forward(self, x: ndarray) -> ndarray:
        assert len(x) == self.shape[0]
        return x

    def grad(self, x) -> ndarray:
        return np.zeros([self.shape[1], self.shape[0]])

    def emp_grad(self, x) -> ndarray:
        assert len(x) == self.shape[0], f"{len(x)} != {self.shape[0]}"
        e = 0.000001
        eps = np.zeros(array(len(x)))
        grad = np.zeros([self.shape[1], self.shape[0]])
        for i in range(0, self.shape[0]):
            eps[i] = e
            y_0 = self.forward(x - eps)
            y_1 = self.forward(x + eps)
            eps[i] = 0.0
            diff = y_1 - y_0
            g = diff / (2 * e)
            grad[:, i] = g
        return grad

    def assert_grad(self, x: ndarray):
        g1 = self.grad(x)
        g2 = self.emp_grad(x)
        assert_allclose(g1, g2)


class Const(Func):

    def __init__(self, val: ndarray):
        super().__init__(array([0, len(val)]))
        self.val = val.copy()

    def forward(self, x):
        return self.val

    def grad(self, x):
        return numpy.zeros([self.shape[1], self.shape[0]])


class Mul(Func):

    def __init__(self, size: int):
        super().__init__(array([2 * size, size]))
        self.size = size

    def forward(self, x: ndarray) -> ndarray:
        assert len(x) == self.shape[0]
        x1, x2 = np.split(x, [self.size])
        y = x1 * x2
        # y = [x0 * x2, x1 * x3]
        return y

    def grad(self, x) -> ndarray:
        x1, x2 = np.split(x, [self.size])
        a = diagflat(x2)
        b = diagflat(x1)
        grad = hstack([a, b])
        return grad


class Identity(Func):

    def __init__(self, size: int):
        super().__init__(array([size, size]))

    def forward(self, x):
        return x

    def grad(self, x):
        return numpy.eye(int(self.shape[0]))


class Concat(Func):
    def __init__(self, f: Func, g: Func):
        size_in = f.shape[0] + g.shape[0]
        size_out = f.shape[1] + g.shape[1]
        super().__init__(shape=array([size_in, size_out]))
        self.f = f
        self.g = g

    def forward(self, x):
        x1, x2 = numpy.split(x, [self.f.shape[0]])
        y1, y2 = self.f(x1), self.g(x2)
        y = hstack([y1, y2])
        return y

    def grad(self, x) -> ndarray:
        x1, x2 = numpy.split(x, [self.f.shape[0]])
        grad = super().grad(x)
        fin = self.f.shape[0]
        fout = self.f.shape[1]
        grad1 = self.f.grad(x1)
        grad2 = self.g.grad(x2)
        grad[:fout, :fin] = grad1
        grad[fout:, fin:] = grad2
        return grad


class Pointwise(Func):
    def __init__(self, size: int, f, grad):
        super().__init__(array([size, size]))
        self.f = f
        self.grad = grad

    def forward(self, x: ndarray) -> ndarray:
        return self.f(x)

    def grad(self, x: ndarray) -> ndarray:
        return diagflat(self.grad(x))


class Sin(Func):
    def __init__(self, size: int):
        super().__init__(array([size, size]))

    def forward(self, x: ndarray) -> ndarray:
        return numpy.sin(x)

    def grad(self, x: ndarray) -> ndarray:
        return diagflat(np.cos(x))


class Cos(Func):
    def __init__(self, size: int):
        super().__init__(array([size, size]))


class Chain(Func):
    def __init__(self, inner: Func, outer: Func):
        super().__init__(array([inner.shape[0], outer.shape[1]]))
        assert inner.shape[1] == outer.shape[0]
        self.inner = inner
        self.outer = outer

    def forward(self, x: ndarray) -> ndarray:
        return self.outer(self.inner(x))

    def grad(self, x) -> ndarray:
        outer_grad = self.outer.grad(self.inner(x))
        inner_grad = self.inner.grad(x)
        res = np.matmul(outer_grad, inner_grad)
        return res


class MatMul(Func):
    def __init__(self, l: int, m: int, n: int):
        super().__init__(shape=array([l * m + m * n, l * n]))
        self.l = l
        self.m = m
        self.n = n

    def forward(self, x: ndarray) -> ndarray:
        a, b = np.split(x, [self.l * self.m])
        a = a.reshape(self.l, self.m)
        b = b.reshape(self.m, self.n)
        y = a @ b
        return y.ravel()

    def grad(self, x) -> ndarray:
        grad = np.zeros([self.shape[1], self.shape[0]])
        x1, x2 = np.split(x, [self.l * self.m])
        x1, x2 = x1.reshape(self.l, self.m), x2.reshape(self.m, self.n)
        lm = self.l * self.m
        grad[:, lm:] = np.kron(x1, np.eye(self.n))
        grad[:, :lm] = np.kron(np.eye(self.l), x2.T)
        return grad

    def grad_timeit(self, x) -> ndarray:
        grad = np.zeros([self.shape[1], self.shape[0]])
        x1, x2 = np.split(x, [self.l * self.m])
        x1, x2 = x1.reshape(4, 3), x2.reshape(3, 5)
        n = self.n
        m = self.m
        l = self.l
        for i in range(self.l):
            ni = i * n
            mi = i * m
            grad[ni : ni + n, mi : mi + m] = x2.T
        lm = l * m
        grad[:, lm:] = np.kron(x1, np.eye(n))
        return grad
