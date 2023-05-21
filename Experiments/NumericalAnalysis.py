"""
__title__ = ''
__author__ = 'Hansen Wong // 王之超'
__time__ = 2021/10/28 16:20
__file__ = NumericalAnalysis.py
# code is far away from bugs with the god animal protecting
    I love animals. They taste delicious.
             ┏┓    ┏┓
            ┏┛┻━━━┛ ┻┓
            ┃         ┃
            ┃  ┳┛  ┗┳  ┃
            ┃     ┻    ┃
            ┗━┓      ┏━┛
              ┃      ┗━━━┓
              ┃  神兽保佑  ┣┓
              ┃　永无BUG！ ┏┛
                ┗┓┓┏━┳┓┏┛
                 ┃┫┫  ┃┫┫
                 ┗┻┛  ┗┻┛
"""
import numpy as np

def Lagrange(X: np.ndarray, Y: np.ndarray, x):
    y = 0.
    for i in range(len(X)):
        L = 1.
        for j in range(len(X)):
            if (j != i):
                L = L * (x - X[j]) / (X[i] - X[j])
        y = y + L * Y[i]
    return y

def Newton_Interpolation(X: np.ndarray, Y: np.ndarray, x):
    N = len(X)
    F = np.zeros([N])
    for i in range(N):
        F[i] = Y[i]
    for i in range(1, N):
        for j in range(N-1, i-1, -1):
            F[j] = (F[j] - F[j-1]) / (X[j] - X[j - i])
    I = 0
    for i in range(N):
        if i == 0:
            I += F[i]
        else:
            k = 1.
            for j in range(i):
               k = k * (x - X[j])
            I += F[i] * k

    return F, I

def linear_interpolation(pointL: np.ndarray, pointR: np.ndarray, xi):
    y = (xi - pointL[0]) / (pointR[0] - pointL[0]) * pointR[1] + (xi - pointR[0])/(pointL[0] - pointR[0]) * pointL[1]
    return y

def piecewiselinear_interploation(X: np.ndarray, Y: np.ndarray, x):
    N = len(X)
    for i in range(N-1):
        if (X[i] <= x and x <= X[i+1]):
            pointL = np.array([X[i], Y[i]])
            pointR = np.array([X[i+1], Y[i+1]])
            y = linear_interpolation(pointL, pointR, x)
            return y

def Trapezoidal(f, a, b):
    return 0.5 * (b - a) * (f(a) + f(b))

def Simpson(f, a, b):
    h = (b-a)/2.0
    c = 0.5 * (a+b)
    return h/3.0 * (f(a) + 4.0 * f(c) + f(b))

def Simpson38(f, a, b):
    h = (b - a) / 3.0
    return 3.0 / 8.0 * h * (f(a) + 3.0 * f(a+h) + 3.0*f(a+2.0*h) + f(a+3.0*h))

def Boole(f, a, b):
    h = (b - a) / 4.0
    return 2.0 / 45.0 * h * (7*f(a) + 32*f(a+h) + 12*f(a+2*h) + 32*f(a+3*h) + 7*f(a+4*h))

def Open_Newton_Cotes(f, a, b, n):
    h = (b - a)/(n + 2)
    if n == 0:
        return 2 * h * f(a + h)
    elif n == 1:
        return 1.5 * h * (f(a + h) + f(a + 2*h))
    elif n == 2:
        return 4*h / 3. * (2*f(a + h) - f(a + 2*h) + 2*f(a + 3*h))
    elif n == 3:
        return 5*h / 24. * (11*f(a + h) + f(a + 2*h) + f(a + 3*h) + 11*f(a + 4*h))

def compositeTrapezoidal(f, a, b, n):
    h = (b - a) / n
    sigma = 0.0
    for j in range(1, n):
        sigma += (2 * f(a + j * h))
    return (sigma + f(a) + f(b)) * h / 2.

def compositeSimpson(f, a, b, n):
    h = (b - a) / n
    sigma = 0.
    for j in range(1, n):
        if j % 2 == 0:
            sigma += (2 * f(a + j * h))
        else:
            sigma += (4 * f(a + j * h))
    return (sigma + f(a) + f(b)) * h / 3.

def Romberg(upper: int, a, b, f):
    """
    :param upper: the number of times of loop
    :param a: lower limits
    :param b: upper limits
    :param f: function
    :return: Romberg integration
    """
    H = b - a
    R = np.zeros([upper, upper])

    R[0][0] = 0.5 * (f(1) + f(2))
    for j in range(2, upper + 1):
        h = H / (2 ** (j - 1))
        s = 0
        for i in range(1, 2 ** (j - 2) + 1):
            s += f(1 + (2 * i - 1) * h)
        R[j - 1][0] = 0.5 * R[j - 2][0] + h * s

        for k in range(2, j + 1):
            R[j - 1][k - 1] = (4 ** (k - 1) * R[j - 1][k - 2] - R[j - 2][k - 2]) / (4 ** (k - 1) - 1)

    print(R)

def Gauss_Legendre(n: int, f):
    """
    :param n: number of points
    :return:
    """
    if n == 1:
        return 2 * f(0)
    elif n == 2:
        return f(1./np.sqrt(3)) + f(-1./np.sqrt(3))
    elif n == 3:
        return 5./9. * f(np.sqrt(.6)) + 5./9. * f(-np.sqrt(.6)) + 8./9. * f(0)

def AB2(f, h, st, ed, y0, y1, y_true):
    """
    :param f: f = dy/dt
    :param h: step
    :param st: start
    :param ed: end
    :param y0: y(0)
    :return: We store the approximated y, true y and the error of each t altogether in an array [4 * N] called
    data, with t in the first row, approximated y in the second row, true y in the third row and error in the
    fourth row.
    """
    stp = int((ed - st)/h)
    t0 = st
    t1 = st + h
    data = np.zeros([4, stp + 1])
    for i in range(stp + 1):
        data[0, i] = t0
        data[1, i] = y0
        data[2, i] = y_true(t0)
        data[3, i] = data[1, i] - data[2, i]
        y2 = y1 + h / 2 * (3 * f(t1, y1) - f(t0, y0))
        t0 += h
        t1 += h
        y0 = y1
        y1 = y2
    return data

def RKor4(f, h, st, ed, y0, y_true):
    """
    :param f: f = dy/dt
    :param h: step
    :param st: start
    :param ed: end
    :param y0: y(0)
    :return: We store the approximated y, true y and the error of each t altogether in an array [4 * N] called data, with t in the first row, approximated y in the second row, true y in the third row and error in the fourth row.
    """
    stp = int((ed - st) / h)
    t = st
    y = y0
    data = np.zeros([4, stp + 1])
    for i in range(stp + 1):
        data[0][i] = t
        data[1][i] = y
        data[2][i] = y_true(t)
        data[3][i] = data[1][i] - data[2][i]
        k1 = h * f(t, y)
        k2 = h * f(t + 0.5 * h, y + 0.5 * k1)
        k3 = h * f(t + 0.5 * h, y + 0.5 * k2)
        k4 = h * f(t + h, y + k3)
        y = y + 1.0 / 6.0 * (k1 + 2 * k2 + 2 * k3 + k4)
        t += h
    return data

def Secant(f, x0, x1, tol):
    while True:
        x2 = x1 - (f(x1)*(x1 - x0)) / (f(x1) - f(x0))
        if abs(x2 - x1) <= tol:
            return x2, abs(x2 - x1)
        else:
            x0 = x1
            x1 = x2

def power_iteration(A: np.ndarray, x0: np.ndarray, tol):
    """
    :param A: given matrix
    :param x0: first guess(non-zero)
    :param tol: tolerance
    :return: main eigenvalue
    """
    # normalised
    x0 = x0 / np.linalg.norm(x0)
    lambda0 = np.inner(
        np.matmul(A, x0).reshape(3),
        x0.reshape(3)
    )

    while True:
        # iteration
        x1 = np.matmul(A, x0)
        # normalised
        x1 = x1 / np.linalg.norm(x1)
        lambda1 = np.inner(np.matmul(A, x1).reshape(3), x1.reshape(3))
        x0 = np.copy(x1)
        # tolerance
        if abs(lambda1 - lambda0) / abs(lambda1) < tol:
            return lambda1
        else:
            lambda0 = lambda1

def Jacobi(A: np.ndarray, b: np.ndarray, x: np.ndarray, stp):
    shape = A.shape
    D = np.zeros_like(A)
    for i in range(shape[0]):
        D[i, i] = A[i, i]

    L = np.zeros_like(A)
    for i in range(shape[0]):
        for j in range(i):
            L[i, j] = A[i, j]

    U = np.zeros_like(A)
    for i in range(shape[0]):
        for j in range(i+1, shape[0]):
            U[i, j] = A[i, j]

    T = -np.dot(np.linalg.inv(D), (L + U))
    c = np.dot(np.linalg.inv(D), b)

    for i in range(stp):
        print(x)
        x = np.dot(T, x) + c

    return x

def G_S(A: np.ndarray, b: np.ndarray, x0: np.ndarray, stp):
    shape = A.shape
    D = np.zeros_like(A)
    for i in range(shape[0]):
        D[i, i] = A[i, i]

    L = np.zeros_like(A)
    for i in range(shape[0]):
        for j in range(i):
            L[i, j] = A[i, j]

    U = np.zeros_like(A)
    for i in range(shape[0]):
        for j in range(i+1, shape[0]):
            U[i, j] = A[i, j]

    G = np.dot(np.linalg.inv(D-L), U)
    f = np.dot(np.linalg.inv(D-L), b)
    for i in range(stp):
        x0 = G * x0 + f

    return x0

def Eular(f, y0, h, st, ed):
    stp = int((ed - st) / h)
    y = y0
    t = st
    print('t\ty(t)\n')
    print('{:.2f}\t{:.6f}\n'.format(t, y))

    for i in range(stp):
        y += h * f(t, y)
    t += h
    print('{:.2f}\t{:.6f}\n'.format(t, y))

