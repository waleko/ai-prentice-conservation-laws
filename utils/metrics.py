import numpy as np
import numba

@numba.njit()
def my_abs(x):
    if x < 0:
        return -x
    else:
        return x 

@numba.njit()
def normalize_angle(x, dim=2 * np.pi):
    while x > dim / 2:
        x -= dim
    while x < -dim / 2:
        x += dim
    return x

@numba.njit()
def sign(x):
    if x < 0:
        return -1.0
    else:
        return 1.0
    
@numba.njit(fastmath=True)
def circle_metric(x, y):
    d = normalize_angle(x[0] - y[0])
    g = sign(d)
    return my_abs(d), np.array([g])

@numba.njit(fastmath=True)
def circle_metric_without_grad(x, y):
    return my_abs(normalize_angle(x[0] - y[0]))

@numba.njit(fastmath=True)
def circle_in_2d_metric(x, y):
    d = x[0] - y[0]
    s = np.sin(d / 2)
    c = np.cos(x[0] - y[0])
    return 2 * my_abs(s), np.array([c * sign(s)])

def circles_and_lines_metric(circles_dims):
    n = len(circles_dims)
    circles_dims = np.array(circles_dims)
    @numba.njit()
    def func(x, y):
        dist_sqr = 0.0
        grad = np.zeros(x.shape[0])
        for i in range(n):
            d = normalize_angle(x[i] - y[i], circles_dims[i])
            grad[i] = d
            dist_sqr += d ** 2
        for i in range(n, x.shape[0]):
            d = x[i] - y[i]
            grad[i] = d
            dist_sqr += d ** 2
        dist = np.sqrt(dist_sqr)
        return dist, grad / (1e+06 + dist)
    
    return func