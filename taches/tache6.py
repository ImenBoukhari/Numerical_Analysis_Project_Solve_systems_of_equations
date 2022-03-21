from cmath import cos
import numpy as np
import numpy.random as rnd
import scipy
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

def armijo_rule(alpha_0, x, f, f_x, grad_x, d_x, c, beta):  # d_x est la direction de descente d_x . grad_x <= 0
    # test f(x_new) \leq f(x_0) + c alpha ps{d_x}{grad_x}
    test = 1
    alpha = alpha_0
    while test:
        x_new = x + alpha * d_x;
        if f(x_new) <= f_x + c * alpha * np.dot(grad_x, d_x):
            test = 0
        else:
            alpha = alpha * beta
    return alpha
def steepest_descent_armijo(f, grad, x0, iterations, error_point, error_grad, c=0.1, L=100, beta=0.5):
    dim = np.max(np.shape(x0))
    x_list = np.zeros([dim, iterations])
    f_list = np.zeros(iterations)
    error_point_list = np.zeros(iterations)
    error_grad_list = np.zeros(iterations)
    x = x0
    x_old = x
    grad_x = grad(x)
    d_x = -grad_x
    f_x = f(x)
    alpha_0 = -(1. / L) * np.dot(d_x, grad_x) / np.power(np.linalg.norm(d_x), 2)
    h = armijo_rule(alpha_0, x, f, f_x, grad_x, d_x, c, beta)
    for i in range(iterations):
        x = x + h * d_x
        grad_x = grad(x)
        f_x = f(x)
        d_x = -grad_x
        alpha_0 = -(1. / L) * np.dot(d_x, grad_x) / np.power(np.linalg.norm(d_x), 2)
        h = armijo_rule(alpha_0, x, f, f_x, grad_x, d_x, c, beta)
        x_list[:, i] = x
        f_list[i] = f_x
        error_point_list[i] = np.linalg.norm(x - x_old)
        error_grad_list[i] = np.linalg.norm(grad_x)

        if i % 1000 == 0:
            print
            "iter={}, x={}, f(x)={}".format(i + 1, x, f(x))

        if (error_point_list[i] < error_point) | (error_grad_list[i] < error_grad):
            break
        x_old = x

    print
    "point error={}, grad error={}, iteration={}, f(x)={}".format(error_point_list[i], error_grad_list[i], i + 1, f(x))
    return {'x_list': x_list[:, 0:i], 'f_list': f_list[0:i], 'error_point_list': error_point_list[0:i],
            'error_point_list': error_point_list[0:i]}

def rosenbrock(X, a=1, b=100):
        x, y = X
        return (a - x) ** 2 + b * (y - x ** 2) ** 2


def rosenbrock_grad(X, a=1, b=100):
        x, y = X
        return np.array([
            2 * (x - a) - 4 * b * x * (y - x ** 2),
            2 * b * (y - x ** 2)
        ])
f = rosenbrock
grad = rosenbrock_grad
x0 = np.array([1.1,2.1])
error_point = 10**-10
error_grad = 10**-10
h = 10**-2
iterations = 10000
print(steepest_descent_armijo(f, grad, x0, iterations, error_point, error_grad))