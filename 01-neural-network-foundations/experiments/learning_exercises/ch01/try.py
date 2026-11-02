import numpy as np
import matplotlib.pyplot as plt


def function_1(x):
    return 2*x**3 + 4*x + 5


def function_2(x):
    return 3*np.exp(x) + 4*x**3 + 5**2


def numeric_diff(func, x):
    h = 1e-4
    return (func(x + h) - func(x - h)) / (2 * h)


x = np.arange(-10, 10, 0.1)

y1 = function_1(x)
y2 = function_2(x)

diff_1 = numeric_diff(function_1, x)
diff_2 = numeric_diff(function_2, x)

plt.plot(x, y1, label="function 1")
plt.plot(x, diff_1, label="derivative of function 1", linestyle="--")

plt.plot(x, y2, label="function 2")
plt.plot(x, diff_2, label="derivative of function 2", linestyle="--")

plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()