import numpy as np
import matplotlib.pyplot as plt


def relative_error(x0, x): return np.abs(x0-x)/np.abs(x0)


eps=np.finfo(np.double).eps
print("Машинная точность:", eps)

print(1 < np.inf)


def plot_error(x0,err):
    mask=np.logical_and(err>0,err<np.inf)
    plt.loglog(x0[mask],err[mask],".k")
    plt.loglog(x0,[eps]*len(err),"--r") # машинная точность для сравнения
    plt.xlabel("$Значение\;аргумента$")
    plt.ylabel("$Относительная\;погрешность$")
    plt.show()


def f_sqrt_sqr(x, n=52):
    for k in range(n): x=np.sqrt(x)
    for k in range(n): x=x*x
    return x


def my_sqrt_sqr(x, n = 10):
    for k in range(n):
        x = np.exp(2 * np.log(x))
        x = np.exp(0.5 * np.log(x))
    return x

x0=np.logspace(-4,4,100)
x = my_sqrt_sqr(x0)
err = relative_error(x0, x)
plot_error(x0, err)