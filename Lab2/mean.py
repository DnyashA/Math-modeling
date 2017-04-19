import numpy as np
import kahan


def samples(n, mean, delta):
    x = np.full((2*n,), mean, dtype=np.double)
    x[:n] += delta
    x[n:] -= delta
    return np.random.permutation(x)


def exact_mean(mean):
    return mean


def exact_variance(delta):
    return delta ** 2


def direct_mean(x):
    return kahan.direct_sum(x) / len(x)


def direct_first_var(x):
    return direct_mean((x - direct_mean(x)) ** 2)


def direct_second_var(x):
    return direct_mean(x ** 2) - direct_mean(x) ** 2


def oneline_second_var(x):
    m = x[0]
    m_sq = x[0] ** 2
    for n in range(1, len(x)):
        m = (m * (n - 1) + x[n]) / n
        m_sq = (m_sq * (n - 1) + x[n] ** 2) / n
    return m_sq - m ** 2


def relative_error(x0, x):
    return kahan.relative_error(x0, x)


