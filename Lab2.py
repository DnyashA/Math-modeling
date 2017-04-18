import numpy as np

def relative_error(x0, x):
    return np.abs(x0-x)/np.abs(x)


def samples(k):
    samples = []
    for i in range(1, k+1):
        samples.append(np.sin(i))
    return samples


def exact_sum(k):
    return 1.0/2 * (np.sin(k) - np.cos(k)/np.tan(1.0/2) + 1/np.tan(1.0/2))


def direct_sum(x):
    summ = 0.0
    for elem in x:
        summ += elem
    return summ


def kahan_sum(x):
    summ = 0.0
    error_summ = 0.0
    for elem in x:
        y = elem - error_summ
        t = summ + y
        error_summ = (t - summ) - y
        summ = t
    return summ


def samples(n, mean, delta):
    x = np.full((2*n,), mean, dtype=np.double)
    x[:n] += delta
    x[n:] -= delta
    return np.random.permutation(x)


"""for better view"""
def exact_mean(mean):
    return mean


def exact_variance(delta):
    return delta ** 2


def direct_mean(x):
    return direct_sum(x) / len(x)


def direct_first_var(x):
    return direct_mean((x - direct_mean(x)) ** 2)


def direct_second_var(x):
    return direct_mean(x ** 2) - direct_mean(x) ** 2


def oneline_second_var(x):
    # accumulated mean
    m = x[0]
    # accumulated mean of squares
    m_sq = x[0] ** 2
    for n in range(1, len(x)):
        m = (m * (n - 1) + x[n]) / n
        m_sq = (m_sq * (n - 1) + x[n] ** 2) / n
    return m_sq - m ** 2


def relative_error(x0, x):
    return relative_error(x0, x)



def kahan_mean(x):
    return kahan_sum(x) / len(x)

def oneline_variance_kahan(x):
    n = len(x)
    m = x[0]  # мат ожидание
    m2 = (x[0] + x[1]) / 2.0  # следующее мат ожидание
    res = (x[0] - m) ** 2 + n * (m - m2) ** 2
    alpha = 0.0
    for i in range(1, n - 1):
        m = m2
        alpha = m - m2
        m2 = (m2 * (i + 1) + x[i + 1]) / (i + 2)
        res = (x[i] - m) ** 2 + alpha ** 2
    # res += alpha * n
    return res


def main(mean, delta):
    x = samples(1000000, mean, delta)

    print("Exact var, Direct first var, My var:")
    print(exact_variance(delta), direct_first_var(x), oneline_variance_kahan(x))

    print()

    print("Размер выборки:\t\t", len(x))
    print("Среднее значение:\t", exact_mean(mean))
    print("Оценка дисперсии:\t", exact_variance(delta))

    print()

    print("Ошибка среднего для встроенной функции:\t\t", relative_error(exact_mean(mean), np.mean(x)))
    print("Ошибка дисперсии для встроенной функции:\t", relative_error(exact_variance(delta), np.var(x)))

    print()

    print("Ошибка среднего для последовательного суммирования:", relative_error(exact_mean(mean), direct_mean(x)))

    print()

    print("Ошибка первой оценки дисперсии для последовательного суммирования:\t",
          relative_error(exact_variance(delta), direct_first_var(x)))
    print("Ошибка моей оценки дисперсии для последовательного суммирования:\t",
          relative_error(exact_variance(delta), oneline_variance_kahan(x)))


main(1e6, 1e-5)