import matplotlib.pyplot as plt
from time import time

from devoir1 import *
from bsplines import draw_curve


def generate_heart_points(m):
    t = np.linspace(0, 2 * np.pi, m)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x, y


def test_bsplines(generating_func, m, n):
    x, y = generating_func(m)

    P_real = np.array((x, y)).T
    P, T = give_control_points(P_real, n)

    draw_curve(P, T, P_real=P_real)


def test_qr(num=100):
    qr(np.array([[1, 2], [3, 4]], dtype="float64")) # pour compiler à l'avance

    m = np.logspace(1, 3, num)
    n = m.copy()
    times = np.zeros(num, dtype="float64")

    for i in range(num):
        A = np.random.rand(int(m[i]), int(n[i]))

        t = time()
        qr(A)
        times[i] = time()-t

    plt.figure()

    plt.title("Complexity analysis of the QR algorithm")

    plt.xlabel("m")
    plt.ylabel("execution time [s]")

    plt.loglog(m, times)
    plt.loglog(m, np.power(m, 3)/300000000, linestyle="dashed")

    plt.grid(True)

    plt.legend(["QR", "x³/300 000 000"])

    plt.show()


if __name__ == "__main__":
    # test_qr(5)

    test_bsplines(generate_heart_points, 100, 15)
    test_bsplines(generate_heart_points, 20, 15)
    test_bsplines(generate_heart_points, 15, 15)