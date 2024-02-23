import matplotlib.pyplot as plt
import numpy as np

def B(i, p, t, T):
    if p == 0: return 1 if T[i] <= t and t < T[i+1] else 0

    return (B(i, p-1, t, T)*(t-T[i])/(T[i+p]-T[i]) if T[i+p]!=T[i] else 0) +\
        (B(i+1, p-1, t, T)*(T[i+1+p]-t)/(T[i+1+p]-T[i+1]) if T[i+1+p]!=T[i+1] else 0)


def draw_splines(T, p=3):
    n = len(T)

    t = np.linspace(T[0], T[-1], 1000)

    plt.figure()
    plt.xticks(T)

    for i in range(n-p-1):
        plt.plot(t, [B(i, p, ti, T) for ti in t])

    plt.show()


def draw_curve(P, T, deg=3, P_real=None):
    t = np.linspace(0, 1, 1000)
    n = len(P)

    plt.figure()

    plt.xlabel("x"); plt.ylabel("y")

    plt.title(f"n = {len(P)}" + f", m = {len(P_real)}" if P_real is not None else "")

    plt.grid(which="major", linestyle=":")

    if P_real is not None:
        plt.scatter(P_real[:, 0], P_real[:, 1], c="b", s=20)
    plt.scatter(P[:, 0], P[:, 1], c="r")

    p = np.array([np.sum([B(i, deg, ti, T)*P[i] for i in range(n)], axis=0) for ti in t])
    plt.plot(p[:, 0], p[:, 1], c="lime")

    plt.legend(([] if P_real is None else ["Original points"]) + ["Control points", "B-spline curve"])

    plt.show()


