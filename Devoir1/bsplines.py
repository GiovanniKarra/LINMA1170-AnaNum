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
    if P_real is not None:
        plt.scatter(P_real[:, 0], P_real[:, 1], c="b")
    plt.scatter(P[:, 0], P[:, 1], c="r")

    p = np.array([np.sum([B(i, deg, ti, T)*P[i] for i in range(n)], axis=0) for ti in t])
    plt.plot(p[:, 0], p[:, 1])

    plt.show()


def generate_heart_points():
    t = np.linspace(0, 2 * np.pi, 50)
    x = 16 * np.sin(t) ** 3
    y = 13 * np.cos(t) - 5 * np.cos(2 * t) - 2 * np.cos(3 * t) - np.cos(4 * t)
    return x, y


# plot les points générés par la fonction generate_heart_points
def plot_heart():
    x, y = generate_heart_points()
    plt.plot(x, y, 'ro')



if __name__ == "__main__":
    pass
