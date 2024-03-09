import numpy as np
import scipy.linalg as sp
import numba as nb
import matplotlib.pyplot as plt
from time import perf_counter


# Complexité = 2m³/3
@nb.jit(nopython=True)
def lu(A):
    m, n = np.shape(A)
    if m != n: return None, None

    U = np.copy(A); L = np.identity(m)

    for i in range(m-1):
        for j in range(i+1, m):
            L[j, i] = U[j, i]/U[i, i]
            for k in range(i, m):
                U[j, k] -= L[j, i]*U[i ,k]

    return L, U


# Pas utilisé
def cholesky(A):
    m, n = np.shape(A)
    if m != n: return None

    R = np.copy(A)

    for i in range(m):
        for j in range(i+1, m):
            for k in range(j, m):
                R[j, k] -= R[i, k]*R[i, j]/R[i, i]
        for k in range(i, m):
            R[i, k] /= np.sqrt(R[i, i])

    return R


def plot_perf(N, min_size, max_size):
    size = np.logspace(np.log10(min_size), np.log10(max_size), N, dtype=int)
    lu_perf = np.zeros(N)
    cholesky_perf = np.zeros(N)

    for i in range(N):
        A = np.random.random((size[i], size[i]))*100
        A = A @ A.T

        timer = perf_counter()
        sp.lu(A)
        lu_perf[i] = perf_counter() - timer

        timer = perf_counter()
        sp.cholesky(A)
        cholesky_perf[i] = perf_counter() - timer

    plt.figure()

    plt.title("Complexity comparison of LU vs Cholesky factorization")

    plt.xlabel("size")
    plt.ylabel("execution time [s]")

    plt.loglog(size[:-2], moving_average(lu_perf))
    plt.loglog(size[:-2], moving_average(cholesky_perf))
    plt.loglog(size[:-2], 1e-9*size[:-2]**3, linestyle="dashed")

    plt.grid(which="major", linestyle="-")
    plt.grid(which="minor", linestyle=":")

    plt.legend(["LU", "Cholesky", "$\mathcal{O}(2m^3/3)$"])

    # plt.show()
    plt.savefig("rapport/images/complcomp.svg", format="svg")

    plt.figure()

    plt.title("Complexity ratio between LU and Cholesky factorization")

    plt.xlabel("size")
    plt.ylabel("LU/Cholesky time ratio")

    plt.loglog(size[:-2], moving_average(lu_perf)/moving_average(cholesky_perf))
    plt.loglog(size, np.ones(N)*2)

    plt.grid(which="major", linestyle="-")
    plt.grid(which="minor", linestyle=":")

    plt.legend(["LU/Cholesky", "ratio = 2"])

    # plt.show()
    plt.savefig("rapport/images/complratio.svg", format="svg")


def moving_average(a, n=3):
    ret = np.cumsum(a, dtype="float64")
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


def cond_test(A, b, N):
    m, n = np.shape(A)
    if n != np.shape(b)[0]: raise Exception("Fuck")

    x = np.linalg.solve(A.T @ A, A.T @ b)
    y = A @ x

    kappa_A = np.linalg.cond(A)
    norm_y = np.linalg.norm(y, 2)
    norm_A = np.linalg.norm(A, 2)
    norm_x = np.linalg.norm(x, 2)
    norm_b = np.linalg.norm(b, 2)

    # min(_, 1) car erreurs d'arrondissement possibles,
    # rendant y plus grand que b en norme quand c'est censé être égal
    theta = np.arccos(min(norm_y/norm_b, 1))

    eta = norm_A*norm_x/norm_y

    # print(f"{kappa_A=}, {theta=}, {eta=}")
    kappa_x_b = kappa_A/(eta*np.cos(theta))
    kappa_x_A = kappa_A + kappa_A*kappa_A*np.tan(theta)/eta

    deltas = np.logspace(-12, 0, N, dtype="float64")
    d_norms = np.zeros((N, 4))
    for i in range(N):
        dA = np.random.random((m, n))*deltas[i]
        d_norms[i][0] = np.linalg.norm(dA, 2)/norm_A

        dx = np.linalg.solve(A+dA, b)-x
        d_norms[i][1] = np.linalg.norm(dx, 2)/norm_x

        db = np.random.random(n)*deltas[i]
        d_norms[i][2] = np.linalg.norm(db, 2)/norm_b

        dx = np.linalg.solve(A, b+db)-x
        d_norms[i][3] = np.linalg.norm(dx, 2)/norm_x

    d_norms = np.array(sorted(d_norms, key=lambda x: x[0]))

    plt.figure()

    plt.title("Perturbation when A is disturbed (m = %d)"%m)

    plt.xlabel("||$\delta$A||/||A||")
    plt.ylabel("||$\delta$x||/||x||")

    plt.loglog(d_norms[:, 0], d_norms[:, 1])
    plt.loglog(d_norms[:, 0], d_norms[:, 0]*kappa_x_A)

    plt.grid(which="major", linestyle="-")
    plt.grid(which="minor", linestyle=":")

    plt.legend(["relative solution perturbation",
                "perturbation upper bound ($\kappa_A$*||$\delta$A||/||A||)"])

    plt.savefig("rapport/images/condA%d.svg"%m, format="svg")
    # plt.show()

    d_norms = np.array(sorted(d_norms, key=lambda x: x[2]))

    plt.figure()

    plt.title("Perturbation when b is disturbed (m = %d)"%m)

    plt.xlabel("||$\delta$b||/||b||")
    plt.ylabel("||$\delta$x||/||x||")

    plt.loglog(d_norms[:, 2], d_norms[:, 3])
    plt.loglog(d_norms[:, 2], d_norms[:, 2]*kappa_x_b)

    plt.grid(which="major", linestyle="-")
    plt.grid(which="minor", linestyle=":")

    plt.legend(["relative solution perturbation",
                "perturbation upper bound ($\kappa_b$*||$\delta$b||/||b||)"])

    plt.savefig("rapport/images/condb%d.svg"%m, format="svg")
    # plt.show()


if __name__ == "__main__":
    # # A = np.array([[1, 1, 1], [5, 9, 7], [1, 7, 68]], dtype="float64")
    # lu(np.array([[1, 1], [2, 1]]))
    # A = np.random.random((4, 4))*100

    # L, U = lu(A)

    # print(f"{A=}\n")

    # print(f"{L=}")
    # print(f"{U=}\n")

    # print(f"{L@U=}")

    # # B = np.array([[5, 0, 0], [9, 7, 0], [12, 3, 5]], dtype="float64")
    # B = np.copy(A)
    # B = B @ B.T

    # R = cholesky(B)
    # R2 = np.linalg.cholesky(B)

    # print(f"{B=}")
    # print(f"{R=}")
    # print(f"{R2=}")

    plot_perf(100, 5, 5000)

    size = 100
    cond_test(np.random.random((size, size)),
              np.random.random(size), 1000)
    
    size = 2
    cond_test(np.random.random((size, size)),
              np.random.random(size), 1000)
