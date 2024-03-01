import numpy as np
import numba as nb


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


if __name__ == "__main__":
    # A = np.array([[1, 1, 1], [5, 9, 7], [1, 7, 68]], dtype="float64")
    lu(np.array([[1, 1], [2, 1]]))
    A = np.random.random((1000, 1000))*100

    L, U = lu(A)

    print(f"{A=}\n")

    print(f"{L=}")
    print(f"{U=}\n")

    print(f"{L@U=}")