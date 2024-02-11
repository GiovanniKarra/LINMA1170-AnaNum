import numpy as np
import numba as nb


# algo tourne en O(mpn)
@nb.jit(nopython=True, parallel=True)
def mult(A, B):
    m, n = np.shape(A)
    _, p = np.shape(B)

    C = np.zeros((m, p))

    for i in nb.prange(m):
        for j in nb.prange(p):
            for k in nb.prange(n):
                C[i][j] += A[i][k]*B[k][j]

    return C


@nb.jit(nopython=True, parallel=True)
def mult_vec(a, b):
    m, n = np.shape(a)[0], np.shape(b)[0]
    if m != n: raise ValueError("arguments don't have the same size")

    sum = 0

    for i in nb.prange(n):
        sum += a[i]*b[i]

    return sum


@nb.jit(nopython=True, parallel=True)
def qr(A):
    m, n = np.shape(A)
    Q = np.zeros((m, n))
    R = np.zeros((n, n))

    for j in nb.prange(n):
        v = A[:, j]
        for i in nb.prange(j):
            R[i, j] = mult_vec(Q[:, i], A[:, j])
            v -= R[i, j]*Q[:, i]
        R[j, j] = mult_vec(v, v)
        Q[:, j] = v/R[j, j]

    return Q, R


def lstsq(A, B):
    pass


if __name__ == "__main__":
    A = np.array([[1, 1], [2, 3], [8, 9]], dtype="float64")
    
    Q, R = qr(A)

    print(Q)
    print(R)

    print(mult(Q, R))
    print(mult_vec(Q[:, 0], Q[:, 1]))