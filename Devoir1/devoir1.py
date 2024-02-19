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


@nb.jit(nopython=True)
def qr(A):
    m, n = np.shape(A)
    Q = np.copy(A)
    R = np.zeros((n, n), dtype="float64")

    for i in range(n):
        R[i, i] = np.sqrt(Q[:, i] @ Q[:, i])
        Q[:, i] /= R[i, i]
        for j in range(i+1, n):
            R[i, j] = Q[:, i] @ Q[:, j]
            Q[:, j] -= R[i, j] * Q[:, i]

    return Q, R

@nb.jit(nopython=True, parallel=True)
def lstsq(A, B):
    m, n = np.shape(A)

    Q, R = qr(A)

    B_ = Q.T @ B

    X = np.zeros(n, dtype="float64")

    for i in range(n-1, -1, -1):
        sum = 0
        for j in nb.prange(i+1, n):
            sum += R[i, j]*X[j]
        X[i] = (B_[i]-sum)/R[i, i]

    return X


if __name__ == "__main__":
    A = np.array([[1, 1, 1], [2, 3, 4], [8, 9, 16]], dtype="float64")
    
    # Q, R = qr(A)

    # print(Q)
    # print(R)

    # print(Q @ R)
    # print(Q[:, 0] @ Q[:, 1])

    B =  np.array([1, 2, 3], dtype="float64")
    X = lstsq(A, B)

    print(X)