import numpy as np
import numba as nb

from bsplines import B, generate_heart_points, draw_curve


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

#@nb.jit(nopython=True, parallel=True)
def lstsq(A, B):
    m, n = np.shape(A)

    Q, R = qr(A)

    B_ = Q.T @ B

    X = np.zeros(np.shape(B_), dtype="float64")

    for i in range(n-1, -1, -1):
        sum = 0
        for j in nb.prange(i+1, n):
            sum += R[i, j]*X[j]
        X[i] = (B_[i]-sum)/R[i, i]

    return X


def give_control_points(P, n, p=3):
    m = np.shape(P)[0]
    t = np.zeros(m)
    P_diff = [np.sqrt((P[i]-P[i-1]) @ (P[i]-P[i-1])) for i in range(1, m)]
    D = np.sum(P_diff)
    for i in range(1, m):
        t[i] = t[i-1] + P_diff[i-1]/D

    T = np.zeros(n+p+1)
    for i in range(4):
        T[i] = 0
        T[-i-1] = 1
    for i in range(1, n-3):
        d = m/(n-3)
        j = int(np.floor(i*d))
        alpha = i*d - j
        T[i+3] = (1-alpha)*t[j-1] + alpha*t[j]

    A = np.array([[B(i, p, t[j], T) for i in range(n)] for j in range(m)])
    
    X = lstsq(A, P)

    return X, T


if __name__ == "__main__":
    p = np.transpose(generate_heart_points())

    P, T = give_control_points(p, 15)

    draw_curve(P, T, P_real=p)