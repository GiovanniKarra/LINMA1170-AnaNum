import time
import numpy as np
import numba as nb
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    SAMPLE_SIZE = 10
    sizes = np.linspace(1, 2000, SAMPLE_SIZE)
    times = np.zeros(SAMPLE_SIZE)

    mult(np.array([[0, 1], [0, 1]]), np.array([[1, 0], [1, 0]]))

    for i, s in enumerate(sizes):
        s = int(s)
        A = np.random.randint(-20, 20, size=(s, s))
        B = np.random.randint(-20, 20, size=(s, s))

        start = time.time()
        mult(A, B)
        delta = time.time() - start

        #print(f"size {s} done !")

        times[i] = delta

    plt.figure()

    plt.plot(sizes, times)
    #plt.plot(sizes, times/np.power(sizes, 3))

    print(100000000000*times/np.power(sizes, 3))

    plt.show()


