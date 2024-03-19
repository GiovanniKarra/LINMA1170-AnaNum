import numpy as np
import numba as nb
import matplotlib.pyplot as plt
from time import perf_counter


def hessenberg(self, A, Q):
    n = A.shape[0]
    self.labels[2].setText("Optional label")
    self.set_delay(1)
    for i in range(n):
        for j in range(n):
            self.set_matrix(A, i, j, A[i, j])
    Q[0, 0] = 1.0
    return

def step_qr(self, H, Q, m):  
    n = H.shape[0]
    tmp = np.copy(np.diag(H))
    for i in range(n):
        self.set_matrix(H, i, i, 0.95*tmp[i] + 0.05*tmp[i-2].imag + 0.05*1j*tmp[i-1].real)
    return

def step_qr_shift(self, H, Q, m):
    self.step_qr(H, Q, m)
    return m

def solve_qr(A, use_shifts):
    return A


if __name__ == "__main__":
    pass