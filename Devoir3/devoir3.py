import numpy as np
import scipy.linalg as sp
import numba as nb
import matplotlib.pyplot as plt
from time import perf_counter


#@nb.jit(nopython=True)
def hessenberg(A, Q):
	n = np.shape(A)[0]
	for i in range(n-2):
		x = A[i+1:n, i][np.newaxis].T
		x[0, 0] += np.sqrt(x.conjugate().T@x)[0, 0]*np.sign(x[0, 0])
		x /= np.sqrt(x.conjugate().T@x)

		A[i+1:n, i:n] -= 2*x @ (x.conjugate().T@A[i+1:n, i:n])
		A[:n, i+1:n] -= 2*(A[:n, i+1:n]@x)@x.conjugate().T
		
		# un peu chelou
		v = np.array([x[j-i-1, 0] if (j > i and j < n) else 0 for j in range(n)], dtype="complex")[np.newaxis].T
		Q[i+1:n, i:n] @= np.identity(n,dtype="complex") - 2 * v @ v.conjugate().T

def givens(a, b):
	return 0, 0

def step_qr(H, Q, m):
	n = np.shape(A)[0]
	c = np.empty(n-1, dtype="complex")
	s = np.empty(n-1, dtype="complex")
	for i in range(n-1):
		c[i], s[i] = givens(H[i, i], H[i+1, i])
		H[i:i+1, i:n] = np.array([[c[i], -s[i]], [s[i], c[i]]], dtype="complex")@H[i:i+1, i:n]

	for i in range(n-1):
		H[:i+1, i:i+1] @= np.array([[c[i], s[i]], [-s[i], c[i]]], dtype="complex")

def step_qr_shift(H, Q, m):
	
	return m

def solve_qr(A, use_shifts):
	return A


if __name__ == "__main__":
	A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="complex")
	# A = np.random.rand(5, 5) * 10
	n = np.shape(A)[0]
	Q = np.empty((n, n), dtype="complex")

	hessenberg(A, Q)
	print(A)
	print(Q)
	# print(sp.hessenberg(A))