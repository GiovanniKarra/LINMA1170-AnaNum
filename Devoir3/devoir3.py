import numpy as np
import scipy.linalg as sp
import numba as nb


#@nb.jit(nopython=True)
def mult(A, B):
	m, n = np.shape(A)
	n2, p = np.shape(B)

	assert n == n2

	C = np.zeros((m, p), dtype="complex")

	for i in range(m):
		for j in range(p):
			for k in range(n):
				C[i, j] += A[i, k]*B[k, j]

	return C


def sign(x):
	return x/norm if (norm := np.abs(x)) > 1e-12 else 1

#@nb.jit(nopython=True)
def hessenberg(A, Q):
	n = np.shape(A)[0]
	v = [np.copy(A[i+1:, i])[np.newaxis].T for i in range(n-2)]
	for i in range(n-2):
		v[i][0, 0] += np.sqrt(mult(v[i].conjugate().T, v[i]))[0, 0]*sign(v[i][0, 0])
		v[i] /= np.sqrt(mult(v[i].conjugate().T, v[i]))

		A[i+1:, i:] -= mult(2*v[i], mult(v[i].conjugate().T, A[i+1:, i:]))
		A[:, i+1:] -= 2*mult(mult(A[:, i+1:], v[i]), v[i].conjugate().T)
	
	Q[...] = np.identity(n, dtype="complex")
	for i in range(n-3, -1, -1):
		Q[i+1:, i+1:] -= 2*mult(v[i], mult(v[i].conjugate().T, Q[i+1:, i+1:]))


#@nb.jit(nopython=True)
def givens(a, b):
	return a/np.sqrt(a**2 + b**2), -b/np.sqrt(a**2 + b**2)


#@nb.jit(nopython=True)
def step_qr(H, Q, m):
	n = np.shape(H)[0]
	c = np.empty(n-1, dtype="complex")
	s = np.empty(n-1, dtype="complex")
	for i in range(n-1):
		c[i], s[i] = givens(H[i, i], H[i+1, i])
		H[i:i+2, i:m] = mult(np.array([[c[i], -s[i]], [s[i], c[i]]], dtype="complex"),
			H[i:i+2, i:m])

	for i in range(m-1):
		g = np.array([[c[i], s[i]], [-s[i], c[i]]], dtype="complex")
		H[:i+2, i:i+2] = mult(H[:i+2, i:i+2], g)

		Q[:, i:i+2] = mult(Q[:, i:i+2], g)


#@nb.jit(nopython=True)
def step_qr_shift(H, Q, m, eps):
	while np.abs(H[m-1, m-2]) > eps:
		delta = (H[m-2, m-2]-H[m-1, m-1])/2
		b_sq = H[m-1, m-2]**2
		denom = np.abs(delta)*np.sqrt(delta**2+b_sq)
		shift = H[m-1, m-1]-sign(delta)*b_sq/denom

		for i in range(m):
			H[i, i] -= shift
		step_qr(H, Q, m)
		for i in range(m):
			H[i, i] += shift

	return m-1


def round_matrix(A, eps):
	A[np.abs(A)<eps] = 0


#@nb.jit(nopython=True)
def solve_qr(A, use_shifts, eps, max_iter):
	k = -1
	n = np.shape(A)[0]
	Q = np.copy(A)
	hessenberg(A, Q)
	if use_shifts:
		m = n
		while m > 1:
			m = step_qr_shift(A, Q, m, eps)
	else:
		for i in range(max_iter):
			step_qr(A, Q, n)
			if np.all(np.abs(np.diag(A)) < eps):
				k = i
				break

	return Q, k


if __name__ == "__main__":
	A = np.array([[1+1j, 2+8j, 3+6j], [4, 5+3j, 6], [7+7j, 8, 9]], dtype="complex")
	B = A.copy()
	# A = np.random.rand(5, 5) * 10
	#n = np.shape(A)[0]
	
	#print(solve_qr(A, True, 1e-15, 100))
	
	solve_qr(A, True, 1e-15, 1000)
	#solve_qr(B, False, 1e-15, 1000)
	
	print(B)
	print(A)