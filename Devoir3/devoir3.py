import numpy as np
import scipy.linalg as sp
import numba as nb

import numba as nb
import numpy as np

@nb.jit(nopython=True)
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


@nb.jit(nopython=True)
def mult_vec(a, b):
	m, n = np.shape(a)[0], np.shape(b)[0]
	assert n == m

	sum = 0

	for i in range(n):
		sum += a[i]*b[i]

	return sum


@nb.jit(nopython=True)
def sign(x, eps=1e-12):
	norm = np.abs(x)
	if norm > eps:
		return x/norm
	
	return 1

@nb.jit(nopython=True)
def to_matrix(a):
	n = np.shape(a)[-1]
	ret = np.empty((1, n), dtype="complex")

	for i in range(n):
		ret[0, i] = a[i]

	return ret


@nb.jit(nopython=True)
def round_matrix(A, eps):
	m, n = np.shape(A)
	for i in range(m):
		for j in range(n):
			if A[i, j].imag < eps:
				A[i, j] = complex(A[i, j].real, 0)
			if A[i, j].real < eps:
				A[i, j] = complex(0, A[i, j].imag)


@nb.jit(nopython=True)
def subdiag_small(A, eps):
	m = np.shape(A)[0]
	for i in range(1, m):
		if np.abs(A[i, i-1]) > eps:
			return False
			
	return True


@nb.jit(nopython=True)
def hessenberg(A, Q):
	n = np.shape(A)[0]
	v = np.copy(A)

	for i in range(n-2):
		v[i+1:, i] = A[i+1:, i].copy()
		x = v[i+1:, i]
		norm_x = np.abs(np.sqrt(mult_vec(x.conjugate(), x)))
		x[0] += norm_x*sign(x[0])
		norm_x = np.abs(np.sqrt(mult_vec(x.conjugate(), x)))
		x /= norm_x if norm_x > 1e-12 else 1

		x_mat = to_matrix(x)
		x_mat_conj = x_mat.conjugate()

		A[i+1:, i:] -= mult(2*x_mat.T, mult(x_mat_conj, A[i+1:, i:]))
		A[:, i+1:] -= 2*mult(mult(A[:, i+1:], x_mat.T), x_mat_conj)
	
	Q[...] = np.identity(n, dtype="complex")
	for i in range(n-3, -1, -1):
		x = v[i+1:, i]
		x_mat = to_matrix(x)
		x_mat_conj = x_mat.conjugate()

		Q[i+1:, i+1:] -= 2*mult(x_mat.T, mult(x_mat_conj, Q[i+1:, i+1:]))


@nb.jit(nopython=True)
def givens(a, b):
	phi = np.angle(b)-np.angle(a)
	comp = np.exp(1j*phi)
	return np.abs(a)/np.sqrt(np.abs(a)**2 + np.abs(b)**2),\
		np.abs(b)/np.sqrt(np.abs(a)**2 + np.abs(b)**2), comp


@nb.jit(nopython=True)
def step_qr(H, Q, m):
	c = np.empty(m-1, dtype="complex")
	s = np.empty(m-1, dtype="complex")
	t = np.empty(m-1, dtype="complex")
	for i in range(m-1):
		c[i], s[i], t[i] = givens(H[i, i], H[i+1, i])
		g_star = np.array([[c[i], s[i]*np.conj(t[i])],
					 	   [-s[i]*t[i], c[i]]], dtype="complex")
		H[i:i+2, i:m] = mult(g_star, H[i:i+2, i:m])

	for i in range(m-1):
		g = np.array([[c[i], -s[i]*np.conj(t[i])],
					  [s[i]*t[i], c[i]]], dtype="complex")
		H[:i+2, i:i+2] = mult(H[:i+2, i:i+2], g)

		Q[:m, i:i+2] = mult(Q[:m, i:i+2], g)


@nb.jit(nopython=True)
def wilkinson_shift(B):
	a = B[0, 0]; b = B[0, 1]; c = B[1, 0]; d = B[1, 1]
	sqrt_rho = np.sqrt(a**2 + d**2 + 4*b*c - 2*a*d)
	aplusd = a+d
	l1 = (aplusd-sqrt_rho)/2; l2 = (aplusd+sqrt_rho)/2
	return l1 if np.abs(l1-d) < np.abs(l2-d) else l2

@nb.jit(nopython=True)
def step_qr_shift(H, Q, m, eps):
	shift = wilkinson_shift(H[m-2:m, m-2:m])

	for i in range(m):
		H[i, i] -= shift
	step_qr(H, Q, m)
	for i in range(m):
		H[i, i] += shift

	return m-(1 if np.abs(H[m-1, m-2]) < eps else 0)


@nb.jit(nopython=True)
def solve_qr(A, use_shifts, eps, max_iter):
	k = -1
	n = np.shape(A)[0]
	Q = np.copy(A)
	hessenberg(A, Q)

	if use_shifts:
		m = n
		for i in range(max_iter):
			if m > 1:
				m = step_qr_shift(A, Q, m, eps)
			else:
				k = i
				break
	else:
		for i in range(max_iter):
			step_qr(A, Q, n)
			if subdiag_small(A, eps):
				k = i
				break

	return Q, k


if __name__ == "__main__":
	np.set_printoptions(precision=1)

	# A = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype="complex")
	# A = np.array([[1+1j, 2+8j, 3+6j], [4, 5+3j, 6], [7+7j, 8, 9]], dtype="complex")
	# A = np.array([[1+1j]], dtype="complex")
	A = np.asarray(np.random.rand(5, 5) * 10, dtype="complex") +\
		np.asarray(np.random.rand(5, 5) * 10, dtype="complex")*1j
	B = A.copy()
	n = np.shape(A)[0]
	
	
	Q = np.empty((n, n), dtype="complex")
	Q, k = solve_qr(A, True, 1e-12, 1000)
	# Q, k2 = solve_qr(A, False, 1e-12, 1000)
	# hessenberg(A, Q)
	# round_matrix(A, 1e-12)

	print(np.allclose(mult(Q, Q.conjugate().T), np.identity(n)))
	print(np.allclose(B, mult(mult(Q, A), Q.conjugate().T)))
	print(mult(Q, Q.conjugate().T))
	# print(A)
	# print(sp.schur(B, output='complex')[0])
	# print(mult(mult(Q, A), Q.conjugate().T))