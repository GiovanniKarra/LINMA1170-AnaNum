import scipy.linalg as sp
import numpy as np
import matplotlib.pyplot as plt
from time import perf_counter

from devoir3 import solve_qr

def plot_complexity(min, max, num, eps, max_iter):
	size = np.logspace(np.log10(min), np.log10(max), num)
	exec_time = np.empty(num, dtype=np.float64)
	exec_time_shift = np.empty(num, dtype=np.float64)
	exec_time_scipy = np.empty(num, dtype=np.float64)
	iter_num = np.empty(num, dtype=int)
	iter_num_shift = np.empty(num, dtype=int)

	for i in range(num):
		n = int(size[i])
		A = np.asarray(np.random.rand(n, n) * 100, dtype="complex") +\
			np.asarray(np.random.rand(n, n) * 100, dtype="complex")*1j

		time = perf_counter()
		_, k = solve_qr(A.copy(), False, eps, max_iter)
		exec_time[i] = perf_counter()-time

		time = perf_counter()
		_, k2 = solve_qr(A.copy(), True, eps, max_iter)
		exec_time_shift[i] = perf_counter()-time

		time = perf_counter()
		_ = sp.schur(A, output="complex")
		exec_time_scipy[i] = perf_counter()-time

		if k == -1: k = max_iter
		if k2 == -1: k2 = max_iter

		iter_num[i] = k
		iter_num_shift[i] = k2

		print("iteration %d/%d (size %d) done !" % (i+1, num, n))


	plt.figure()

	plt.title("Complexity analysis of the QR Algorithm (max_iter = %d)"%max_iter)

	plt.xlabel("matrix size")
	plt.ylabel("execution time [s]")

	plt.loglog(size, exec_time)
	plt.loglog(size, exec_time_shift)
	plt.loglog(size, exec_time_scipy)
	plt.loglog(size, 1e-9*size**3, linestyle="dashed")

	plt.grid(which="major", linestyle="-")
	plt.grid(which="minor", linestyle=":")

	plt.legend(["QR Algorithm (no shifts)", "QR Algorithm", "scipy.linalg.schur",
				"$\mathcal{O}(m^3)$"])

	# plt.show()
	plt.savefig("rapport/images/complexity%d.svg"%max_iter, format="svg")

	plt.figure()

	plt.title("Number of iterations before convergence")

	plt.xlabel("matrix size")
	plt.ylabel("Iteration count")

	plt.semilogx(size, iter_num)
	plt.semilogx(size, iter_num_shift)
	plt.semilogx(size, np.ones(num)*max_iter, linestyle="dashed", c="r")

	plt.grid(which="major", linestyle="-")
	plt.grid(which="minor", linestyle=":")

	plt.legend(["No shift", "With shift", "Max iterations allowed"])

	# plt.show()
	plt.savefig("rapport/images/itercount%d.svg"%max_iter, format="svg")


if __name__ == "__main__":
	# pour compiler
	solve_qr(np.array([[2, 1, 1], [1, 2, 1], [1, 1, 2]], dtype="complex"), True, 1e-12, 10)

	plot_complexity(10, 1000, 20, 1e-12, 1000)
	plot_complexity(10, 1000, 20, 1e-12, 5000)