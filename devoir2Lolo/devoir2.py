import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import scipy.linalg as la

@nb.njit(fastmath=True, cache=True)
def decomp_lu(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n-1):
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]
    return L, U

def decomp_lu2(A):
    n = A.shape[0]
    L = np.eye(n)
    U = A.copy()
    for i in range(n-1):
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]
    return L, U


def time_tests_lu(n=3):
    times = []
    sizes = [10]
    sizes += [i for i in range(50, 1001, 50)]
    decomp_lu(np.random.rand(3, 3))  
    for size in sizes:
        A = np.random.rand(size, size)
        time_lu = 0
        for i in range(n):
            start = time.process_time()
            decomp_lu(A)  
            end = time.process_time()
            time_lu += end - start
        times.append(time_lu / n)
    

    times2 = []
    for size in sizes:
        A = np.random.rand(size, size)
        time_lu = 0
        for i in range(n):
            start = time.process_time()
            decomp_lu2(A)
            end = time.process_time()
            time_lu += end - start
        times2.append(time_lu / n)
    
    plt.figure(figsize=(10, 5))
    plt.loglog(sizes, times,label='LU Numba')
    plt.loglog(sizes, times2,label='LU no Numba')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**2 for i in range(len(sizes))], label='O(n^2)', linestyle='--')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**3 for i in range(len(sizes))], label='O(n^3)', linestyle='--')
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.title('Time tests of decomp_lu(A) according to matrix size from 10 to 1000 with A nxn')
    plt.legend()
    plt.grid()
    plt.savefig('devoir2/images/lu_time_tests.svg')  # Change the path and file extension to .svg
    

time_tests_lu(1)
    
def compare_LU_cholesky(n=3):
    times = []
    sizes = [10]
    sizes += [i for i in range(50, 1001, 50)]
    decomp_lu(np.random.rand(3, 3))  
    for size in sizes:
        A = np.random.rand(size, size)
        time_lu = 0
        for i in range(n):
            start = time.process_time()
            decomp_lu(A)  
            end = time.process_time()
            time_lu += end - start
        times.append(time_lu / n)
    
    times2 = []
    for size in sizes:
        A = np.random.rand(size, size)
        A = A.T @ A
        time_lu = 0
        for i in range(n):
            start = time.process_time()
            np.linalg.cholesky(A)
            end = time.process_time()
            time_lu += end - start
        times2.append(time_lu / n)
    
    plt.figure(figsize=(10, 5))
    plt.loglog(sizes, times,label='LU')
    plt.loglog(sizes, times2,label='Cholesky')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**2 for i in range(len(sizes))], label='O(n^2)', linestyle='--')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**3 for i in range(len(sizes))], label='O(n^3)', linestyle='--')
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.title('Time tests of decomp_lu(A) and np.linalg.cholesky(A) according to matrix size from 10 to 1000 with A nxn')
    plt.legend()
    plt.grid()
    plt.savefig('devoir2/images/lu_cholesky_time_tests.svg')    

compare_LU_cholesky(1)



