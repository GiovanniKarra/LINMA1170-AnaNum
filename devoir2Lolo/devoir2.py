import numpy as np
import numba as nb
import time
import matplotlib.pyplot as plt
import scipy.linalg as la

@nb.jit()
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
    U = A
    for i in range(n-1):
        for j in range(i+1, n):
            L[j, i] = U[j, i] / U[i, i]
            U[j, i:] = U[j, i:] - L[j, i] * U[i, i:]
    return L, U


def time_tests_lu(n=3):
    sizes = np.logspace(1, 4, num=20, dtype=int)
    decomp_lu(np.random.rand(3, 3)) 
    A = [0]*len(sizes)
    times = np.zeros(len(sizes))
    for i in range(len(sizes)):
        B = np.random.random((sizes[i], sizes[i]))
        A[i] = B.T @ B
        print(sizes[i])
        for j in range(n):
            times[i] -= time.perf_counter()/n
            decomp_lu(A[i])  
            times[i] += time.perf_counter()/n
            
        
    

    """times2 = []
    for size in sizes:
        A = np.random.random((size, size))+np.eye(size)*5
        time_lu = 0
        for i in range(n):
            start = time.process_time()
            decomp_lu2(A)
            end = time.process_time()
            time_lu += end - start
        times2.append(time_lu / n)"""
    
    plt.figure(figsize=(10, 5))
    plt.loglog(sizes, times,label='LU Numba')
    #plt.loglog(sizes, times2,label='LU no Numba')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**2 for i in range(len(sizes))], label='O(n^2)', linestyle='--')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**3 for i in range(len(sizes))], label='O(n^3)', linestyle='--')
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.title('Time tests of decomp_lu(A) according to matrix size from 10 to 10000 with A nxn')
    plt.legend()
    plt.grid()
    plt.savefig('devoir2Lolo/images/lu_time_tests.svg')  # Change the path and file extension to .svg
    print("LU time tests done")
    return times, A
    

times, A = time_tests_lu(1)
    
def compare_LU_cholesky(n=3, times = None, A=None):
    sizes = np.logspace(1, 4, num=20, dtype=int)
    decomp_lu(np.random.rand(3, 3)) 
    
    if times is None:
        if A is None:
            A = []
        times = np.zeros(len(sizes))
        for i in range(len(sizes)):
            B = np.random.random((sizes[i], sizes[i]))
            A.append(B.T @ B)
            print(sizes[i])
            for j in range(n):
                times[i] -= time.perf_counter()/n
                np.linalg.lu(A[i])  
                times[i] += time.perf_counter()/n
    
    times2 = np.zeros(len(sizes))
    for i in range(len(sizes)):
        
        print(sizes[i])
        for j in range(n):
            times2[i] -= time.perf_counter()/n
            np.linalg.cholesky(A[i])
            times2[i] += time.perf_counter()/n
            

        
    
    plt.figure(figsize=(10, 5))
    plt.loglog(sizes, times,label='LU')
    plt.loglog(sizes, times2,label='Cholesky')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**2 for i in range(len(sizes))], label='O(n^2)', linestyle='--')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**3 for i in range(len(sizes))], label='O(n^3)', linestyle='--')
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.title('Time tests of decomp_lu(A) and np.linalg.cholesky(A) according to matrix size from 10 to 10000 with A nxn')
    plt.legend()
    plt.grid()
    plt.savefig('devoir2Lolo/images/lu_cholesky_time_tests.svg')    

    ratio = [times[i]/times2[i] for i in range(len(sizes))]

    plt.figure(figsize=(10, 5))
    plt.plot(sizes, ratio)
    plt.xlabel('Size')
    plt.ylabel('Ratio')
    plt.title('Ratio of time between decomp_lu(A) and np.linalg.cholesky(A) according to matrix size from 10 to 10000 with A nxn')
    plt.grid()
    plt.savefig('devoir2Lolo/images/lu_cholesky_ratio.svg')  # Change the path and file extension to .svg

    

compare_LU_cholesky(1,A=A)

def condition() :
    A = np.random.randn(3,3)
    A = A.T @ A
    b = np.random.randn(3)
    b = A.T @ b
    x = np.linalg.solve(A, b)
    kappa = np.linalg.cond(A)
    p = 1000
    delta = np.zeros((p,3))
    for k in range(p):
        Ap = A + 1e-10 * np.random.randn(3,3)
        xp = np.linalg.solve(Ap, b)
        delta[k,:] = ((xp - x) / np.linalg.norm(x)) / (np.linalg.norm(Ap - A) / np.linalg.norm(A))
    fig,ax = plt.subplots()
    ax.scatter(delta[:,0], delta[:,1])
    circle = plt.Circle((0.0,0.0), kappa, fill=False)
    ax.add_patch(circle)
    plt.savefig('devoir2Lolo/images/condition.svg') 
    
    print(f'{kappa = }')
    print(f'{np.max(np.linalg.norm(delta, axis=1)) = }')
    plt.show()

condition()




