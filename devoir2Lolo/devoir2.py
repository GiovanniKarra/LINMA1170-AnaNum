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
    


def time_tests_lu(n=3):
    sizes = np.logspace(1, np.log10(2000), num=40, dtype=int)
    decomp_lu(np.random.rand(3, 3)) 
    A = [0]*len(sizes)
    times = np.zeros(len(sizes))
    times2 = np.zeros(len(sizes))
    for i in range(len(sizes)):
        B = np.random.random((sizes[i], sizes[i]))
        A[i] = B.T @ B
        print(sizes[i])
        for j in range(n):
            times[i] -= time.perf_counter()/n
            decomp_lu(A[i])  
            times[i] += time.perf_counter()/n
            times2[i] -= time.perf_counter()/n
            la.lu(A[i])
            times2[i] += time.perf_counter()/n
            
        
    

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
    plt.loglog(sizes, times2,label='Scipy LU')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**2 for i in range(len(sizes))], label='O(n^2)', linestyle='--')
    plt.loglog(sizes, [10**-5 * (sizes[i]/sizes[0])**3 for i in range(len(sizes))], label='O(n^3)', linestyle='--')
    plt.xlabel('Size')
    plt.ylabel('Time')
    plt.title('Time tests of decomp_lu(A) according to matrix size from 10 to 10000 with A nxn')
    plt.legend()
    plt.grid()
    plt.savefig('devoir2Lolo/images/lu_time_tests.pdf')  # Change the path and file extension to .svg
    print("LU time tests done")
    return times, A
    

times, A = time_tests_lu(1)
    
def compare_LU_cholesky(n=3, times = None, A=None):
    sizes = np.logspace(1, np.log10(2000), num=40, dtype=int)
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
                la.lu(A[i])  
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
    plt.title('Time tests of scipy.linalg.lu(A) and np.linalg.cholesky(A) according to matrix size from 10 to 10000 with A nxn')
    plt.legend()
    plt.grid()
    plt.savefig('devoir2Lolo/images/lu_cholesky_time_tests.pdf')    

    ratio = [times[i]/times2[i] for i in range(len(sizes))]
    print("Ratio tends to : ", ratio[-1])

    plt.figure(figsize=(10, 5))
    plt.loglog(sizes, ratio)
    #Add Red line for 2 
    plt.axhline(y=2, color='r', linestyle='--')
    plt.xlabel('Size')
    plt.ylabel('Ratio')
    plt.title('Ratio of time between decomp_lu(A) and np.linalg.cholesky(A) according to matrix size from 10 to 10000 with A nxn')
    plt.grid()
    plt.savefig('devoir2Lolo/images/lu_cholesky_ratio.pdf')  # Change the path and file extension to .svg

    

compare_LU_cholesky(1,A=A)

def condition() :
    A = np.random.randn(2,2)
    b = np.random.randn(2)
    x = np.linalg.solve(A.T@A, A.T@b)
    kappa = np.linalg.cond(A)
    p = 1000
    delta = np.zeros((p,2))
    for k in range(p):
        Ap = A + 1e-10 * np.random.randn(2,2)
        xp = np.linalg.solve(Ap.T @ Ap, Ap.T @ b)
        delta[k,:] = ((xp - x) / np.linalg.norm(x)) / (np.linalg.norm(Ap - A) / np.linalg.norm(A))
    fig,ax = plt.subplots()
    ax.scatter(delta[:,0], delta[:,1])
    circle = plt.Circle((0.0,0.0), kappa, fill=False)
    ax.add_patch(circle)
    plt.title('Condition number of A')
    plt.savefig('devoir2Lolo/images/condition_A.pdf') 
    
    print(f'{kappa = }')
    print(f'{np.max(np.linalg.norm(delta, axis=1)) = }')

    deltab = np.zeros((p,2))
    for k in range(p):
        bp = b + 1e-10 * np.random.randn(2)
        xp = np.linalg.solve(A.T @ A, A.T @ bp)
        deltab[k,:] = ((xp - x) / np.linalg.norm(x)) / (np.linalg.norm(bp - b) / np.linalg.norm(b))
    fig,ax = plt.subplots()
    ax.scatter(deltab[:,0], deltab[:,1])
    circle = plt.Circle((0.0,0.0), kappa, fill=False)
    ax.add_patch(circle)
    #Add title
    plt.title('Condition number of B')
    plt.savefig('devoir2Lolo/images/condition_b.pdf')
    plt.show()

condition()




