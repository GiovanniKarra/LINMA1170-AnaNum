from devoir3 import *
import matplotlib.pyplot as plt
import time
import scipy as sp

A = np.array([[1,2,3],[4,5,6],[7,8,9]])
B = np.array([[1,2,3],[4,5,6],[7,8,9]])
C = mult(A,B)
D = A @ B
assert np.allclose(C,D)

def test_hessenberg(n,mult=10):
    count_false = 0
    max = [0,0]
    A = np.random.rand(4,4) + 1j * np.random.rand(4,4)
    Q = np.empty((4,4),dtype=np.complex128)
    hessenberg(A,Q)
    for i in range(1,n+1):
        s = i * mult
        if i % (n//10) == 0:
            print("We're at matrix of size",s,"x",s)
        A = np.random.rand(s,s) + 1j * np.random.rand(s,s)
        Q = np.empty((s,s),dtype=np.complex128)
        start = time.perf_counter()
        hessenberg(A,Q)
        end = time.perf_counter()
        max = [s,end-start]
        I = conj(Q) @ Q
        if not np.allclose(I,np.identity(s)):
            count_false += 1
            print("Failed")
    if count_false == 0:
        print("Tous les tests ont réussi")
    print("Temps d'exécution maximal :",max[1],"pour une matrice de taille",max[0],"x",max[0])
    return count_false

def test_solve_qr_shift(n,mult=10):
    count_false = 0
    max = [0,0]
    A = np.random.rand(8,8) + 1j * np.random.rand(8,8)
    solve_qr(A,True)
    for i in range(1,n+1):
        s = i * mult
        if i % (n//10) == 0:
            print("We're at matrix of size",s,"x",s)
        A = np.random.rand(s,s) + 1j * np.random.rand(s,s)
        A2 = A.copy()
        start = time.perf_counter()
        U,k = solve_qr(A,True,1e-12,1000)
        end = time.perf_counter()
        
        max = [s,end-start]
        I = conj(U) @ U
        if not np.allclose(I,np.identity(s))  or not np.allclose(U @ A @ conj(U),A2):
            count_false += 1
            print("Failed")
    if count_false == 0:
        print("Tous les tests ont réussi avec shift")
    print("Temps d'exécution maximal :",max[1],"pour une matrice de taille",max[0],"x",max[0])
    return count_false

def test_solve_qr_no_shift(n,mult=10):
    count_false = 0
    max = [0,0]
    A = np.random.rand(4,4) + 1j * np.random.rand(4,4)
    solve_qr(A,False)
    for i in range(1,n+1):
        s = i * mult
        if i % (n//10) == 0:
            print("We're at matrix of size",s,"x",s)
        A = np.random.rand(s,s) + 1j * np.random.rand(s,s)
        A2 = A.copy()
        start = time.perf_counter()
        U,k = solve_qr(A,False,1e-12,1000)
        end = time.perf_counter()
      
        max = [s,end-start]
        I = conj(U) @ U
        if not np.allclose(I,np.identity(s)) or not np.allclose(U @ A @ conj(U),A2):
            count_false += 1
            print("Failed")
    if count_false == 0:
        print("Tous les tests ont réussi sans shift")
    print("Temps d'exécution maximal :",max[1],"pour une matrice de taille",max[0],"x",max[0])
    return count_false


def plot_complexity_hessenberg(n):
    times = []
    logrange = np.logspace(1, n, 30,dtype=int)
    A = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    Q = np.empty((4, 4), dtype=np.complex128)
    hessenberg(A, Q)
    for i in range(30):
        s = logrange[i]
        A = np.random.rand(s, s) + 1j * np.random.rand(s, s)
        Q = np.empty((s, s), dtype=np.complex128)
        start = time.time()
        hessenberg(A, Q)
        end = time.time()
        times.append(end - start)
    plt.figure()
    plt.loglog(logrange, times, label="hessenberg")
    plt.loglog(logrange, [i**2 for i in logrange], label="n^2")
    plt.loglog(logrange, [i**3 for i in logrange], label="n^3")
    plt.xlabel("n")
    plt.ylabel("Time (s)")
    plt.title("Complexity of hessenberg")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/complexity_hessenberg_" + str(round(10**n)) + ".pdf")
    
def plot_complexity_solve_qr(n):
    times = []
    logrange = np.logspace(1, n, 30,dtype=int)
    A = np.random.rand(4, 4) + 1j * np.random.rand(4, 4)
    solve_qr(A, True)
    for i in range(30):
        s = logrange[i]
        A = np.random.rand(s, s) + 1j * np.random.rand(s, s)
        start = time.time()
        solve_qr(A, True)
        end = time.time()
        times.append(end - start)

    plt.figure()
    plt.loglog(logrange, times,label="solve_qr")
    plt.loglog(logrange, [(i/100)**2 for i in logrange], label="n^2")
    plt.loglog(logrange, [(i/100)**3 for i in logrange], label="n^3")
    plt.xlabel("n")
    plt.ylabel("Time (s)")
    plt.title("Complexity of solve_qr")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/complexity_solve_qr_" + str(round(10**n)) + ".pdf")
    
def compare_solve_qr_complexity(n) :
    logrange = np.logspace(1, n, 20,dtype=int)
    times_shift = []
    times_no_shift = []
    times_scipy = []
    for i in range(20):
        s = logrange[i]
        if s < 0 :
            print(s)
        A = np.random.rand(s, s) + 1j * np.random.rand(s, s)
        A2 = A.copy()
        A3 = A.copy()  
        start = time.time()
        solve_qr(A, True)
        end = time.time()
        times_shift.append(end - start)
        start = time.time()
        solve_qr(A2, False)
        end = time.time()
        times_no_shift.append(end - start)
        start = time.time()
        sp.linalg.schur(A3)
        end = time.time()
        times_scipy.append(end - start)
    plt.figure()
    plt.loglog(logrange, times_shift, label="Avec shift")
    plt.loglog(logrange, times_no_shift, label="Sans shift")
    plt.loglog(logrange, times_scipy, label="Scipy.schur")
    plt.loglog(logrange, [(i/100)**2 for i in logrange], label="n^2", linestyle='--')
    plt.loglog(logrange, [(i/100)**3 for i in logrange], label="n^3", linestyle='--')
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité de solve_qr")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/compare_solve_qr_complexity.pdf")
    plt.figure()
    plt.plot(logrange, [times_shift[i]/times_scipy[i] for i in range(20)], label="Avec shift")
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Ratio de complexité")
    plt.title("Ratio de complexité entre solve_qr et scipy.schur")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/compare_solve_qr_complexity_ratio.pdf")

def compare_hessenberg_complexity(n) :
    logrange = np.logspace(1, n, 20,dtype=int)
    times = []
    times_scipy = []
    for i in range(20):
        s = logrange[i]
        A = np.random.rand(s, s) + 1j * np.random.rand(s, s)
        Q = np.empty((s, s), dtype=np.complex128)
        A2 = A.copy() 
        start = time.time()
        hessenberg(A,Q)
        end = time.time()
        times.append(end - start)
        start = time.time()
        sp.linalg.hessenberg(A2)
        end = time.time()
        times_scipy.append(end - start)
    plt.figure()
    plt.loglog(logrange, times, label="Hessenberg")
    plt.loglog(logrange, times_scipy, label="Scipy.hessenberg")
    plt.loglog(logrange, [(i/100)**2 for i in logrange], label="n^2",linestyle='--')
    plt.loglog(logrange, [(i/100)**3 for i in logrange], label="n^3",linestyle='--')
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Temps d'exécution (s)")
    plt.title("Complexité de hessenberg")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/compare_hessenberg_complexity.pdf")

def compare_solve_qr_convergence(n,max_iter=1000):
    logrange = np.logspace(1, n, 20,dtype=int)
    iters_shift = []
    iters_no_shift = []
    for i in range(20):
        s = logrange[i]
        A = np.random.rand(s, s) + 1j * np.random.rand(s, s)
        A2 = A.copy()
        _, k = solve_qr(A, True,eps=1e-13,max_iter=max_iter)
        if k == -1:
            k = max_iter
        iters_shift.append(k)
        _, k = solve_qr(A2, False,eps=1e-13,max_iter=max_iter)
        if k == -1:
            k = max_iter
        iters_no_shift.append(k)
    plt.figure()
    plt.plot(logrange, iters_shift, label="Avec shift")
    plt.plot(logrange, iters_no_shift, label="Sans shift")
    
    plt.xscale('log')
    plt.axhline(y=max_iter, color='k', linestyle='--', label="Nombre maximal d'itérations")
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Nombre d'itérations")
    plt.title("Convergence de solve_qr")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/compare_solve_qr_convergence_" + str(max_iter) + ".pdf")
    plt.figure()
    plt.plot(logrange, [iters_no_shift[i]/iters_shift[i] for i in range(20)], label="Ratio de convergence")
    plt.xscale('log')
    plt.xlabel("Taille de la matrice")
    plt.ylabel("Ratio de convergence")
    plt.title("Ratio de convergence entre solve_qr avec et sans shift")
    plt.legend()
    plt.grid()
    plt.savefig("devoir3Lolo/images/compare_solve_qr_convergence_ratio_" + str(max_iter) + ".pdf")

def launch_compares(n,max_iter):
    compare_solve_qr_complexity(n)
    #compare_hessenberg_complexity(n)
    # if len(max_iter) > 0:
    #     for i in max_iter:
    #         compare_solve_qr_convergence(n,i)
    # else : 
    #     compare_solve_qr_convergence(n,max_iter)
    
c = test_hessenberg(10)
c += test_solve_qr_shift(10)
c += test_solve_qr_no_shift(10)
n = 1000
launch_compares(np.log10(n),[1000,2500,5000])



    