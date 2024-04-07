import numpy as np
import numba as nb


@nb.njit()
def conj(A) :
    return np.transpose(A.conjugate())

@nb.njit()
def givens(xi,xj):
    if xi != 0 or xj != 0:
        ang = np.angle(xj) - np.angle(xi)
        c = abs(xi) / np.hypot(abs(xi),abs(xj))
        s = abs(xj) / np.hypot(abs(xi),abs(xj))
        return np.array([[c,-s * np.exp(-1j*ang)],
                         [s * np.exp(1j*ang),c]])
    else : 
        return np.identity(2,dtype=np.complex128)


@nb.njit()
def mult(A,B) :
    if A.shape[1] != B.shape[0] :
        raise ValueError("Dimensions non compatibles")
    C = np.empty((A.shape[0],B.shape[1]),dtype=np.complex128)
    for i in range(A.shape[0]) :
        for j in range(B.shape[1]) :
            C[i,j] = 0.0 + 0.0j
            for k in range(A.shape[1]) :
                C[i,j] += A[i,k] * B[k,j]
            
    return C

@nb.njit()
def norm(v) :
    sum = 0
    for i in range(len(v)) :
        sum += np.abs(v[i])**2
    return np.sqrt(sum)

@nb.njit()
def sign(x) :
    if np.abs(x) < 1e-15:
        return 1
    return x/np.abs(x)

@nb.njit()
def hessenberg(A,P) :
    n = A.shape[0]
    for i in range(n):
        for j in range(n):
            P[i,j] = 0.0 + 0.0j
        P[i,i] = 1.0 + 0.0j    
    U = np.empty((len(A),len(A)),dtype=np.complex128)
    for k in range(n-2) :
        x = np.empty(n-k-1,dtype=np.complex128)
        x[:] = A[k+1:,k]
        x[0] += sign(x[0])  * norm(x)
        nor = norm(x)
        if np.abs(nor) > 1e-15 :
            x /= np.abs(nor)
        x = x[:, np.newaxis]
        x_star = conj(x)
        U[k+1:,k] = x[:,0]
        A[k+1:,k:] -= 2*mult(x, mult(x_star, A[k+1:, k:]))
        A[:,k+1:] -= 2*mult(mult(A[:,k+1:],x ), x_star)
    for k in range(n-3,-1,-1):
        v = U[k+1:,k][:, np.newaxis]
        P[k+1:,k+1:] -= 2 * mult(v ,mult(conj(v), P[k+1:,k+1:]))
    
    

@nb.njit()
def step_qr(H,U,m) :
    list = {}
    for k in range(m-1):
        g = givens(H[k,k], H[k+1,k])
        list[k] = g
        gs = conj(g)
        H[k:k+2, k:] = mult(gs,H[k:k+2, k:])
        
    
    for k in range(m-1):
        g = list[k]
        H[:k+2,k:k+2] = mult(H[:k+2,k:k+2],g)
        U[:,k:k+2] = mult(U[:,k:k+2],g)
    
@nb.njit()
def shift(A):
    h1 = A[0,0]
    h2 = A[0,1]
    h3 = A[1,0]
    h4 = A[1,1]
    l1 = ((h1 + h4) + np.sqrt((h1+h4)**2 - 4*(h1*h4 - h2*h3))) / 2
    l2 = ((h1 + h4) - np.sqrt((h1+h4)**2 - 4*(h1*h4 - h2*h3))) / 2
    if abs(l1 - h4) < np.abs(l2 - h4):
        return l1
    else:
        return l2

@nb.njit()
def step_qr_shift(H,Q, m,eps=1e-15) :
    mu = shift(H[m-2:m,m-2:m])
    for i in range(m):
        H[i,i] -= mu
    step_qr(H,Q,m)
    for i in range(m):
        H[i,i] += mu
    if np.abs(H[m-1,m-2]) < eps:
        return m-1
    
    return m

@nb.njit()
def sousdiag(H):
    n = len(H)
    sum = 0
    for i in range(n-1):
        sum += np.abs(H[i+1,i])**2
    return np.sqrt(sum)  

@nb.njit()
def solve_qr(A, use_shifts,eps=1e-13,max_iter=5000) :
    n = len(A)
    U = np.empty((n,n),dtype=np.complex128)
    hessenberg(A,U)
    m = n
    k = 0
    if use_shifts :
        for i in range(max_iter) :
            k += 1
            m = step_qr_shift(A,U,m)
            if sousdiag(A) < eps or m<= 1 :
                return U,k
    else :    
        for i in range(max_iter) :
            k += 1
            step_qr(A,U,m)
            if sousdiag(A) < eps or m <= 1 :
                return U,k
    if k == max_iter :
        return U,-1
    return U,k

