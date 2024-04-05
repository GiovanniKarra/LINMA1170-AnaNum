import numpy as np
import numba as nb
import scipy.linalg as sp


#@nb.jit()
def conjugate_transpose(self,A) :
    if len(A.shape) == 1 :
        B = np.ndarray((A.shape[0],),dtype=complex)
        for i in range(A.shape[0]) :
            B[i] = A[i].conjugate()
        return B
    B = np.ndarray((A.shape[1],A.shape[0]),dtype=complex)
    for i in range(A.shape[0]) :
        for j in range(A.shape[1]) :
            B[j,i] = A[i,j].conjugate()
    return B

def givens(self,xi,xj):
    c = xi / (abs(xi)**2 + abs(xj)**2)**0.5
    s = -xj / (abs(xi)**2 + abs(xj)**2)**0.5
    return c, s

    

def norm(self,v) :
    sum = 0
    for i in range(len(v)) :
        sum += v[i].real**2 + v[i].imag**2
    return sum**0.5

def hessenberg(self,A,P) :
    n = len(A)
    for i in range(len(P)):
        P[i,:] = [1.0 + 0.0j if k == i else 0 for k in range(len(P))]
    U = np.ndarray((len(A),len(A)),dtype=complex)
    for k in range(n-2) :
        x = A[k+1:,k]
        v = x.copy()
         
        v[0] = x[0] + np.sign(x[0]) * norm(x)
        if norm(v) != 0 :
            v = v / norm(v)
        v = np.reshape(v,(len(v),1))
        v_star = v.conjugate().T
        
        U[k+1:,k] = v[:,0].copy()
        A[k+1:,k:] -= 2 * v @ (v_star @ A[k+1:, k:])
        A[:,k+1:] -= 2 * (A[:,k+1:] @ v )@ v_star
    for k in range(n-3,-1,-1):
        v = U[k+1:,k].copy()
        v = np.reshape(v,(len(v),1))
        P[k+1:,k+1:] = P[k+1:,k+1:] - 2 * v @ (v.conjugate().T @ P[k+1:,k+1:])
    return A,P


def new_hessenberg(self,A,Q) :
    n = np.shape(A)[0]
    v = np.empty(n-2, dtype=np.ndarray)
    for i in range(n-2):
        x = np.copy(A[i+1:, i])[np.newaxis].T
        x[0, 0] += np.sqrt(x.conjugate().T@x)[0, 0]*\
            (x[0, 0]/norm if (norm := np.abs(x[0, 0])) > 1e-12 else 1)
        x /= np.sqrt(x.conjugate().T@x)

        v[i] = x

        A[i+1:, i:] -= 2*x @ (x.conjugate().T@A[i+1:, i:])
        A[:, i+1:] -= 2*(A[:, i+1:]@x)@x.conjugate().T
    
    Q[...] = np.identity(n, dtype="complex")
    for i in range(n-3, -1, -1):
        Q[i+1:, i+1:] -= 2*v[i] @ (v[i].conjugate().T@Q[i+1:, i+1:])

n = 4
A = np.ndarray((n,n),dtype=complex)
P = np.ndarray((n,n),dtype=complex)

for i in range(n) :
    for k in range(n) :
        A[i,k] = complex(i+k,i-k)
A2 = A.copy()
A3 = A.copy()   
hessenberg(A,P)
print(P @ conjugate_transpose(P))
print("\n\n___________  PHP* - A  ____________\n\n")
print(P @ A @ conjugate_transpose(P) - A2)
Q = np.ndarray((n,n),dtype=complex)
new_hessenberg(A2,Q)
print("\n\n___________  QHQ* - A  ____________\n\n")
print(Q @ A2 @ conjugate_transpose(Q) - A3)
        


#@nb.jit()
def step_qr(self,H,U,m) :
    for k in range(m-1):
        c, s = givens(H[k,k], H[k+1,k])
        H[k:k+2, k:] = np.array([[c, -s], [s, c]]) @ H[k:k+2, k:]
        U[:,k:k+2] = U[:,k:k+2] @ np.array([[c, s], [-s, c]])
    
    for k in range(m-1):
        H[:k+1,k:k+2] = H[:k+1,k:k+2] @ np.array([[c, s], [-s, c]])
    

#@nb.jit()
def step_qr_shift(self,H,Q, m) :
    step_qr(H,Q,m)
    d = (H[m-2,m-2] - H[m-1,m-1]) / 2
    m_new = H[m-1,m-1] - np.sign(d) * (H[m-1,m-2]**2) / (np.abs(d) + (d**2 + H[m-1,m-2]**2)**0.5)


    

#@nb.jit()
def solve_qr(self,A, use_shifts,eps,max_iter) :
    n = len(A)
    H = A.copy()
    U = np.identity(n,dtype=complex)
    hessenberg(H,U)
    m = n
    k = 0
    for i in range(max_iter) :
        k += 1
        if use_shifts :
            H,U = step_qr_shift(H,U,m)
        else :
            H,U = step_qr(H,U,m)
        m -= 1
        if np.linalg.norm(np.diag(H,-1)) < eps :
            return H,Q
    if k == max_iter :
        return U,-1
    return U,k


n = 4
A = np.ndarray((n,n),dtype=complex)
P = np.ndarray((n,n),dtype=complex)

for i in range(n) :
    for k in range(n) :
        A[i,k] = complex(i+k,i-k)
A2 = A.copy()
A3 = A.copy()
H,Q = solve_qr(A,False,1e-12,10000)
print("\n\n___________  QHQ* - A  ____________\n\n")
