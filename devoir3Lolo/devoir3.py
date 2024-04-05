import numpy as np
import numba as nb
import scipy.linalg as sp
import matplotlib.pyplot as plt

#@nb.jit()
def conj(A) :
    return A.conjugate().T

def givens(xi,xj):
    if xi != 0 or xj != 0:
        c = abs(xi) / (abs(xi)**2 + abs(xj)**2)**0.5
        s = abs(xj) / (abs(xi)**2 + abs(xj)**2)**0.5
    else : 
        c = 1
        s = 0
    return c, s

def mult(A,B) :
    if A.shape[1] != B.shape[0] :
        raise ValueError("Matrix dimensions do not match")
    C = np.ndarray((A.shape[0],B.shape[1]),dtype=complex)
    for i in range(A.shape[0]) :
        for j in range(B.shape[1]) :
            sum = 0
            for k in range(A.shape[1]) :
                sum += A[i,k] * B[k,j]
            C[i,j] = sum
    return C
    

def norm(v) :
    return np.sqrt((v.conjugate().T @ v))


def sign(x) :
    return x/np.abs(x)

def hessenberg(A,P) :
    n = A.shape[0]
    for i in range(len(P)):
        P[i,:] = [1.0 + 0.0j if k == i else 0 for k in range(len(P))]
    U = np.ndarray((len(A),len(A)),dtype=np.complex128)
    for k in range(n-2) :
        x = A[k+1:,k].copy()
        x[0] = x[0] + sign(x[0]) * sp.norm(x)
        if sp.norm(x) > 1e-13 :
            x /= sp.norm(x)
        x = np.reshape(x,(len(x),1))
        x_star = x.conjugate().T
        U[k+1:,k] = x[:,0].copy()
        A[k+1:,k:] -= 2*x @ x_star @ A[k+1:, k:]
        A[:,k+1:] -= 2*(A[:,k+1:] @ x )@ x_star
    for k in range(n-3,-1,-1):
        v = U[k+1:,k].copy()
        v = np.reshape(v,(len(v),1))
        P[k+1:,k+1:] = P[k+1:,k+1:] - 2 * v @ (v.conjugate().T @ P[k+1:,k+1:])
    A[:] = np.where(np.abs(A) < 1e-13, 0, A)
    P[:] = np.where(np.abs(P) < 1e-13, 0, P)
    return 
    

#@nb.jit()
def step_qr(H,U,m) :
    """Ensuite, une transformation unitaire Q est appliquée à gauche et à droite. On vous demande ici d’écrire
une fonction Python step_qr(H, U, m) où
• H est un numpy.ndarray de type complex, de taille n x n qui contient une matrice sous forme
Hessenberg
• U est un numpy.ndarray de type complex, de taille n x n qui contient la matrice de transformation
unitaire — H = U ∗ AU
• m est la dimension de la matrice active
• En sortie, le tableau H a été réécrit par RQ où le produit QR est une décomposition qr de H. U
est également mis à jour."""
    c = np.zeros(m+1,dtype=complex)
    s = np.zeros(m+1,dtype=complex)
    for k in range(m-1):
        c[k], s[k] = givens(H[k,k], H[k+1,k])
        H[k:k+2, k:] = np.array([[c[k], -s[k]], [s[k], c[k]]]) @ H[k:k+2, k:]
        U[:,k:k+2] = U[:,k:k+2] @ np.array([[c[k], s[k]], [-s[k], c[k]]])
    
    for k in range(m-1):
        H[:k+2,k:k+2] = H[:k+2,k:k+2] @ np.array([[c[k], s[k]], [-s[k], c[k]]])
    

#@nb.jit()
def step_qr_shift(H,Q, m) :
    """Dans la majorité des cas, il est en possible d’itérer uniquement avec la fonction step_qr(H, Q) jusqu’à
convergence à la forme de Schur. Néanmoins, il existe une manière simple d’accélérer fortement la
convergence de l’algorithme en introduisant un shift σ; on utilisera ici le Wilkinson shift. On calcule
alors la décomposition QR de H − σI, au lieu de H.
On vous demande d’écrire une fonction Python m_new = step_qr_shift(H, Q, m) où
• H, Q et m sont définis comme ci-dessus
• m_new est la dimension active après avoir effectué l’itération
• En sortie, les tableaux H et Q sont mis à jours comme ci-dessus"""
    step_qr(H,Q,m)
    d = (H[m-2,m-2] - H[m-1,m-1]) / 2
    m_new = H[m-1,m-1] - np.sign(d) * (H[m-1,m-2]**2) / (np.abs(d) + (d**2 + H[m-1,m-2]**2)**0.5)
    return m_new


    

#@nb.jit()
def solve_qr(A, use_shifts,eps,max_iter) :
    """Pour synthétiser, on vous demande également une fonction Python U, k = solve_qr(A, use_shifts,
eps, max_iter) où
• A est un numpy.ndarray de type complex, de taille n x n qui contient la matrice A en entrée, et
la matrice T en sortie
• U est un numpy.ndarray de type complex, de taille n x n qui contient la transformation unitaire
U telle que A = U T U ∗
• use_shifts est un booléen qui indique si on souhaite utiliser des shifts
• eps est un float qui détermine le critère d’arrêt : on considère que l’algorithme a “convergé”
lorsque les entrées sous-diagonales de A sont inférieures en norme à eps
• k est un int qui indique le nombre d’itérations qr qui ont été nécessaires, ou bien -1 si on a
atteint max_iter itérations"""
    n = len(A)
    H = A.copy()
    U = np.identity(n,dtype=complex)
    hessenberg(H,U)
    m = n
    k = 0
    for i in range(max_iter) :
        k += 1
        if use_shifts :
            m = step_qr_shift(H,U,m)
        else :
            step_qr(H,U,m)
            
        diff = conj(U) @ A @ U - H
        diff = np.where(np.abs(diff) < 1e-13, 0, diff)
        if np.linalg.norm(np.diag(H,-1)) < eps and np.linalg.norm(diff) < eps:
            return U,k
    A = H
    if k == max_iter :

        return U,-1
    return U,k


n = 40
A = np.ndarray((n,n),dtype=complex)
P = np.ndarray((n,n),dtype=complex)

A = np.random.rand(n,n) #+ 1j * np.random.rand(n,n)
A2 = A.copy()
A3 = A.copy()
U,k = solve_qr(A,False,1e-15,10000)
print(k)
print(U @ A @ conj(U) - A2)
plt.spy(U @ A @ conj(U) - A2)
plt.show()
