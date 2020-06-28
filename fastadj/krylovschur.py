
import numpy as np

ARPACKDIR = '/usr/local/include/arpack'
from scipy.linalg import eigh


# based strongly on matlab.sparsefun.eigs
def normalized_adjacency_eigs_ks(adj, k=6, tol=0):
    n = adj.n
    d_invsqrt = 1 / np.sqrt(adj.apply(np.ones(n)))
    
    def normalized_adjacency_matvec(v):
        return d_invsqrt * adj.apply(d_invsqrt * v)
    
    

def krylov_schur_eigs(operator, n, k=6, tol=0):
    if tol <= 0:
        tol = 1e-14
    maxit = 300
    p = min(max(2*k,20),n)
    k0 = k
    
    v = np.random.randn(n)
    v /= np.linalg.norm(v)
    
    V = np.zeros((n,p))
    d = None
    c = None
    norm_res = 0
    just_restarted = False
    size_V = 0      # == 0 in first iteration, == k afterwards
    
        
    mm = 0
    while True:
        mm += 1
        
        H = np.zeros((p,p))
        for i in range(size_V):
            H[i,i] = d[i]
            H[i,k] = c[i]
            H[k,i] = c[i]
            
            
        for jj in range(size_V, p):
            V[:,jj] = v
            r = operator(v)
            alpha = np.dot(v, r)
            
            if jj == 0:
                r -= alpha*v
            elif just_restarted:
                r -= V[:,:jj+1] @ (V[:,:jj+1].T @ r)
                just_restarted = False
            else:
                r -= (alpha*v + norm_res * V[:,jj-1])

            v, norm_res = __robust_reorthogonalize__(V, r, jj)
            if v is None:
                raise RuntimeError('Krylov-Schur eigenvalue computation: Unable to orthogonalize residual')
            
            H[jj,jj] = alpha
            if jj < p-1:
                H[jj,jj+1] = norm_res
                H[jj+1,jj] = norm_res
        
        d, U = eigh(H)
        
        ind = np.argsort(-d)
        
        converged_mask = abs(norm_res * U[-1, ind[:k0]]) < tol*np.maximum(np.finfo(float).eps ** (2/3), abs(d[ind[:k0]]))
        nconv = converged_mask.sum()
        
        if nconv >= k0 or mm == maxit:
            ind = ind[:k0]
            return d[ind], V @ U[:,ind]
        
        # Adjust k to prevent stagnating
        k = k0 + min(nconv, (p - k0) // 2)
        if k == 1 and p > 3:
            k = p // 2
            
        ind = ind[:k]
        d = d[ind]
        U = U[:,ind]
        V[:,:k] = V @ U
        
        c = norm_res * U[-1, :]
        just_restarted = True
        size_V = k        
    

def __robust_reorthogonalize__(V, r, index):
    norm_r0 = np.linalg.norm(r)
    w = np.zeros(index+1)
    
    for num_reorths in range(5):
        dw = V[:,:index+1].T @ r
        w += dw
        r -= V[:,:index+1] @ dw
        
        norm_res = np.linalg.norm(r)
        if norm_res > norm_r0 / np.sqrt(2):
            break
        norm_r0 = norm_res
    else:
        # cannot reorthogonalize, invariant subspace found
        
        for restart in range(3):
            r = np.random.randn(r.size)
            dw = V[:,:index+1].T @ r
            r /= np.linalg.norm(r)
            
            for num_reorths in range(5):
                r -= V[:,:index+1] @ dw
                r /= np.linalg.norm(r)
                
                dw = V[:,:index+1].T @ r
                
                if abs(1 - np.linalg.norm(r)) <= 1e-10 and all(abs(dw) < 1e-10):
                    return r, 0
        
        return None, 0
        
    return r/norm_res, norm_res
    
        