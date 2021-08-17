
import numpy as np
from scipy.linalg import eigh



def krylov_schur_eigs(operator, n, k=6, tol=0, W=None):
    if tol is None or tol <= 0:
        tol = 1e-14
    max_iter = 300
    eps = np.finfo(float).eps ** (2/3)
    p = min(max(2*k,20), n)
    k0 = k
    
    v = np.random.randn(n)
    if W is not None:
        v -= W @ (W.T @ v)
    v /= np.linalg.norm(v)
    
    V = np.zeros((n,p))
    H = np.zeros((p,p))
    d = None
    c = None
    r_norm = 0
    just_restarted = False
    
    
    for it in range(max_iter):
        
        
        for jj in range(0 if it == 0 else k, p):
            V[:,jj] = v
            r = operator(v)
            alpha = np.dot(v, r)
            
            if jj == 0:
                r -= alpha*v
            elif just_restarted:
                r -= V[:,:jj+1] @ (V[:,:jj+1].T @ r)
                just_restarted = False
            else:
                r -= (alpha*v + r_norm * V[:,jj-1])

            v, r_norm = robust_reorth(r, V[:,:jj+1], W)
            if v is None:
                raise RuntimeError('Krylov-Schur eigenvalue computation: Unable to orthogonalize residual')
            
            H[jj,jj] = alpha
            if jj < p-1:
                H[jj,jj+1] = r_norm
                H[jj+1,jj] = r_norm
        
        d, U = eigh(H)
        
        ind = np.argsort(-d)
        
        converged_mask = abs(r_norm * U[-1, ind[:k0]]) < tol*np.maximum(eps, abs(d[ind[:k0]]))
        num_converged = converged_mask.sum()
        
        if num_converged >= k0 or it == max_iter-1:
            break
        
        # Adjust k to prevent stagnating
        k = k0 + min(num_converged, (p - k0) // 2)
        if k == 1 and p > 3:
            k = p // 2
            
        ind = ind[:k]
        d = d[ind]
        U = U[:,ind]
        V[:,:k] = V @ U
        c = r_norm * U[-1, :]
        
        H = np.zeros((p,p))
        for i in range(k):
            H[i,i] = d[i]
            H[i,k] = c[i]
            H[k,i] = c[i]
            
        just_restarted = True
            
        
    ind = ind[:k0]
    return d[ind], V @ U[:,ind]




def robust_reorth(x, V, W=None, num_reorth=5, num_restarts=3, tol=1e-10):
    norm = np.linalg.norm(x)
    
    for _ in range(num_reorth):
        x -= V @ (V.T @ x)
        if W is not None:
            x -= W @ (W.T @ x)
        
        norm_new = np.linalg.norm(x)
        if norm_new > norm / np.sqrt(2):
            return x/norm_new, norm_new
        
        norm = norm_new
    
    
    # cannot reorthogonalize, invariant subspace found
    
    for __ in range(num_restarts):
        x = np.random.randn(x.size)
        VTx = V.T @ x
        # x /= np.linalg.norm(x)
        
        for _ in range(num_reorth):
            x -= V @ VTx
            if W is not None:
                x -= W @ (W.T @ x)
            x /= np.linalg.norm(x)
            VTx = V.T @ x
            
            if abs(1 - np.linalg.norm(x)) < tol and all(abs(VTx) < tol):
                return x, 0.0
    
    return None, 0.0
        
    
    
        