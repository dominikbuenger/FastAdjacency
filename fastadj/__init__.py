
from .core import AdjacencyMatrix

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator

from .krylovschur import krylov_schur_eigs

class AccuracySetup:
    presets = {
        'rough': (16, 1, 2, 0.0, 1e-2),
        'default': (32, 1, 4, 0.0, 1e-3),
        'fine': (64, 8, 7, 0.0, 1e-8)
    }
    
    def __init__(self, N=None, p=None, m=None, eps=None, eigs_tol=None, preset=None):
        
        if preset is not None:
            self.N, self.p, self.m, self.eps, self.eigs_tol = self.presets[preset]
        else:
            assert all(x is not None for x in [N, p, m, eps]), "AccuracySetup parameters N, p, m, eps must be given"
            self.eigs_tol = 0
        
        if N is not None: self.N = N
        if p is not None: self.p = p
        if m is not None: self.m = m
        if eps is not None: self.eps = eps
        if eigs_tol is not None: self.eigs_tol = eigs_tol
    

def scale_points(points, eps=0):
    points -= points.mean(axis=0)
    rho = (0.2499 - 0.5*eps) / np.sqrt((points ** 2).sum(axis=1).max())
    return rho, rho*points
    

def fixed_adjacency_matrix(points, sigma, setup='default'):
    if isinstance(setup, str):
        setup = AccuracySetup(preset=setup)
    
    rho, points = scale_points(points)
    
    adj = AdjacencyMatrix(points.shape[1], rho*sigma, setup.N, setup.p, setup.m, setup.eps)
    adj.points = points
    
    return adj


def normalized_adjacency_eigs(adj, k=6, tol=0, method='krylov-schur', shift=1, one_shift=2):
    n = adj.n
    u1 = np.sqrt(adj.apply(np.ones(n)))
    d_invsqrt = 1 / u1
    u1 /= np.linalg.norm(u1)
    
    if k == 1:
        return np.array([1.0]), u1[:, np.newaxis]

    def matvec(v):
        w = d_invsqrt * adj.apply(d_invsqrt * v)
        if shift != 0:
            w += shift*v
        if one_shift != 0:
            w -= one_shift * (u1 @ v) * u1
        return w
    
    if one_shift != 0:
        k -= 1
    
    if method == 'arpack-fortran':
        if shift != 0 or one_shift != 0:
            raise ValueError("Eigenvalue computation method 'arpack-fortran' is incompatible with shifts")
        w, U = adj.normalized_eigs(k, tol=tol)
    elif method == 'arpack-scipy':
        operator = LinearOperator((n,n), dtype=np.float64, matvec=matvec)
        w, U = eigsh(operator, k=k, which='LA' if shift == 0 else 'LM', tol=tol)
    elif method == 'krylov-schur':
        w, U = krylov_schur_eigs(matvec, n, k=k, tol=tol)
    else:
        raise ValueError("Unknown eigenvalue computation method: {}".format(method))
        
    ind = np.argsort(-w)
    w = w[ind]
    U = U[:, ind]
    
    if shift != 0:
        w -= shift
    if one_shift != 0:
        w = np.hstack((1.0, w))
        U = np.hstack((u1[:,np.newaxis], U))
    
    return w, U


def normalized_laplacian_norm(adj, tol=0):
    n = adj.n
    d_invsqrt = 1 / np.sqrt(adj.apply(np.ones(n)))
    
    def matvec(v):
        return v - d_invsqrt * adj.apply(d_invsqrt * v)
    
    nrm = eigsh(LinearOperator((n,n), dtype=np.float64, matvec=matvec),
                k = 1,
                which = 'LM',
                tol = tol,
                return_eigenvectors = False)[0]
    
    return nrm
    
    
