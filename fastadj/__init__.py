
from .core import AdjacencyMatrix

import numpy as np
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.linalg import eigh

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
    points = points - points.mean(axis=0)
    rho = (0.2499 - 0.5*eps) / np.sqrt((points ** 2).sum(axis=1).max())
    return rho, rho*points
    

def fixed_adjacency_matrix(points, sigma, setup='default'):
    if isinstance(setup, str):
        setup = AccuracySetup(preset=setup)
    
    rho, points = scale_points(points)
    
    adj = AdjacencyMatrix(points.shape[0], rho*sigma, setup.N, setup.p, setup.m, setup.eps)
    adj.points = points
    
    return adj


def normalized_adjacency_eigs(adj, k=6, tol=0):
    n = adj.n
    d_invsqrt = 1 / np.sqrt(adj.apply(np.ones(n)))
    
    # def signless_laplacian_matvec(v):
    #     return v + d_invsqrt * adj.apply(d_invsqrt * v)
    
    # signless_laplacian = LinearOperator((n,n), dtype=np.float64, matvec=signless_laplacian_matvec)
    
    # w, U = eigsh(signless_laplacian, k=k, tol=tol)
    
    # w = w - 1
    
    def normalized_adj_matvec(v):
        return d_invsqrt * adj.apply(d_invsqrt * v)
    
    normalized_adjacency = LinearOperator((n,n), dtype=np.float64, matvec=normalized_adj_matvec)
    
    w, U = eigsh(normalized_adjacency, k=k, which='LA', tol=tol)
    
    
    return w, U

