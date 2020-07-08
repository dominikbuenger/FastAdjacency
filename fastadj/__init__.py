
from .core import AdjacencyCore

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


class AdjacencyMatrix():
    
    def __init__(self, points, sigma, setup='default', diagonal=0.0):
        if isinstance(setup, str):
            setup = AccuracySetup(preset=setup)
        
        self.setup = setup
        self.scaling_factor = 1
        self.core = None
        
        self._sigma = sigma
        self.points = points
        
        if diagonal != 0:
            self.diagonal = diagonal
    
    def _setup_core(self, d):
        self.core = AdjacencyCore(d, self.scaling_factor*self._sigma, 
                                  self.setup.N, self.setup.p, self.setup.m, self.setup.eps)
    
    @property
    def d(self):
        return self.core.d
    
    @property
    def n(self):
        return self.core.n
    
    @property
    def scaled_sigma(self):
        return self.core.sigma
    
    @property
    def sigma(self):
        return self._sigma
    
    @sigma.setter
    def sigma(self, sigma):
        points = self.core.points
        self._sigma = sigma
        self._setup_core(points.shape[1])
        self.core.points = points
    
    @property
    def scaled_points(self):
        return self.core.points
    
    @property
    def points(self):
        return self.points_center + self.core.points / self.scaling_factor
    
    @points.setter
    def points(self, points):
        
        d = points.shape[1]
        
        self.points_center = points.mean(axis=0)
        points -= self.points_center
        
        radius = np.sqrt((points ** 2).sum(axis=1).max())
        allowed_radius = 0.2499 - 0.5*self.setup.eps
        
        if self.core is None or \
                d != self.core.d or \
                radius*self.scaling_factor > allowed_radius or \
                radius*self.scaling_factor < 0.125:
            self.scaling_factor = allowed_radius / radius
            self._setup_core(d)
            
        self.core.points = points * self.scaling_factor

    @property
    def diagonal(self):
        return self.core.diagonal
    
    @diagonal.setter
    def diagonal(self, diag):
        self.core.diagonal = diag

    def apply(self, v):
        return self.core.apply(v)
    
    def normalized_eigs(self, k=6, method='krylov-schur', shift=1, one_shift=2, tol=None):
        if tol is None:
            tol = self.setup.eigs_tol
            
        n = self.core.n
        u1 = np.sqrt(self.core.apply(np.ones(n)))
        d_invsqrt = 1 / u1
        u1 /= np.linalg.norm(u1)
        
        if k == 1:
            return np.array([1.0]), u1[:, np.newaxis]
    
        def matvec(v):
            w = d_invsqrt * self.core.apply(d_invsqrt * v)
            if shift != 0:
                w += shift*v
            if one_shift != 0:
                w -= one_shift * (u1 @ v) * u1
            return w
        
        if one_shift != 0:
            k -= 1
        
        if method == 'arpack-fortran':
            if not hasattr(self.core, 'normalized_eigs'):
                raise ValueError("Eigenvalue computation method 'arpack-fortran' has not been built")
            if shift != 0 or one_shift != 0:
                raise ValueError("Eigenvalue computation method 'arpack-fortran' is incompatible with shifts")
            w, U = self.core.normalized_eigs(k, tol=tol)
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
        


    def normalized_laplacian_norm(self, tol=None):
        if tol is None:
            tol = self.setup.eigs_tol
            
        n = self.core.n
        d_invsqrt = 1 / np.sqrt(self.core.apply(np.ones(n)))
        
        def matvec(v):
            return v - d_invsqrt * self.core.apply(d_invsqrt * v)
        
        nrm = eigsh(LinearOperator((n,n), dtype=np.float64, matvec=matvec),
                    k = 1,
                    which = 'LM',
                    tol = tol,
                    return_eigenvectors = False)[0]
        
        return nrm
    
    
