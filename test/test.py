
import fastadj
import numpy as np
from time import perf_counter as timer


n = 10000
d = 3
sigma = 1.0
numev = 11

x = np.random.randn(n, d)

adj = fastadj.AdjacencyMatrix(x, sigma, 'default')

print("Setup done")

degrees = adj.apply(np.ones(n))

print("Avg/min/max degree:", degrees.mean(), degrees.min(), degrees.max())


tic = timer()

nrm = adj.normalized_laplacian_norm()

time_nrm = timer() - tic
print("Normalized Laplacian norm: {}   (computed in {} seconds)".format(nrm, time_nrm))


tic = timer()

w, U = adj.normalized_eigs(numev)

time_eigs = timer() - tic
print("Time for eigenvalue computation: {} seconds".format(time_eigs))

d_invsqrt = 1 / np.sqrt(adj.apply(np.ones(n)))

for i in range(w.size):
	res = np.linalg.norm(d_invsqrt * adj.apply(d_invsqrt * U[:,i]) - U[:,i] * w[i])
	print("Eigenvalue #{}: {:.4f} - Residual: {:.4e}".format(i, w[i], res))

