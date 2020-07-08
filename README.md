# FastAdjacency
### Python extension to compute fast approximate multiplication with Gaussian adjacency matrices

This package provides a Python interface to a part of the [NFFT3](https://github.com/NFFT/nfft) library to quickly approximate adjacency matrices of fully connected graphs with Gaussian edge weights.
For a given $d$-dimensional *point cloud* of $n$ points $x_i \in \mathbb{R}^d$, $i=1,\ldots,n$, the associated adjacency matrix has the form
$$
A = (a_{ij})_{i,j=1}^n \in \mathbb{R}^{n \times n}, \qquad A_{ij} = \begin{cases} 0 & \text{if } i=j, \\ \exp\left(\frac{-\|\mathbf{x}_i - \mathbf{x}_j\|^2}{2 \sigma^2}\right) & \text{else,} \end{cases}
$$
where $\sigma$ is a shape parameter. 
The quality of the approximation depends on a handful of parameters. 
If these are kept fixed, the computational cost of a single matrix-vector product depends linearly on $n$ but expontially on $d$, as opposed to the naive computations depending quadratically on $n$ and linearly on $d$.
For that reason, this software is targeted at the case of very large $n$ and small $d=2,3,4$.


# Installation

* This software has been tested with Python 3.7.

* This software currently depends on the *Julia interface* of the NFFT3 library.
  - Download the NFFT3 source from [https://github.com/NFFT/nfft].
  - Configure, build and install NFFT3 following the instructions on the homepage. Make sure to configure the library with `--enable-julia`.

* Navigate to the FastAdjacency folder.

* Edit `config.ini` and set the `base_dir` variable in the `\[NFFT\]` section to the path to your NFFT3 installation folder.

* Run `make` and `make install`.

* The `\[ARPACK\]` config options are only experimental and currently not supported.

* To rebuild with a changed config, run `make clean` before `make`.


# Usage

This package mainly consists of the following three classes:

* `AccuracySetup` stores setup parameters for the NFFT Fastsum algorithm. You can load one of three presets via `AccuracySetup(preset='name')`. Available preset names are `'default'`, `'fine'`, and `'rough'`.

* `AdjacencyCore` is the core type of fastadj, implemented in C. It serves as a wrapper for the NFFT `fastsum` type. Although it can be instantiated directly, we advise against it.

* `AdjacencyMatrix` is a convenient wrapper for `AdjacencyCore`, written in Python. This is the recommended way to use this package.

See [`test/test.py`] for an example.
