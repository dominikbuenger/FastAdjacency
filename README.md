# FastAdjacency
### Python extension to compute fast approximate multiplication with Gaussian adjacency matrices

This package provides a Python interface to a part of the [NFFT3](https://github.com/NFFT/nfft) library to quickly approximate adjacency matrices of fully connected graphs with Gaussian edge weights.
See [test/showcase.ipynb](test/showcase.ipynb) for an overview over the method.

# Installation

* This software has been tested with Python 3.7.

* This software currently depends on the *Julia interface* of the NFFT3 library.
  - Download the NFFT3 source from [https://github.com/NFFT/nfft](https://github.com/NFFT/nfft).
  - Configure, build and install NFFT3 following the instructions on the homepage. Make sure to configure the library with `--enable-julia`.

* Navigate to the FastAdjacency folder.

* Export the path to your NFFT3 installation folder as the `NFFT_BASE` environment variable via `export NFFT_BASE=/path/to/your/NFFT/base/directory`. 

* From the same terminal, run `make` and `make install`. Optionally run `make check` to test your installation.

* To rebuild, run `make clean` before `make`.


# Usage

This package mainly consists of the following three classes:

* `AccuracySetup` stores setup parameters for the NFFT Fastsum algorithm. You can load one of three presets via `AccuracySetup(preset='name')`. Available preset names are `'default'`, `'fine'`, and `'rough'`.

* `AdjacencyCore` is the core type of fastadj, implemented in C. It serves as a wrapper for the NFFT `fastsum` type. Although it can be instantiated directly, we advise against it.

* `AdjacencyMatrix` is a convenient wrapper for `AdjacencyCore`, written in Python. This is the recommended way to use this package.

See [`test/showcase.ipynb`](test/showcase.ipynb) and [`test/test.py`](test/test.py) for an example.
