from distutils.core import setup, Extension
import os
import numpy as np
import glob

def filter_paths(paths, *filenames):
    return [x for x in paths if len(x) > 0 and any(len(glob.glob(os.path.join(x, f))) > 0 for f in filenames)]

# set include and library directories
include_dirs = [os.path.join(np.__path__[0], 'core', 'include')]

# load library dirs from environment
library_dirs = filter_paths(os.environ.get('LD_LIBRARY_PATH').split(':'), 'libnfftjulia.so*', 'libfftw3.so*')

macros = [('MAJOR_VERSION', '0'), ('MINOR_VERSION', '2')]

# C extension fastadj.core
core_ext = Extension('fastadj.core',
    define_macros = macros,
    include_dirs = include_dirs,
    libraries = ['nfftjulia', 'fftw3', 'm'],
    library_dirs = library_dirs,
    runtime_library_dirs = library_dirs,
    sources = ['fastadj/core.c'])

# run setup
setup(name = 'fastadj',
    version = '0.2',
    description = 'Fast multiplication with Gaussian adjacency matrices using NFFT/Fastsum',
    author = 'Dominik Alfke',
    author_email = 'dominik.alfke@mathematik.tu-chemnitz.de',
    url = 'https://github.com/dominikalfke/FastAdjacency',
    packages = ['fastadj'],
    py_modules = [],
    ext_modules = [core_ext])
