from distutils.core import setup, Extension
import os
import numpy as np


# read nfft base dir 
nfft_base = os.environ.get('NFFT_BASE')
assert nfft_base is not None, 'Environment variable NFFT_BASE must be exported before setup'

# set include and library directories
include_dirs = [os.path.join(nfft_base, 'include'), 
                os.path.join(nfft_base, 'applications', 'fastsum'),
                os.path.join(np.__path__[0], 'core', 'include')]

library_dirs = [os.path.join(nfft_base, 'julia', 'fastsum')]

macros = [('MAJOR_VERSION', '0'), ('MINOR_VERSION', '2')]

# C extension fastadj.core
core_ext = Extension('fastadj.core',
    define_macros = macros,
    include_dirs = include_dirs,
    libraries = ['fastsumjulia'],
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
