from distutils.core import setup, Extension
from configparser import ConfigParser
import os.path as osp


# read config file
config = ConfigParser()
config.read('config.ini')

# read nfft base dir and set include directories
nfft_base = config['NFFT']['base_dir']
include_dirs = [osp.join(nfft_base, 'include'), osp.join(nfft_base, 'applications', 'fastsum')]

# read numpy include dir
numpy_include_dir = config['NUMPY']['include_dir']
if len(numpy_include_dir) > 0:
    include_dirs.append(numpy_include_dir)

macros = [('MAJOR_VERSION', '0'), ('MINOR_VERSION', '2')]

# check if arpack is available
arpack_available = int(config['ARPACK']['available'])
arpack_include_dir = config['ARPACK']['include_dir']
if arpack_available:
    macros.append(('BUILD_EIGS', '1'))
    if arpack_include_dir:
        include_dirs.append(arpack_include_dir)
    

# C extension fastadj.core
core_ext = Extension('fastadj.core',
    define_macros = macros,
    include_dirs = include_dirs,
    libraries = ['fastsumjulia'],
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
