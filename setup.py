from distutils.core import setup, Extension

NFFTDIR = '/home/dalfke/phd/nfft/nfft'

core_ext = Extension('fastadj.core',
    define_macros = [('MAJOR_VERSION', '0'), ('MINOR_VERSION', '1')],
    include_dirs = [NFFTDIR + '/include', NFFTDIR + '/applications/fastsum'],
    libraries = ['fastsumjulia'],
    # library_dirs = [NFFTDIR + '/julia/fastsum'],
    sources = ['fastadj/core.c'])

setup(name = 'fastadj',
    version = '1.0',
    description = 'Fast multiplication with Gaussian adjacency matrices using NFFT/Fastsum',
    author = 'Dominik Alfke',
    author_email = 'dominik.alfke@mathematik.tu-chemnitz.de',
    url = 'https://github.com/dominikalfke/FastAdjacency',
    packages = ['fastadj'],
    py_modules = [],
    ext_modules = [core_ext])
