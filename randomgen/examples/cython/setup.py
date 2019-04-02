# python setup.py build_ext -i
from distutils.core import setup
from os.path import join

import numpy as np
from Cython.Build import cythonize
from setuptools.extension import Extension

extending = Extension("extending",
                      sources=['extending.pyx'],
                      include_dirs=[np.get_include()])
distributions = Extension("extending_distributions",
                          sources=['extending_distributions.pyx',
                                   join('..', '..', '..', 'randomgen', 'src',
                                        'distributions', 'distributions.c')],
                          include_dirs=[np.get_include()])

extensions = [extending, distributions]

setup(
    ext_modules=cythonize(extensions)
)
