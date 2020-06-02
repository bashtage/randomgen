# python setup.py build_ext -i
from distutils.core import setup
from os.path import join

from Cython.Build import cythonize
import numpy as np
from setuptools.extension import Extension

extending = Extension(
    "extending", sources=["extending.pyx"], include_dirs=[np.get_include()]
)
distributions = Extension(
    "extending_distributions",
    sources=[
        "extending_distributions.pyx",
        join("..", "..", "..", "randomgen", "src", "distributions", "distributions.c"),
    ],
    include_dirs=[np.get_include()],
)
low_level = Extension(
    "low_level", sources=["low_level.pyx"], include_dirs=[np.get_include()]
)

extensions = [extending, distributions, low_level]

setup(ext_modules=cythonize(extensions))
