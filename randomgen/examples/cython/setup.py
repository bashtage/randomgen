# python setup.py build_ext -i
from setuptools import setup
from setuptools.extension import Extension

from os.path import join

from Cython.Build import cythonize
import numpy as np

extending = Extension(
    "extending", sources=["extending.pyx"], include_dirs=[np.get_include()]
)
distributions = Extension(
    "extending_distributions",
    sources=[
        "extending_distributions.pyx",
        join(
            "..",
            "..",
            "..",
            "randomgen",
            "src",
            "distributions",
            "distributions.orig.c",
        ),
    ],
    include_dirs=[np.get_include()],
)
low_level = Extension(
    "low_level", sources=["low_level.pyx"], include_dirs=[np.get_include()]
)

extensions = [extending, distributions, low_level]

setup(ext_modules=cythonize(extensions))
