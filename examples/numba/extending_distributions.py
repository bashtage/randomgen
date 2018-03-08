"""

On *nix, execute in core_prng/src/distributions

export PYTHON_INCLUDE=#path to Python's include folder, usually ${PYTHON_HOME}/include/python${PYTHON_VERSION}m
export NUMPY_INCLUDE=#path to numpy's include folder, usually ${PYTHON_HOME}/lib/python${PYTHON_VERSION}/site-packages/numpy/core/include
gcc -shared -o libdistributions.so -fPIC distributions.c -I${NUMPY_INCLUDE} -I${PYTHON_INCLUDE}
mv libdistributions.so ../../../examples/numba/

On Windows

rem PYTHON_HOME is setup dependent, this is an example
set PYTHON_HOME=c:\Anaconda
cl.exe /LD .\distributions.c -DDLL_EXPORT -I%PYTHON_HOME%\lib\site-packages\numpy\core\include -I%PYTHON_HOME%\include %PYTHON_HOME%\libs\python36.lib
move distributions.dll ../../../examples/numba/
"""
import numpy as np
from cffi import FFI
from core_prng import Xoroshiro128
import numba as nb

ffi = FFI()
lib = ffi.dlopen('./libdistributions.so')
ffi.cdef("""
double random_gauss(void *prng_state);
double random_gauss_zig(void *prng_state);
""")
x = Xoroshiro128()
xffi = x.cffi
prng = xffi.prng

random_gauss = lib.random_gauss
random_gauss_zig = lib.random_gauss_zig


def normals(n, prng):
    out = np.empty(n)
    for i in range(n):
        out[i] = random_gauss(prng)
    return out


def normals_zig(n, prng):
    out = np.empty(n)
    for i in range(n):
        out[i] = random_gauss_zig(prng)
    return out


normalsj = nb.jit(normals, nopython=True)
normals_zigj = nb.jit(normals_zig, nopython=True)

prng_address = int(ffi.cast('uintptr_t', prng))

norm = normalsj(1000, prng_address)
norm_zig = normals_zigj(1000, prng_address)
