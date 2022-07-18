# RandomGen

This package contains additional bit generators for NumPy's
`Generator` and an `ExtendedGenerator` exposing methods not in `Generator`.


**Continuous Integration**

[![Build Status](https://dev.azure.com/kevinksheppard0207/kevinksheppard/_apis/build/status/bashtage.randomgen?branchName=main)](https://dev.azure.com/kevinksheppard0207/kevinksheppard/_build/latest?definitionId=2&branchName=main)
[![Appveyor Build Status](https://ci.appveyor.com/api/projects/status/46r9xsmuwi8bxsbj/branch/main?svg=true)](https://ci.appveyor.com/project/bashtage/randomgen/branch/main)
[![Build Status](https://cloud.drone.io/api/badges/bashtage/randomgen/status.svg)](https://cloud.drone.io/bashtage/randomgen)
[![FreeBSD Status on Cirrus](https://api.cirrus-ci.com/github/bashtage/randomgen.svg?branch=main)](https://cirrus-ci.com/github/bashtage/randomgen)

**Coverage**

[![codecov](https://codecov.io/gh/bashtage/randomgen/branch/main/graph/badge.svg)](https://codecov.io/gh/bashtage/randomgen)

**Latest Release**

[![PyPI version](https://badge.fury.io/py/randomgen.svg)](https://pypi.org/project/randomgen/)
[![Anacnoda Cloud](https://anaconda.org/conda-forge/randomgen/badges/version.svg)](https://anaconda.org/conda-forge/randomgen)

**License**

[![NCSA License](https://img.shields.io/badge/License-NCSA-blue.svg)](https://opensource.org/licenses/NCSA)
[![BSD License](https://img.shields.io/badge/License-BSD%203--Clause-blue.svg)](https://opensource.org/licenses/BSD-3-Clause)
[![DOI](https://zenodo.org/badge/122181085.svg)](https://zenodo.org/badge/latestdoi/122181085)

This is a library and generic interface for alternative random
generators in Python and NumPy.

## New Features

The the [development documentation](https://bashtage.github.io/randomgen/change-log.html) for the latest features,
or the [stable documentation](https://bashtage.github.io/randomgen/devel/change-log.html) for the latest released features.


# WARNINGS

## Changes in v1.24

``Generator`` and ``RandomState`` have been **removed**.

## Changes from 1.18 to 1.19

``Generator`` and ``RandomState`` have been officially deprecated in 1.19, and will
warn with a ``FutureWarning`` about their removal. They will also receive virtually
no maintenance. It is now time to move to NumPy's ``np.random.Generator`` which has
features not in ``randomstate.Generator`` and is maintained more actively.

A few distributions that are not present in ``np.random.Generator`` have been moved
to ``randomstate.ExtendedGenerator``:

* `multivariate_normal`: which supports broadcasting
* `uintegers`: fast 32 and 64-bit uniform integers
* `complex_normal`: scalar complex normals

There are no plans to remove any of the bit generators, e.g., ``AESCounter``,
``ThreeFry``, or ``PCG64``. 

### Changes from 1.16 to 1.18
There are many changes between v1.16.x and v1.18.x. These reflect API
decision taken in conjunction with NumPy in preparation of the core
of `randomgen` being used as the preferred random number generator in
NumPy. These all issue `DeprecationWarning`s except for `BasicRNG.generator`
which raises `NotImplementedError`. The C-API has also changed to reflect
the preferred naming the underlying Pseudo-RNGs, which are now known as
bit generators (or `BigGenerator`s).

## Future Plans

* Add some distributions that are not supported in NumPy. _Ongoing_
* Add any interesting bit generators I come across. _Recent additions include the DXSM and CM-DXSM variants of PCG64 and the LXM generator._

## Included Pseudo Random Number Generators

This module includes a number of alternative random
number generators in addition to the MT19937 that is included in NumPy.
The RNGs include:

* Cryptographic cipher-based random number generator based on AES, ChaCha20, HC128 and Speck128.
* [MT19937](https://github.com/numpy/numpy/blob/main/numpy/random/mtrand/),
 the NumPy rng
* [dSFMT](http://www.math.sci.hiroshima-u.ac.jp/~m-mat/MT/SFMT/) a
  SSE2-aware version of the MT19937 generator that is especially fast at
  generating doubles
* [xoroshiro128+](https://prng.di.unimi.it/),
  [xorshift1024*φ](https://prng.di.unimi.it/),
  [xoshiro256**](https://prng.di.unimi.it/),
  and [xoshiro512**](https://prng.di.unimi.it/)
* [PCG64](https://www.pcg-random.org/)
* ThreeFry and Philox from [Random123](https://www.deshawresearch.com/resources_random123.html)
* Other cryptographic-based generators: `AESCounter`, `SPECK128`, `ChaCha`, and `HC128`.
* Hardware (non-reproducible) random number generator on AMD64 using `RDRAND`.
* Chaotic PRNGS: Small-Fast Chaotic (`SFC64`) and Jenkin's Small-Fast (`JSF`).

  
## Status

* Builds and passes all tests on:
  * Linux 32/64 bit, Python 2.7, 3.5, 3.6, 3.7
  * Linux (ARM/ARM64), Python 3.7
  * OSX 64-bit, Python 2.7, 3.5, 3.6, 3.7
  * Windows 32/64 bit, Python 2.7, 3.5, 3.6, 3.7
  * FreeBSD 64-bit

## Version

The package version matches the latest version of NumPy when the package
is released.

## Documentation

Documentation for the latest release is available on
[my GitHub pages](https://bashtage.github.io/randomgen/). Documentation for
the latest commit (unreleased) is available under
[devel](https://bashtage.github.io/randomgen/devel/).


## Requirements
Building requires:

* Python (3.6, 3.7, 3.8, 3.9, 3.10)
* NumPy (1.17+)
* Cython (0.29+)
* tempita (0.5+), if not provided by Cython

Testing requires pytest (6+).

**Note:** it might work with other versions but only tested with these
versions.

## Development and Testing

All development has been on 64-bit Linux, and it is regularly tested on
Travis-CI (Linux-AMD64, Linux-PPC-LE, Linus-S390X, and OSX), Appveyor (Windows 32/64),
Cirrus (FreeBSD) and Drone.io (ARM/ARM64 Linux).

Tests are in place for all RNGs. The MT19937 is tested against
NumPy's implementation for identical results. It also passes NumPy's
test suite where still relevant.

## Installing

Either install from PyPi using

```bash
python -m pip install randomgen
```

or, if you want the latest version,

```bash
python -m pip install git+https://github.com/bashtage/randomgen.git
```

or from a cloned repo,

```bash
python -m pip install .
```

If you use conda, you can install using conda forge

```bash
conda install -c conda-forge randomgen
```

### SSE2

`dSFTM` makes use of SSE2 by default.  If you have a very old computer
or are building on non-x86, you can install using:

```bash
export RANDOMGEN_NO_SSE2=1
python -m pip install . 
```

### Windows

Either use a binary installer, or if building from scratch, use
Python 3.6/3.7 with Visual Studio 2015 Build Toolx.

## License

Dual: BSD 3-Clause and NCSA, plus sub licenses for components.
