#!/usr/bin/env bash

if [[ ${USE_CONDA} == "true" ]]; then
  conda config --set always_yes true
  conda update --all --quiet
  conda create -n randomgen-test python=${PYTHON_VERSION} -y
  conda init
  source activate randomgen-test
  which python
  CMD="conda install numpy"
else
  CMD="python -m pip install cffi numpy"
fi

# Not all available in conda
python -m pip install "setuptools_scm[toml]>=9.2.0,<10"  wheel pip "black[jupyter]~=25.9.0" isort flake8 threadpoolctl meson-python ninja meson ruff --upgrade

EXTRA="pytest pytest-xdist coverage pytest-cov pytest-randomly colorama"

if [[ -n ${NUMPY} ]]; then CMD="$CMD~=${NUMPY}"; fi;
CMD="$CMD cython"
if [[ -n ${CYTHON_VER} ]]; then CMD="$CMD~=${CYTHON_VER}"; fi;
CMD="$CMD pandas"
CMD="$CMD $EXTRA"
if [[ ${USE_CONDA} == "true" ]]; then CMD="$CMD numba"; fi;
if [[ ${USE_SCIPY} == "true" ]]; then CMD="$CMD scipy"; fi;
if [[ ${USE_NUMBA} == "true" ]]; then CMD="$CMD numba==$NUMBA"; fi;
echo $CMD
eval $CMD


if [ "${PIP_PRE}" = true ]; then
  python -m pip uninstall -y numpy pandas scipy
  python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy pandas scipy --upgrade --use-deprecated=legacy-resolver
fi
