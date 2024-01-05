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
  CMD="python -m pip install numpy"
fi

# Not all available in conda
python -m pip install setuptools "setuptools_scm[toml]<7" "oldest-supported-numpy" wheel pip black==22.6.0 isort flake8 --upgrade

EXTRA="pytest pytest-xdist coverage pytest-cov colorama"

if [[ -n ${NUMPY} ]]; then CMD="$CMD~=${NUMPY}"; fi;
CMD="$CMD cython"
if [[ -n ${CYTHON} ]]; then CMD="$CMD~=${CYTHON}"; fi;
CMD="$CMD pandas"
CMD="$CMD $EXTRA"
if [[ ${USE_CONDA} == "true" ]]; then CMD="$CMD numba"; fi;
if [[ ${USE_SCIPY} == "true" ]]; then CMD="$CMD scipy"; fi;
echo $CMD
eval $CMD


if [ "${PIP_PRE}" = true ]; then
  python -m pip uninstall -y numpy pandas scipy
  python -m pip install -i https://pypi.anaconda.org/scientific-python-nightly-wheels/simple numpy pandas scipy --upgrade --use-deprecated=legacy-resolver
fi
