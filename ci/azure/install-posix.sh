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
python -m pip install setuptools wheel pip black==20.8b1 isort flake8 --upgrade

EXTRA="pytest pytest-xdist coverage pytest-cov"

if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD cython"
if [[ -n ${CYTHON} ]]; then CMD="$CMD==${CYTHON}"; fi;
CMD="$CMD pandas"
CMD="$CMD $EXTRA"
if [[ ${USE_CONDA} == "true" ]]; then CMD="$CMD numba"; fi;
if [[ ${USE_SCIPY} == "true" ]]; then CMD="$CMD scipy"; fi;
echo $CMD
eval $CMD
