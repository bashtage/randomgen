#!/usr/bin/env bash

python -m pip install setuptools wheel pip --upgrade
EXTRA="pytest pytest-xdist coverage pytest-cov black==20.8b1 isort flake8"

CMD="python -m pip install numpy"
if [[ -n ${NUMPY} ]]; then CMD="$CMD==${NUMPY}"; fi;
CMD="$CMD cython"
if [[ -n ${CYTHON} ]]; then CMD="$CMD==${CYTHON}"; fi;
CMD="$CMD pandas"
CMD="$CMD $EXTRA"
echo $CMD
eval $CMD
