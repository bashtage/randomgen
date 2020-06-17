#!/usr/bin/env bash

pip install --upgrade pip
pip install numpy cython pytest pytest-xdist setuptools --upgrade
if [[ -n ${NUMPY} ]]; then
  echo pip install numpy=="${NUMPY}" --upgrade --pre
  pip install numpy=="${NUMPY}" --upgrade --pre
fi
if [[ -n ${CYTHON} ]]; then
  echo pip install Cython=="${CYTHON}" --pre --upgrade
  pip install Cython=="${CYTHON}" --pre --upgrade
fi
if [[ -n ${PYPI_PRE} && ${PYPI_PRE} == true ]]; then
  echo install numpy Cython --pre --upgrade
  pip install numpy Cython --pre --upgrade
fi


if [[ -z ${PPC64_LE} && -z ${S390X} ]]; then
    pip install pandas --upgrade
    # Blocked on DRONE-CI
    if [[ -z ${DRONE} ]]; then
        pip install numba --upgrade
    fi
fi
