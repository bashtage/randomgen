#!/usr/bin/env bash

pip install --upgrade pip
pip install cython pytest setuptools --upgrade
if [[ -z ${NUMPY} ]]; then
  pip install numpy --upgrade
else
  pip install numpy=="${NUMPY}" --upgrade --pre -v
fi

if [[ -z ${PPC64_LE} && -z ${S390X} ]]; then
    pip install pandas --upgrade
    # Blocked on DRONE-CI
    if [[ -z ${DRONE} ]]; then
        pip install numba --upgrade
    fi
fi
