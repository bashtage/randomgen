#!/usr/bin/env bash

pip install --upgrade pip
pip install cython pytest setuptools --upgrade
if [[ -z ${NUMPY} ]]; then
  echo pip install numpy pandas
else
  echo pip install numpy=="${NUMPY}" pandas
fi

if [[ -z ${PPC64_LE} ]]; then pip install numpy pandas; fi
