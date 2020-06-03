#!/usr/bin/env bash

pip install --upgrade pip
pip install cython pytest setuptools --upgrade
if [[ -z ${NUMPY} ]]; then
  pip install numpy pandas
else
  pip install numpy=="${NUMPY}" --upgrade --pre
fi
pip install pandas

if [[ -z ${PPC64_LE} ]]; then pip install numpy pandas; fi
