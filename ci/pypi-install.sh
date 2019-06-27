#!/usr/bin/env bash

pip install --upgrade pip
pip install numpy cython pandas pytest setuptools nose
if [[ -z ${PPC64_LE} ]]; then pip install numpy pandas; fi
