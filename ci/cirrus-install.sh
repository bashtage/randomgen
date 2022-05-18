#!/usr/bin/env bash


pkg install -y python39  py39-numpy py39-cython wget git
python3.9 -m ensurepip --upgrade
python3.9 -m pip install wheel setuptools_scm[toml] pytest
python3.9 -m pip list
git fetch --tags
python3.9 -m pip install -e . --no-build-isolation
