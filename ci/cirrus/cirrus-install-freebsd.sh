#!/usr/bin/env bash


pkg install -y python39  py39-numpy py39-cython wget git
python3.9 -m ensurepip --upgrade
python3.9 -m pip install --upgrade pip setuptools wheel "setuptools_scm[toml]>=7.1.0,<8.0.0" pytest pytest-xdist packaging
python3.9 -m pip list
git fetch --tags
python3.9 -m pip install . --no-build-isolation -vv
python3.9 -m pip list
