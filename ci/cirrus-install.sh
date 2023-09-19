#!/usr/bin/env bash


pkg install -y python310  py310-numpy py310-cython3 wget git
python3.10 -m ensurepip --upgrade
python3.10 -m pip install wheel "setuptools_scm[toml]>=7.1.0,<8.0.0" pytest packaging
python3.10 -m pip list
git fetch --tags
python3.10 -m pip install . --no-build-isolation
python3.10 -m pip list
