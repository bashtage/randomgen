#!/usr/bin/env bash


pkg install -y python39  py39-numpy py39-cython wget git
python3.9 -m ensurepip --upgrade
# wget https://bootstrap.pypa.io/get-pip.py
# python get-pip.py
python3.9 -m pip install pytest wheel --user
python3.9 -m pip install . --user
# python3.9 -m pip wheel . -w wheelhouse/
# WHL=$(ls -t wheelhouse/randomgen-*.whl | head -n1)
# python3.9 -m pip install install ${WHL} --user
