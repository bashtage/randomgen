#!/usr/bin/env bash


pkg install -y python39  py39-numpy py39-cython wget
python3.9 -m ensurepip --upgrade
# wget https://bootstrap.pypa.io/get-pip.py
# python get-pip.py
python3.9 -m pip install pytest wheel --user
python3.9 -m pip install -e . --user
