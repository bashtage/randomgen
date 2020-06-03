#!/usr/bin/env bash

pkg install -y python37 py37-pip py37-numpy py37-cython py37-pytest
python3.7 setup.py develop
