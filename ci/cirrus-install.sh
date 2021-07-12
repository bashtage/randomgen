#!/usr/bin/env bash

pkg install -y python38 py38-pip py38-numpy py38-cython py38-pytest
python3.8 setup.py develop
