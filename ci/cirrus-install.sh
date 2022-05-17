#!/usr/bin/env bash

freebsd-update fetch
freebsd-update install

pkg install -y python39 py39-pip py39-numpy py39-cython py39-pytest
python3.9 -m pip install -e . --no-build-isolation

