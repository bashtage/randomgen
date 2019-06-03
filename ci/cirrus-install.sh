#!/usr/bin/env bash

pkg install -y py36-pip py36-numpy py36-cython py36-pytest
python3.6 setup.py develop
