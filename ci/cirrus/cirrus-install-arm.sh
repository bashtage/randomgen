#!/usr/bin/env bash

apt-get update -y
apt-get install build-essential git -y
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade  --upgrade pip setuptools wheel pytest pytest-xdist packaging
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip list
git fetch --tags
python3 -m pip install . --no-build-isolation -vv
python3 -m pip list
