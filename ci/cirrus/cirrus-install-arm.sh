#!/usr/bin/env bash

apt-get update -y
apt-get install build-essential -y
python3 -m ensurepip --upgrade
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 -m pip install -r requirements-dev.txt
python3 -m pip list
git fetch --tags
python3 -m pip install . --no-build-isolation
python3 -m pip list
