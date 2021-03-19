#!/usr/bin/env python3
"""
This file contains an exampls of a configuration file
that can be used by practrand-driver.

Typical usage:

python practrand-driver.py -if practrand-driver-config.py | \
    ./RNG_test stdin64 -tlmax 1TB -multithreaded
"""
import numpy as np

import randomgen as rg

ENTROPY = 86316980830225721106033794313786972513572058861498566720023788662568817403978
ss = rg.SeedSequence(ENTROPY >> 128)
bg = rg.SFC64(ss)

seen = set()
remaining = NUM = 8192
while remaining:
    vals = bg.random_raw(remaining) | np.uint64(0x1)
    seen.update(vals.tolist())
    remaining = NUM - len(seen)

bitgens = []
for k in seen:
    bitgens.append(rg.SFC64(rg.SeedSequence(ENTROPY), k=k))
output = 64
