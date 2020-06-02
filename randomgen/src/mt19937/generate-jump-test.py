"""
Compile jump_mt19937.c using

gcc jump_mt19937.c -O2 -o jump

or

cl jump_mt19937.c -Ox

Get the state using NumPy's state initialization

poly-128 is the 2**128 poly computed using the original author's code
clist_mt19937 is the polynomial shipped by the original author
"""
import hashlib
import os
import platform
import pprint
import shutil
import subprocess

import black
import numpy as np
from numpy.random import MT19937 as NP19937

from randomgen import MT19937

SEEDS = [0, 384908324, [839438204, 980239840, 859048019, 821]]
STEPS = [10, 312, 511]

if platform.platform() == "linux":
    EXECUTABLE = "./jump"
else:
    EXECUTABLE = "./jump_mt19937.exe"


def save_state(bit_gen, file_name):
    state = bit_gen.state
    key = state["state"]["key"]
    pos = state["state"]["pos"]
    with open(file_name, "w") as f:
        for k in key:
            f.write(f"{k}\n")
        f.write(f"{pos}\n")


def parse_output(text):
    lines = text.split("\n")

    state = {"key": [], "pos": -1}
    states = [state]
    pf = []
    for line in lines:
        parts = line.split(":")
        if "pf[" in parts[0]:
            pf.append(int(parts[1].strip()))
        elif "[" in parts[0]:
            state["key"].append(int(parts[1].strip()))
        elif ".ptr" in parts[0]:
            state["pos"] = int(parts[1].strip())
        elif "=====" in line:
            state["key"] = np.asarray(state["key"], dtype="uint32")
            state = {"key": [], "pos": -1}
            states.append(state)
    return states[:-1], pf


values = {}
for poly in ("poly-128", "clist_mt19937"):
    shutil.copy(f"{poly}.txt", "jump-poly.txt")
    fn = "_jump_tester" if poly == "clist_mt19937" else "jumped"
    for seed, step in zip(SEEDS, STEPS):
        seed_tpl = (seed,) if isinstance(seed, int) else tuple(seed)
        key = (fn, seed_tpl, step)
        values[key] = {}
        np_mt19937 = NP19937(seed)
        mt19937 = MT19937(mode="sequence")
        mt19937.state = np_mt19937.state
        mt19937.random_raw(step)
        file_name = f"state-{seed}-{step}.csv"
        save_state(mt19937, file_name=file_name)
        hash = hashlib.md5(mt19937.state["state"]["key"])
        values[key]["initial"] = {
            "key_md5": hash.hexdigest(),
            "pos": mt19937.state["state"]["pos"],
        }
        if os.path.exists("state.txt"):
            os.unlink("state.txt")
        shutil.copy(file_name, "state.txt")
        out = subprocess.run(EXECUTABLE, stdout=subprocess.PIPE)
        parsed, pf = parse_output(out.stdout.decode("utf8"))
        hash = hashlib.md5(parsed[-1]["key"])
        values[key]["jumped"] = {"key_md5": hash.hexdigest(), "pos": parsed[-1]["pos"]}
        with open(f"out-{fn}-{seed}-{step}.txt", "w") as o:
            o.write(out.stdout.decode("utf8").replace("\r\n", "\n"))
        if "128" in poly:
            jumped = mt19937.jumped()
        else:
            jumped = mt19937._jump_tester()
        hash = hashlib.md5(jumped.state["state"]["key"])
        pos = jumped.state["state"]["pos"]
        assert values[key]["jumped"] == {"key_md5": hash.hexdigest(), "pos": pos}

txt = "JUMP_TEST_DATA=" + pprint.pformat(values)
fm = black.FileMode(target_versions=black.PY36_VERSIONS)
with open("jump-test-values.txt", "w") as jt:
    jt.write(black.format_file_contents(txt, fast=False, mode=fm))
