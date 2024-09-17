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
from typing import cast

import black
import numpy as np

from randomgen import MT19937

SEEDS = [0, 384908324, [839438204, 980239840, 859048019, 821]]
STEPS = [10, 312, 511]

if platform.platform() == "linux":
    EXECUTABLE = "./jump"
else:
    EXECUTABLE = "./jump_mt19937.exe"


def save_state(bit_gen: MT19937, file_name: str) -> None:
    bit_gen_state = cast(dict[str, int | np.ndarray], bit_gen.state["state"])
    state_key = cast(np.ndarray, bit_gen_state["key"])
    state_pos = bit_gen_state["pos"]
    with open(file_name, "w") as f:
        for k in state_key:
            f.write(f"{k}\n")
        f.write(f"{state_pos}\n")


def parse_output(text: str) -> tuple[list[dict[str, list | int]], list[int]]:
    lines = text.split("\n")
    key_list: list[int] = []
    output_state = {"key": key_list, "pos": -1}
    states = [output_state]
    pf = []
    for line in lines:
        parts = line.split(":")
        if "pf[" in parts[0]:
            pf.append(int(parts[1].strip()))
        elif "[" in parts[0]:
            output_state["key"].append(int(parts[1].strip()))
        elif ".ptr" in parts[0]:
            output_state["pos"] = int(parts[1].strip())
        elif "=====" in line:
            output_state["key"] = np.asarray(output_state["key"], dtype="uint32")
            output_state = {"key": [], "pos": -1}
            states.append(output_state)
    return states[:-1], pf


values: dict[tuple[str, tuple[int, ...], int], dict] = {}
for poly in ("poly-128", "clist_mt19937"):
    shutil.copy(f"{poly}.txt", "jump-poly.txt")
    fn = "_jump_tester" if poly == "clist_mt19937" else "jumped"
    for seed, step in zip(SEEDS, STEPS):
        seed_tpl = (seed,) if isinstance(seed, int) else tuple(seed)
        key = (fn, seed_tpl, step)
        values[key] = {}
        np_mt19937 = np.random.MT19937(seed)
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
fm = black.FileMode(
    target_versions={black.TargetVersion.PY37, black.TargetVersion.PY38}
)
with open("jump-test-values.txt", "w") as jt:
    jt.write(black.format_file_contents(txt, fast=False, mode=fm))
