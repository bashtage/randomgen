import os
import struct
import timeit

import numpy as np
from numpy.random import RandomState
import pandas as pd

rs = RandomState()

SETUP = """
import numpy as np
if '{bitgen}' == 'numpy':
    import numpy.random
    rg = numpy.random.RandomState()
    rg.random_sample()
else:
    from randomgen import {bitgen}
    from numpy.random import Generator
    rg = Generator({bitgen}())
    rg.random()
"""

scale_32 = scale_64 = 1
if struct.calcsize("P") == 8 and os.name != "nt":
    # 64 bit
    scale_32 = 0.5
else:
    scale_64 = 2

PRNGS = [
    'AESCounter',
    'ChaCha',
    'DSFMT',
    'EFIIX64',
    'HC128',
    'JSF',
    'LCG128Mix',
    'LXM',
    'MT19937',
    'MT64',
    'MWC128X',
    'MWC128X32',
    'MWC192X',
    'MWC64X32',
    'PCG32',
    'PCG64',
    'PCG64DXSM',
    'Philox',
    'Romu',
    'SFC64',
    'SFMT',
    'SPECK128',
    'ThreeFry',
    'Xoroshiro128',
    'Xorshift1024',
    'Xoshiro256',
    'Xoshiro512',
    'numpy',
 ]

PRNGS = sorted(set(PRNGS))


def timer(code, setup):
    return 1000 * min(timeit.Timer(code, setup=setup).repeat(10, 10)) / 10.0


def print_legend(legend):
    print("\n" + legend + "\n" + "*" * max(60, len(legend)))


def add_color(val):
    color = str(2) if val > 0 else str(1)
    return "\33[38;5;" + color + "m" + f"{val:0.1f}%" + "\33[0m"


def run_timer(command, numpy_command=None, setup="", random_type=""):
    print("-" * 80)
    if numpy_command is None:
        numpy_command = command

    res = {}
    for bitgen in PRNGS:
        cmd = numpy_command if bitgen == "numpy" else command
        res[bitgen] = timer(cmd, setup=setup.format(bitgen=bitgen))

    s = pd.Series(res).sort_index()
    t = s.apply(lambda x: f"{x:0.2f} ms")
    print_legend("Time to produce 1,000,000 " + random_type)
    print(t)

    p = 1000.0 / s
    p = p.apply(lambda x: f"{x:0.2f} million")
    print_legend(random_type + " per second")
    print(p)

    baseline = "numpy"
    p = 1000.0 / s
    p = p / p[baseline] * 100 - 100
    print(p)
    p = p.drop(baseline, axis=0)
    p = p.apply(add_color)
    print_legend("Speed-up relative to NumPy")
    print(p)
    print("-" * 80)


def timer_raw():
    command = "rg._bit_generator.random_raw(size=1000000, output=False)"
    info = np.iinfo(np.int32)
    command_numpy = "rg.random_integers({max},size=1000000)"
    command_numpy = command_numpy.format(max=info.max)
    run_timer(command, command_numpy, SETUP, "Raw Values")


def timer_uniform():
    command = "rg.random(1000000)"
    command_numpy = "rg.random_sample(1000000)"
    run_timer(command, command_numpy, SETUP, "Uniforms")


def timer_bounded(bits=8, max=95, use_masked=True):
    """
    Timer for 8-bit bounded values.

    Parameters
    ----------
    bits : {8, 16, 32, 64}
        Bit width of unsigned output type
    max : int
        Upper bound for range. Lower is always 0.  Must be <= 2**bits.
    use_masked: bool
        If True, masking and rejection sampling is used to generate a random
        number in an interval. If False, Lemire's algorithm is used if
        available to generate a random number in an interval.

    Notes
    -----
    Lemire's algorithm has improved performance when {max}+1 is not a
    power of two.
    """
    if bits not in (8, 16, 32, 64):
        raise ValueError("bits must be one of 8, 16, 32, 64.")
    minimum = 0
    command = "rg.integers({min}, {max}+1, 1000000, dtype=np.uint{bits})"
    command = command.format(min=minimum, max=max, bits=bits)

    command_numpy = "rg.randint({min}, {max}+1, 1000000, dtype=np.uint{bits})"
    command_numpy = command_numpy.format(min=minimum, max=max, bits=bits)

    run_timer(
        command,
        command_numpy,
        SETUP,
        f"{bits}-bit bounded unsigned integers (max={max}, "
        f"use_masked={use_masked})",
    )


def timer_32bit():
    info = np.iinfo(np.uint32)
    minimum, maximum = info.min, info.max
    command = "rg.integers(2**32, size=1000000, dtype='uint32')"
    command_numpy = "rg.randint({min}, {max}+1, 1000000, dtype=np.uint32)"
    command_numpy = command_numpy.format(min=minimum, max=maximum)
    run_timer(command, command_numpy, SETUP, "32-bit unsigned integers")


def timer_64bit():
    info = np.iinfo(np.uint64)
    minimum, maximum = info.min, info.max
    command = "rg.integers(2**64, size=1000000, dtype='uint64')"
    command_numpy = "rg.randint({min}, {max}+1, 1000000, dtype=np.uint64)"
    command_numpy = command_numpy.format(min=minimum, max=maximum)
    run_timer(command, command_numpy, SETUP, "64-bit unsigned integers")


def timer_normal_zig():
    command = "rg.standard_normal(1000000)"
    command_numpy = "rg.standard_normal(1000000)"
    run_timer(command, command_numpy, SETUP, "Standard normals (Ziggurat)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--full",
        help="Run benchmarks for a wide range of "
        "distributions. If not provided, only tests the "
        "production of uniform values.",
        dest="full",
        action="store_true",
    )
    parser.add_argument(
        "-bi",
        "--bounded-ints",
        help="Included benchmark coverage of the bounded "
        "integer generators in a full run.",
        dest="bounded_ints",
        action="store_true",
    )
    args = parser.parse_args()

    timer_uniform()
    if args.full:
        timer_raw()
        timer_32bit()
        timer_64bit()
        timer_normal_zig()

        if args.bounded_ints:
            timer_bounded(use_masked=True)
            # Worst case for Numpy.
            timer_bounded(max=64, use_masked=False)
            # Typ. avrg. case for Numpy.
            timer_bounded(max=95, use_masked=False)
            # Best case for Numpy.
            timer_bounded(max=127, use_masked=False)

            timer_bounded(16, use_masked=True)
            # Worst case for Numpy.
            timer_bounded(16, max=1024, use_masked=False)
            # Typ. avrg. case for Numpy.
            timer_bounded(16, max=1535, use_masked=False)
            # Best case for Numpy.
            timer_bounded(16, max=2047, use_masked=False)

            timer_bounded(32, use_masked=True)
            # Worst case for Numpy.
            timer_bounded(32, max=1024, use_masked=False)
            # Typ. avrg. case for Numpy.
            timer_bounded(32, max=1535, use_masked=False)
            # Best case for Numpy.
            timer_bounded(32, max=2047, use_masked=False)

            timer_bounded(64, use_masked=True)
            # Worst case for Numpy.
            timer_bounded(64, max=1024, use_masked=False)
            # Typ. avrg. case for Numpy.
            timer_bounded(64, max=1535, use_masked=False)
            # Best case for Numpy.
            timer_bounded(64, max=2047, use_masked=False)
