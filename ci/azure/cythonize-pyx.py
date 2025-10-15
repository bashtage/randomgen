import subprocess

base = "cython --verbose -M --fast-fail -3 -X cpow=True -X boundscheck=False -X wraparound=False -X cdivision=True -X binding=True -X linetrace=True".split(
    " "
)

pyx_files = [
    "randomgen/aes.pyx",
    "randomgen/broadcasting.pyx",
    "randomgen/blabla.pyx",
    "randomgen/chacha.pyx",
    "randomgen/common.pyx",
    "randomgen/dsfmt.pyx",
    "randomgen/efiix64.pyx",
    "randomgen/entropy.pyx",
    "randomgen/generator.pyx",
    "randomgen/hc128.pyx",
    "randomgen/jsf.pyx",
    "randomgen/lxm.pyx",
    "randomgen/mt19937.pyx",
    "randomgen/mt64.pyx",
    "randomgen/mtrand.pyx",
    "randomgen/pcg32.pyx",
    "randomgen/pcg64.pyx",
    "randomgen/philox.pyx",
    "randomgen/rdrand.pyx",
    "randomgen/romu.pyx",
    "randomgen/sfc.pyx",
    "randomgen/sfmt.pyx",
    "randomgen/speck128.pyx",
    "randomgen/squares.pyx",
    "randomgen/threefry.pyx",
    "randomgen/tyche.pyx",
    "randomgen/wrapper.pyx",
    "randomgen/xoroshiro128.pyx",
    "randomgen/xorshift1024.pyx",
    "randomgen/xoshiro256.pyx",
    "randomgen/xoshiro512.pyx",
    "randomgen/_seed_sequence.pyx",
    "randomgen/tests/_shims.pyx",
]
for pyx_file in pyx_files:
    print(f"Cythonizing {pyx_file}")
    cmd = base + [pyx_file]
    print(" ".join(cmd))
    subprocess.call(cmd)
