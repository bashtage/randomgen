from setuptools import Distribution, find_packages, setup
from setuptools.extension import Extension

from distutils.version import LooseVersion
import glob
import io
import os
from os.path import exists, join, splitext
import platform
import struct
import sys

from Cython.Build import cythonize
import Cython.Compiler.Options
import numpy as np

import versioneer

try:
    from Cython import Tempita as tempita
except ImportError:
    try:
        import tempita
    except ImportError:
        raise ImportError("tempita required to install, use pip install tempita")

with open("requirements.txt") as f:
    required = f.read().splitlines()

CYTHON_COVERAGE = os.environ.get("RANDOMGEN_CYTHON_COVERAGE", "0") in (
    "true",
    "1",
    "True",
)
if CYTHON_COVERAGE:
    print(
        "Building with coverage for cython modules, "
        "RANDOMGEN_CYTHON_COVERAGE=" + os.environ["RANDOMGEN_CYTHON_COVERAGE"]
    )

LONG_DESCRIPTION = io.open("README.md", encoding="utf-8").read()
Cython.Compiler.Options.annotate = True

# Make a guess as to whether SSE2 is present for now, TODO: Improve
INTEL_LIKE = any(
    [
        val in k.lower()
        for k in platform.uname()
        for val in ("x86", "i686", "i386", "amd64")
    ]
)
machine_processor = platform.machine() + platform.processor()
ARM_LIKE = any([machine_processor.startswith(name) for name in ("arm", "aarch")])
if ARM_LIKE:
    print("Processor appears to be ARM")
USE_SSE2 = INTEL_LIKE
print("Building with SSE?: {0}".format(USE_SSE2))
if "--no-sse2" in sys.argv:
    USE_SSE2 = False
    sys.argv.remove("--no-sse2")

MOD_DIR = "./randomgen"


def src_join(*fname):
    return join(MOD_DIR, "src", join(*fname))


DEBUG = os.environ.get("RANDOMGEN_DEBUG", False) in (1, "1", "True", "true")
if DEBUG:
    print("Debug build, RANDOMGEN_DEBUG=" + os.environ["RANDOMGEN_DEBUG"])

EXTRA_INCLUDE_DIRS = [np.get_include()]
EXTRA_LINK_ARGS = [] if os.name == "nt" else []
EXTRA_LIBRARIES = ["m"] if os.name != "nt" else []
# Undef for manylinux
EXTRA_COMPILE_ARGS = (
    ["/Zp16"] if os.name == "nt" else ["-std=c99", "-U__GNUC_GNU_INLINE__"]
)
UNDEF_MACROS = []
if os.name == "nt":
    EXTRA_LINK_ARGS = ["/LTCG", "/OPT:REF", "Advapi32.lib", "Kernel32.lib"]
    if DEBUG:
        EXTRA_LINK_ARGS += ["-debug"]
        EXTRA_COMPILE_ARGS += ["-Zi", "/Od"]
        UNDEF_MACROS += ["NDEBUG"]
    if sys.version_info < (3, 0):
        EXTRA_INCLUDE_DIRS += [src_join("common")]
elif DEBUG:
    EXTRA_COMPILE_ARGS += ["-g", "-O0"]
    EXTRA_LINK_ARGS += ["-g"]
    UNDEF_MACROS += ["NDEBUG"]

if Cython.__version__ >= LooseVersion("0.29"):
    DEFS = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
else:
    DEFS = [("NPY_NO_DEPRECATED_API", "0")]

if CYTHON_COVERAGE:
    DEFS += [("CYTHON_TRACE", "1"), ("CYTHON_TRACE_NOGIL", "1")]

PCG64_DEFS = DEFS[:]
if sys.maxsize < 2 ** 32 or os.name == "nt":
    # Force emulated mode here
    PCG64_DEFS += [("PCG_FORCE_EMULATED_128BIT_MATH", "1")]

DSFMT_DEFS = DEFS[:] + [("DSFMT_MEXP", "19937")]
SFMT_DEFS = DEFS[:] + [("SFMT_MEXP", "19937")]
PHILOX_DEFS = DEFS[:] + [("R123_USE_PHILOX_64BIT", "1")]
RDRAND_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:]
SSSE3_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:]
AES_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:]

if USE_SSE2:
    if os.name == "nt":
        EXTRA_COMPILE_ARGS += ["/wd4146", "/GL"]
        if struct.calcsize("P") < 8:
            EXTRA_COMPILE_ARGS += ["/arch:SSE2"]
            SSSE3_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:]
    else:
        EXTRA_COMPILE_ARGS += ["-msse2"]
        RDRAND_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:] + ["-mrdrnd"]
        SSSE3_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:] + ["-mssse3"]
        AES_COMPILE_ARGS = EXTRA_COMPILE_ARGS[:] + ["-maes"]
    DSFMT_DEFS += [("HAVE_SSE2", "1")]
    SFMT_DEFS += [("HAVE_SSE2", "1")]

files = glob.glob("./randomgen/*.in") + glob.glob("./randomgen/legacy/*.in")
for templated_file in files:
    output_file_name = splitext(templated_file)[0]
    with open(templated_file, "r") as source_file:
        template = tempita.Template(source_file.read())
    processed = template.substitute().replace("\r\n", "\n")
    contents = ""
    if exists(output_file_name):
        with open(output_file_name, "r") as output_file:
            contents = output_file.read()
    if contents != processed:
        print("Processing {0} to {1}".format(templated_file, output_file_name))
        with open(output_file_name, "w", newline="\n") as output_file:
            output_file.write(processed)

extensions = []
for name in (
    "bounded_integers",
    "common",
    "entropy",
    "generator",
    "legacy.bounded_integers",
    "mtrand",
    "_seed_sequence",
):
    extra_source = []
    extra_macros = []
    extra_incl = []

    source = ["randomgen/{0}.pyx".format(name.replace(".", "/"))]

    legacy = name in ("legacy.bounded_integers", "mtrand")
    if name in ("bounded_integers", "generator") or legacy:
        extra_source = [src_join("distributions", "distributions.c")]
        if name == "generator":
            extra_source += [
                src_join("distributions", "logfactorial.c"),
                src_join("distributions", "hypergeometric.c"),
            ]
        if legacy:
            extra_source += [src_join("legacy", "legacy-distributions.c")]
            extra_macros = [("RANDOMGEN_LEGACY", "1")]
    elif name == "entropy":
        extra_source = [src_join("entropy", "entropy.c")]
        extra_incl = [src_join("entropy")]

    ext = Extension(
        "randomgen.{0}".format(name),
        source + extra_source,
        libraries=EXTRA_LIBRARIES,
        include_dirs=EXTRA_INCLUDE_DIRS + extra_incl,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=DEFS + extra_macros,
        undef_macros=UNDEF_MACROS,
    )
    extensions.append(ext)

CPU_FEATURES = [src_join("common", "cpu_features.c")]
ALIGNED_MALLOC = [src_join("aligned_malloc", "aligned_malloc.c")]


def bit_generator(
    name,
    c_name=None,
    aligned=False,
    cpu_features=False,
    defs=None,
    compile_args=None,
    extra_source=None,
):
    c_name = name if c_name is None else c_name
    defs = DEFS if defs is None else defs

    sources = ["randomgen/{0}.pyx".format(name), src_join(c_name, c_name + ".c")]
    if cpu_features:
        sources += CPU_FEATURES
    if aligned:
        sources += ALIGNED_MALLOC
    if extra_source is not None:
        sources += [extra_source]
    compile_args = EXTRA_COMPILE_ARGS if compile_args is None else compile_args

    ext = Extension(
        "randomgen.{0}".format(name),
        sources,
        include_dirs=EXTRA_INCLUDE_DIRS,
        libraries=EXTRA_LIBRARIES,
        extra_compile_args=compile_args,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=defs,
        undef_macros=UNDEF_MACROS,
    )
    extensions.append(ext)


bit_generator(
    "aes",
    c_name="aesctr",
    cpu_features=True,
    aligned=True,
    compile_args=AES_COMPILE_ARGS,
)
bit_generator(
    "chacha", cpu_features=True, aligned=True, compile_args=SSSE3_COMPILE_ARGS
)
bit_generator(
    "dsfmt",
    aligned=True,
    defs=DSFMT_DEFS,
    extra_source=src_join("dsfmt", "dSFMT-jump.c"),
)
bit_generator("hc128", c_name="hc-128")
bit_generator("jsf")
bit_generator("mt19937", extra_source=src_join("mt19937", "mt19937-jump.c"))
bit_generator("mt64")
bit_generator("pcg32")
# PCG requires special treatment since it contains multiple bit gens
ext = Extension(
    "randomgen.pcg64",
    ["randomgen/pcg64.pyx"]
    + [
        src_join("pcg64", "pcg64-common.c"),
        src_join("pcg64", "pcg64-v2.c"),
        src_join("pcg64", "lcg128mix.c"),
    ],
    libraries=EXTRA_LIBRARIES,
    include_dirs=EXTRA_INCLUDE_DIRS,
    extra_compile_args=EXTRA_COMPILE_ARGS,
    extra_link_args=EXTRA_LINK_ARGS,
    define_macros=DEFS,
    undef_macros=UNDEF_MACROS,
)
extensions.append(ext)

bit_generator("philox", defs=PHILOX_DEFS)
bit_generator("rdrand", cpu_features=True, compile_args=RDRAND_COMPILE_ARGS)
bit_generator(
    "sfmt", aligned=True, defs=SFMT_DEFS, extra_source=src_join("sfmt", "sfmt-jump.c")
)
bit_generator(
    "speck128",
    c_name="speck-128",
    cpu_features=True,
    aligned=True,
    compile_args=SSSE3_COMPILE_ARGS,
)
bit_generator("threefry")
bit_generator("xoroshiro128")
bit_generator("xorshift1024")
bit_generator("xoshiro256")
bit_generator("xoshiro512")
bit_generator("lxm")
bit_generator("sfc")
bit_generator("efiix64")
bit_generator("romu")
extensions.append(
    Extension(
        "randomgen.wrapper",
        ["randomgen/wrapper.pyx"],
        include_dirs=EXTRA_INCLUDE_DIRS,
        libraries=EXTRA_LIBRARIES,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=DEFS,
        undef_macros=UNDEF_MACROS,
    )
)
extensions.append(
    Extension(
        "randomgen.tests._shims",
        ["randomgen/tests/_shims.pyx"],
        include_dirs=EXTRA_INCLUDE_DIRS,
        libraries=EXTRA_LIBRARIES,
        extra_compile_args=EXTRA_COMPILE_ARGS,
        extra_link_args=EXTRA_LINK_ARGS,
        define_macros=DEFS,
        undef_macros=UNDEF_MACROS,
    )
)

classifiers = [
    "Development Status :: 5 - Production/Stable",
    "Environment :: Console",
    "Intended Audience :: End Users/Desktop",
    "Intended Audience :: Financial and Insurance Industry",
    "Intended Audience :: Information Technology",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved",
    "Operating System :: MacOS :: MacOS X",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
    "Operating System :: Unix",
    "Programming Language :: C",
    "Programming Language :: Cython",
    "Programming Language :: Python :: 3.6",
    "Programming Language :: Python :: 3.7",
    "Programming Language :: Python :: 3.8",
    "Topic :: Adaptive Technologies",
    "Topic :: Artistic Software",
    "Topic :: Office/Business :: Financial",
    "Topic :: Scientific/Engineering",
    "Topic :: Security :: Cryptography",
]


class BinaryDistribution(Distribution):
    def is_pure(self):
        return False


setup(
    name="randomgen",
    version=versioneer.get_version(),
    classifiers=classifiers,
    cmdclass=versioneer.get_cmdclass(),
    ext_modules=cythonize(
        extensions,
        compiler_directives={"language_level": "3", "linetrace": CYTHON_COVERAGE},
        force=CYTHON_COVERAGE or DEBUG,
        gdb_debug=DEBUG,
    ),
    packages=find_packages(),
    package_dir={"randomgen": "./randomgen"},
    package_data={
        "": ["*.h", "*.pxi", "*.pyx", "*.pxd", "*.in"],
        "randomgen.tests.data": ["*.csv"],
    },
    include_package_data=True,
    license="NCSA",
    author="Kevin Sheppard",
    author_email="kevin.k.sheppard@gmail.com",
    distclass=BinaryDistribution,
    long_description=LONG_DESCRIPTION,
    long_description_content_type="text/markdown",
    description="Random generator supporting multiple PRNGs",
    url="https://github.com/bashtage/randomgen",
    keywords=[
        "pseudo random numbers",
        "PRNG",
        "RNG",
        "RandomState",
        "random",
        "random numbers",
        "parallel random numbers",
        "PCG",
        "XorShift",
        "dSFMT",
        "MT19937",
        "Random123",
        "ThreeFry",
        "Philox",
        "ChaCha",
        "AES",
        "SPECK",
        "RDRAND",
    ],
    zip_safe=False,
    install_requires=required,
    python_requires='>=3',
)
