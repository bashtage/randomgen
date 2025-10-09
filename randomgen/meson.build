incdir_numpy = run_command(
    py,
    [
        '-c',
        '''
import os
cwd = os.getcwd()

# Protect from import errors due to module names
os.chdir(os.path.join('..', 'tools'))
import numpy as np
os.chdir(cwd)

try:
    # Check if include directory is inside the dir
    # e.g. a venv created inside the dir
    # If so, convert it to a relative path
    incdir = os.path.relpath(np.get_include())
except Exception as exc:
    incdir = np.get_include()
print(incdir)
     ''',
    ],
    check: true,
).stdout().strip()

incdir_numpy_abs = run_command(py,
['-c', 'import os; os.chdir(".."); import numpy; print(numpy.get_include())'],
check: true
).stdout().strip()
inc_np = include_directories(incdir_numpy)
inc_np_random = include_directories(incdir_numpy / 'numpy' / 'random')
inc_aligned_malloc = include_directories('src/aligned_malloc')
npymath_path = incdir_numpy_abs / '..' / 'lib'
npyrandom_path = incdir_numpy_abs / '..' / '..' / 'random' / 'lib'
npymath_lib = cc.find_library('npymath', dirs: npymath_path)
npyrandom_lib = cc.find_library('npyrandom', dirs: npyrandom_path)
# Copy the main __init__.py to the build dir.
# Some submodules (linalg, special, optimize) add pxd files to this.
# Needed to trick Cython, it won't do a relative import outside a package
_cython_tree = [fs.copyfile('__init__.py')]

# TODO: Check these flags
# TODO: Enable cython coverage in CI
# "language_level": "3",
# "binding": True,
# "linetrace": CYTHON_COVERAGE,

_pcg64_def = []
if get_option('pcg64-force-emulated')
  _pcg64_def += ['-DPCG_FORCE_EMULATED_128BIT_MATH=1']
endif

cython_args = [
    '-X cpow=True',
    '-X boundscheck=False',
    '-X wraparound=False',
    '-X cdivision=True',
    '-X binding=True'
]
cython_c_args =  ['-DNPY_NO_DEPRECATED_API=NPY_1_7_API_VERSION']
cython_libs = []
sse2_def = []
if cc.get_id() == 'msvc' or cc.get_id() == 'clang-cl'
  cython_c_args += cc.get_supported_arguments(['/Zp16','/wd4146', '/GL','/arch:SSE4.2'])
  if cc.has_argument('/arch:SSE4.2')
    sse2_def += ['-DHAVE_SSE2=1']
  endif
  add_project_link_arguments('/LTCG', language : 'c')
  add_project_link_arguments('/OPT:REF', language : 'c')
  add_project_link_arguments('Advapi32.lib', language : 'c')
  add_project_link_arguments('Kernel32.lib', language : 'c')
else
  cython_c_args += ['-std=c99', '-U__GNUC_GNU_INLINE__']
  add_project_link_arguments('-lm', language : 'c')
  cython_c_args += cc.get_supported_arguments(['-msse2', '-mrdrnd', '-mssse3', '-maes'])
  if cc.has_argument('-msse2')
    sse2_def += ['-DHAVE_SSE2=1']
  endif
endif

if get_option('cython-coverage')
  message('Building with Cython coverage support')
  cython_args += ['-X linetrace=True']
  cython_c_args += ['-DCYTHON_TRACE=1']
endif


rg_dir = py.get_install_dir() / 'randomgen'
# Generate version.py for sdist
meson.add_dist_script(
   ['_build/git_version.py', '--meson-dist', '--write',
     'randomgen/_version.py']
)
if not fs.exists('_version.py')
  generate_version = custom_target(
    'generate-version',
    install: true,
    build_always_stale: true,
    build_by_default: true,
    output: '_version.py',
    input: '_build/git_version.py',
    command: [py, '@INPUT@', '--write', '@OUTPUT@'],
    install_dir: rg_dir,
    install_tag: 'python-runtime'
  )
else
  # When building from sdist, version.py exists and should be included
  py.install_sources(
    ['_version.py'],
    subdir : 'randomgen'
  )
endif



inc_aligned_malloc = include_directories('src' / 'aligned_malloc')
CPU_FEATURES = ['src/common/cpu_features.c']
ALIGNED_MALLOC = ['src/aligned_malloc/aligned_malloc.c']

standard_bit_generators = [
    ['mt64', ['mt64.pyx', 'src/mt64/mt64.c'], 'src/mt64'],
    ['pcg32', ['pcg32.pyx', 'src/pcg32/pcg32.c'], 'src/pcg32'],
    ['threefry', ['threefry.pyx', 'src/threefry/threefry.c'], 'src/threefry'],
    ['xoroshiro128', ['xoroshiro128.pyx', 'src/xoroshiro128/xoroshiro128.c'], 'src/xoroshiro128'],
    ['xorshift1024', ['xorshift1024.pyx', 'src/xorshift1024/xorshift1024.c'], 'src/xorshift1024'],
    ['xoshiro256', ['xoshiro256.pyx', 'src/xoshiro256/xoshiro256.c'], 'src/xoshiro256'],
    ['xoshiro512', ['xoshiro512.pyx', 'src/xoshiro512/xoshiro512.c'], 'src/xoshiro512'],
    ['lxm', ['lxm.pyx', 'src/lxm/lxm.c'], 'src/lxm'],
    ['jsf', ['jsf.pyx', 'src/jsf/jsf.c'], 'src/jsf'],
    ['sfc', ['sfc.pyx', 'src/sfc/sfc.c'], 'src/sfc'],
    ['efiix64', ['efiix64.pyx', 'src/efiix64/efiix64.c'], 'src/efiix64'],
    ['romu', ['romu.pyx', 'src/romu/romu.c'], 'src/romu'],
    ['tyche', ['tyche.pyx', 'src/tyche/tyche.c'], 'src/tyche'],
    ['hc128', ['hc128.pyx', 'src/hc-128/hc-128.c'], 'src/hc-128'],
]
foreach gen: standard_bit_generators
    py.extension_module(
        gen[0],
        gen[1],
        install: true,
        include_directories: [inc_np, inc_aligned_malloc, include_directories(gen[2])],
        subdir: 'randomgen',
        cython_args: cython_args,
        c_args: cython_c_args,
    )
endforeach

subdirs_list = [
    '_build',
    'compat',
    'examples',
    'legacy',
    'src',
    'tests',
]

foreach subdir : subdirs_list
    install_subdir(subdir, install_dir: py.get_install_dir() / 'randomgen')
endforeach


py.extension_module(
    'aes',
    ['aes.pyx', 'src/aesctr/aesctr.c'] + CPU_FEATURES + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/aesctr')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'chacha',
    ['chacha.pyx', 'src/chacha/chacha.c'] + CPU_FEATURES + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/chacha')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'dsfmt',
    ['dsfmt.pyx', 'src/dsfmt/dsfmt.c', 'src/dsfmt/dSFMT-jump.c'] + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/dsfmt')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args + ['-DDSFMT_MEXP=19937'] + sse2_def,
)
py.extension_module(
    'mt19937',
    ['mt19937.pyx', 'src/mt19937/mt19937.c', 'src/mt19937/mt19937-jump.c'],
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/mt19937')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'philox',
    ['philox.pyx', 'src/philox/philox.c'],
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/philox')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args + ['-DR123_USE_PHILOX_64BIT=1'],
)
py.extension_module(
    'rdrand',
    ['rdrand.pyx', 'src/rdrand/rdrand.c'] + CPU_FEATURES,
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/rdrand')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'sfmt',
    ['sfmt.pyx', 'src/sfmt/sfmt.c', 'src/sfmt/sfmt-jump.c'] + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/sfmt')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args + ['-DSFMT_MEXP=19937'] + sse2_def,
)
py.extension_module(
    'speck128',
    ['speck128.pyx', 'src/speck-128/speck-128.c'] + ALIGNED_MALLOC + CPU_FEATURES,
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, include_directories('src/speck-128')],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args + ['-DDSFMT_MEXP=19937'],
)
py.extension_module(
    'common',
    ['common.pyx'] + CPU_FEATURES + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_np_random, inc_aligned_malloc],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    '_seed_sequence',
    ['_seed_sequence.pyx'] + CPU_FEATURES + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_np_random],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'broadcasting',
    ['broadcasting.pyx'] + CPU_FEATURES + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_np_random],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'generator',
    ['generator.pyx', 'src/distributions/rg-distributions.c'] + CPU_FEATURES + ALIGNED_MALLOC,
    install: true,
    include_directories: [inc_np, inc_np_random, inc_aligned_malloc],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
    dependencies: [npymath_lib, npyrandom_lib],
)
py.extension_module(
    'entropy',
    ['entropy.pyx', 'src/entropy/entropy.c'],
    install: true,
    include_directories: [inc_np, inc_np_random, 'src/entropy'],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'wrapper',
    ['wrapper.pyx'],
    install: true,
    include_directories: [inc_np, inc_np_random, inc_aligned_malloc],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'pcg64',
    ['pcg64.pyx','src/pcg64/pcg64-common.c','src/pcg64/pcg64-v2.c','src/pcg64/lcg128mix.c'],
    install: true,
    include_directories: [inc_np, inc_aligned_malloc, inc_np_random],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)
py.extension_module(
    'squares',
    ['squares.pyx'],
    install: true,
    include_directories: [inc_np, inc_np_random, inc_aligned_malloc],
    subdir: 'randomgen',
    cython_args: cython_args,
    c_args: cython_c_args,
)

subdir('tests')

top_level_py_list = [
    '__init__.py',
    '_deprecated_value.py',
    '_pickle.py',
    '_register.py',
    'conftest.py',
    'seed_sequence.py',
    'typing.py',
    '_seed_sequence.pyi',
    'aes.pyi',
    'chacha.pyi',
    'common.pyi',
    'dsfmt.pyi',
    'efiix64.pyi',
    'entropy.pyi',
    'generator.pyi',
    'hc128.pyi',
    'jsf.pyi',
    'lxm.pyi',
    'mt19937.pyi',
    'mt64.pyi',
    'mtrand.pyi',
    'pcg32.pyi',
    'pcg64.pyi',
    'philox.pyi',
    'rdrand.pyi',
    'romu.pyi',
    'sfc.pyi',
    'sfmt.pyi',
    'speck128.pyi',
    'threefry.pyi',
    'tyche.pyi',
    'wrapper.pyi',
    'xoroshiro128.pyi',
    'xorshift1024.pyi',
    'xoshiro256.pyi',
    'xoshiro512.pyi',
]
py.install_sources(top_level_py_list, subdir: 'randomgen')
