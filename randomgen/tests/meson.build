py.extension_module(
    '_shims',
    ['_shims.pyx', '_shim_dist.c', '..' / 'src' / 'aligned_malloc' / 'aligned_malloc.c'],
    install: true,
    include_directories: [inc_np, inc_np_random, include_directories('..' / 'src' / 'aligned_malloc')],
    subdir: 'randomgen/tests',
    cython_args: cython_args,
    c_args: cython_c_args,
    link_args: version_link_args,
)
