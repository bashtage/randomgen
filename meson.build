project(
  'randomgen',
  'c', 'cython',
  license: 'NCSA OR BSD-3-Clause',
  meson_version: '>= 1.9.0',
  default_options: [],
  version:  run_command(['randomgen/_build/git_version.py'], check: true).stdout().strip(),
)

cc = meson.get_compiler('c')
cy = meson.get_compiler('cython')
cython = find_program(cy.cmd_array()[0])

if not cy.version().version_compare('>=3.1.0')
  error('randomgen requires Cython >= 3.1.0')
endif

py = import('python').find_installation(pure: false)
fs = import('fs')

subdir('randomgen')
