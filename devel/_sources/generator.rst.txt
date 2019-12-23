Random Generator
----------------
The :class:`~randomgen.generator.Generator` provides access to
a wide range of distributions, and served as a replacement for
:class:`~numpy.random.RandomState`.  The main difference between
the two is that :class:`~randomgen.generator.Generator` relies
on an additional bit generator to manage state and generate the random
bits which are then transformed into random values from useful
distributions. The default bit generator used by
:class:`~randomgen.generator.Generator` is
:class:`~randomgen.xoroshiro128.Xoroshiro128`.  The bit generator can be
changed by passing an instantized bit generator to
:class:`~randomgen.generator.Generator`.

.. currentmodule:: randomgen.generator

.. autoclass::
   Generator

Seed and State Manipulation
===========================
.. autosummary::
   :toctree: generated/

   ~Generator.seed
   ~Generator.state
   ~Generator.bit_generator

Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~Generator.rand
   ~Generator.randn
   ~Generator.integers
   ~Generator.random
   ~Generator.choice
   ~Generator.bytes
   ~Generator.uintegers

Permutations
============
.. autosummary::
   :toctree: generated/

   ~Generator.shuffle
   ~Generator.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~Generator.beta
   ~Generator.binomial
   ~Generator.chisquare
   ~Generator.complex_normal
   ~Generator.dirichlet
   ~Generator.exponential
   ~Generator.f
   ~Generator.gamma
   ~Generator.geometric
   ~Generator.gumbel
   ~Generator.hypergeometric
   ~Generator.laplace
   ~Generator.logistic
   ~Generator.lognormal
   ~Generator.logseries
   ~Generator.multinomial
   ~Generator.multivariate_normal
   ~Generator.negative_binomial
   ~Generator.noncentral_chisquare
   ~Generator.noncentral_f
   ~Generator.normal
   ~Generator.pareto
   ~Generator.poisson
   ~Generator.power
   ~Generator.rayleigh
   ~Generator.standard_cauchy
   ~Generator.standard_exponential
   ~Generator.standard_gamma
   ~Generator.standard_normal
   ~Generator.standard_t
   ~Generator.triangular
   ~Generator.uniform
   ~Generator.vonmises
   ~Generator.wald
   ~Generator.weibull
   ~Generator.zipf