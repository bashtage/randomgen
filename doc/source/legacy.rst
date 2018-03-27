Legacy Random Generation
------------------------
The :class:`~randomgen.legacy.LegacyGenerator` provides access to 
legacy generators.  These all depend on Box-Muller normals or
inverse CDF exponentials or gammas. This class should only be used 
if it is essential to have randoms that are identical to what
would have been produced by NumPy. 

:class:`~randomgen.legacy.LegacyGenerator` add additional information
to the state which is required when using Box-Muller normals since these
are produced in pairs. It is important to use 
:attr:`~randomgen.legacy.LegacyGenerator.state` 
when accessing the state so that these extra values are saved. 

.. code-block:: python
  
   from randomgen import MT19937
   from randomgen.legacy import LegacyGenerator
   from numpy.random import RandomState
      # Use same seed
   rs = RandomState(12345)
   mt19937 = MT19937(12345)
   lg = LegacyGenerator(mt19937)

   # Identical output
   rs.standard_normal()
   lg.standard_normal()

   rs.random_sample()
   lg.random_sample()

   rs.standard_exponential()
   lg.standard_exponential()
   

.. currentmodule:: randomgen.legacy

.. autoclass::
   LegacyGenerator

Seeding and State
=================

.. autosummary::
   :toctree: generated/

   ~LegacyGenerator.state
   
Simple random data
==================
.. autosummary::
   :toctree: generated/

   ~LegacyGenerator.rand
   ~LegacyGenerator.randn
   ~LegacyGenerator.randint
   ~LegacyGenerator.random_integers
   ~LegacyGenerator.random_sample
   ~LegacyGenerator.choice
   ~LegacyGenerator.bytes
   ~LegacyGenerator.random_uintegers
   ~LegacyGenerator.random_raw

Permutations
============
.. autosummary::
   :toctree: generated/

   ~LegacyGenerator.shuffle
   ~LegacyGenerator.permutation

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~LegacyGenerator.beta
   ~LegacyGenerator.binomial
   ~LegacyGenerator.chisquare
   ~LegacyGenerator.complex_normal
   ~LegacyGenerator.dirichlet
   ~LegacyGenerator.exponential
   ~LegacyGenerator.f
   ~LegacyGenerator.gamma
   ~LegacyGenerator.geometric
   ~LegacyGenerator.gumbel
   ~LegacyGenerator.hypergeometric
   ~LegacyGenerator.laplace
   ~LegacyGenerator.logistic
   ~LegacyGenerator.lognormal
   ~LegacyGenerator.logseries
   ~LegacyGenerator.multinomial
   ~LegacyGenerator.multivariate_normal
   ~LegacyGenerator.negative_binomial
   ~LegacyGenerator.noncentral_chisquare
   ~LegacyGenerator.noncentral_f
   ~LegacyGenerator.normal
   ~LegacyGenerator.pareto
   ~LegacyGenerator.poisson
   ~LegacyGenerator.power
   ~LegacyGenerator.rayleigh
   ~LegacyGenerator.standard_cauchy
   ~LegacyGenerator.standard_exponential
   ~LegacyGenerator.standard_gamma
   ~LegacyGenerator.standard_normal
   ~LegacyGenerator.standard_t
   ~LegacyGenerator.triangular
   ~LegacyGenerator.uniform
   ~LegacyGenerator.vonmises
   ~LegacyGenerator.wald
   ~LegacyGenerator.weibull
   ~LegacyGenerator.zipf