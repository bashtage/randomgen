Extended Generator
------------------
The :class:`~randomgen.generator.ExtendedGenerator` provides access to
a small number of distributions that are not present in NumPy.
The default bit generator used by
:class:`~randomgen.generator.ExtendedGenerator` is
:class:`~randomgen.pcg64.PCG64`.  The bit generator can be
changed by passing an instantized bit generator to
:class:`~randomgen.generator.ExtendedGenerator`. It is also possible
to share a bit generator with an instance of NumPy's :class:`numpy.random.Generator`.

.. currentmodule:: randomgen.generator

.. autoclass::
   ExtendedGenerator

Seed and State Manipulation
===========================
.. autosummary::
   :toctree: generated/

   ~ExtendedGenerator.state
   ~ExtendedGenerator.bit_generator

Distributions
=============
.. autosummary::
   :toctree: generated/

   ~ExtendedGenerator.uintegers
   ~ExtendedGenerator.random
   ~ExtendedGenerator.complex_normal
   ~ExtendedGenerator.multivariate_normal
   ~ExtendedGenerator.multivariate_complex_normal
   ~ExtendedGenerator.standard_wishart
   ~ExtendedGenerator.wishart
