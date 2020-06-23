User-defined Bit Generators
---------------------------
Most bit generators use Cython to wrap an implementation in C. While this
method leads to best case performance, it is necessarily involved.
:class:`~randomgen.wrapper.UserBitGenerator` allows bit generators to be
written in Python (slow) or numba (fast) or for existing PRNGs to be
wrapped. The bit generator can then be used with a
:class:`~numpy.random.Generator`. See `the demonstration notebook`_ for
examples.

.. module:: randomgen.wrapper

.. currentmodule:: randomgen.wrapper

.. autoclass:: UserBitGenerator

From Low-level Objects
======================

.. autosummary::
   :toctree: generated/

   ~UserBitGenerator.from_cfunc
   ~UserBitGenerator.from_ctypes

State
=====

.. autosummary::
   :toctree: generated/

   ~UserBitGenerator.state

Extending
=========
.. autosummary::
   :toctree: generated/

   ~UserBitGenerator.cffi
   ~UserBitGenerator.ctypes

Testing
=======
.. autosummary::
   :toctree: generated/

   ~UserBitGenerator.random_raw

.. _the demonstration notebook: ../custom-bit-generators.ipynb
