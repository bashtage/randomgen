Seed Sequences
--------------

.. module:: randomgen._seed_sequence

.. currentmodule:: randomgen._seed_sequence

.. note::

  randomgen imports ``SeedSequece`` from NumPy if available, and only falls back to a vendored
  copy if not.  The correct import location is ``randomgen.seed_sequence`` which handles to
  selection of the correct ``SeedSequence``.

.. autoclass:: SeedSequence

Using a SeedSequence
====================

.. autosummary::
   :toctree: generated/

   ~SeedSequence.generate_state
   ~SeedSequence.spawn

State
=====

.. autosummary::
   :toctree: generated/

   ~SeedSequence.state