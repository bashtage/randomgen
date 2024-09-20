Legacy Random Generation
------------------------


.. container:: admonition danger

  .. raw:: html

      <p class="admonition-title"> Removed </p>

.. danger::

   ``RandomState`` has been completely **removed** in randomgen 2.0.0.
   You should be using :class:`numpy.random.Generator`, or if you must
   have full stability (e.g., for writing tests) or backward compatibility
   with NumPy before 1.17, :class:`numpy.random.RandomState`.
