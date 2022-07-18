Future Plans
------------

A substantial portion of randomgen has been merged into NumPy. Revamping NumPy's random
number generation was always the goal of this project (and its predecessor
`NextGen NumPy RandomState <https://github.com/bashtage/ng-numpy-randomstate>`_),
and so it has succeeded.

The future plans for randomgen are:

* ``Generator`` and ``RandomState`` have been **removed** in 1.23.
* Put the novel methods of ``Generator`` in a
  :class:`~randomgen.generator.ExtendedGenerator`. :class:`~randomgen.generator.ExtendedGenerator`
  will be maintained, although it is possible that some of the methods may
  migrate to NumPy.
* Add useful distributions that are not supported in NumPy. Pull requests adding useful
  generators are welcome.
* Add any novel and interesting bit generators, and extend that capabilities of existing ones.

