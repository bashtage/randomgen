Future Plans
------------

A substantial portion of randomgen has been merged into NumPy. Revamping NumPy's random
number generation was always the goal of this project (and its predecessor
`NextGen NumPy RandomState <https://github.com/bashtage/ng-numpy-randomstate>`_),
and so it has succeeded.

The future plans for randomgen are:

* Remove :class:`~randomgen.generator.Generator` and :class:`~randomgen.mtrand.RandomState`. These
  duplicate NumPy and will diverge over time.  The versions in NumPy are authoritative. These
  have been deprecated as of version 1.19 and will be removed in 1.21.
* Put the novel methods of :class:`~randomgen.generator.Generator` in a
  :class:`~randomgen.generator.ExtendedGenerator`. :class:`~randomgen.generator.ExtendedGenerator`
  will be maintained, although it is possible that some of the methods may
  migrate to NumPy.
* Add useful distributions that are not supported in NumPy. Pull requests adding useful
  generators are welcome.
* Add any novel and interesting bit generators, and extend that capabilities of existing ones.

