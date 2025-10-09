import itertools
import sys

__all__ = ["zip"]


# Remove after Python 3.10 is the minimum supported version
if sys.version_info >= (3, 10):
    zip = zip  # noqa: PLW0127
else:

    def zip(*args, strict=False):
        """Backport of zip(..., strict=...) from Python 3.10+."""
        sentinel = object()
        for combo in itertools.zip_longest(*args, fillvalue=sentinel):
            if strict and sentinel in combo:
                raise ValueError("zip() argument lengths differ")
            yield combo
