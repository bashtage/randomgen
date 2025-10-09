import functools
import sys


class CustomPartial:
    def __init__(self, func, *args, **kwargs):
        self._func = func
        self._part = functools.partial(func, *args, **kwargs)
        if sys.version_info >= (3, 13):
            self._part = staticmethod(self._part)

    @property
    def func(self):
        return self._func

    def __call__(self, *args, **kwargs):
        return self._part(*args, **kwargs)
