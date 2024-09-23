class _DeprecatedValueType:
    """Special keyword value for deprecated arguments..

    The instance of this class may be used as the default value assigned to a
    keyword if the parameter is deprecated.
    """

    __instance = None

    def __new__(cls):
        # ensure that only one instance exists
        if not cls.__instance:
            cls.__instance = super().__new__(cls)
        return cls.__instance

    def __repr__(self):
        return "<deprecated>"


_DeprecatedValue = _DeprecatedValueType()

__all__ = ["_DeprecatedValue"]
