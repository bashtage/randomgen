from randomgen.common cimport (
    byteswap_little_endian,
    int_to_array,
    object_to_int,
    view_little_endian,
)

import numpy as np


def view_little_endian_shim(arr, dtype):
    return view_little_endian(arr, dtype)


def int_to_array_shim(value, name, bits, uint_size):
    return np.asarray(int_to_array(value, name, bits, uint_size))


def byteswap_little_endian_shim(arr):
    return byteswap_little_endian(arr)

def object_to_int_shim(val, bits, name, default_bits=64, allowed_sizes=(64,)):
    return object_to_int(val, bits, name, default_bits, allowed_sizes)
