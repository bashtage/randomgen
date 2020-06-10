from randomgen.common cimport view_little_endian, int_to_array, byteswap_little_endian
import numpy as np


def view_little_endian_shim(arr, dtype):
    return view_little_endian(arr, dtype)


def int_to_array_shim(value, name, bits, uint_size):
    return np.asarray(int_to_array(value, name, bits, uint_size))


def byteswap_little_endian_shim(arr):
    return byteswap_little_endian(arr)
