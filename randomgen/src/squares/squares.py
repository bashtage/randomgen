import numpy as np
from numpy.random import SeedSequence


class SequenceSampled:
    def __init__(self, seed_seq: SeedSequence, initial_draw=10):
        self.seed_seq = seed_seq
        self.randoms_drawm = initial_draw
        self.random_well = seed_seq.generate_state(self.randoms_drawm, dtype=np.uint64)
        self.bit_well = 0
        self.bit_well_loc = 0
        self.index = 0
        self.refill()

    def refill(self):
        self.bit_well = self.random_well[self.index]
        self.bit_well_loc = 0
        self.index += 1
        print(self.index)
        if self.index > self.random_well.shape[0]:
            self.randoms_drawm *= 2
            self.random_well = self.seed_seq.generate_state(
                self.randoms_drawm, dtype=np.uint64
            )

    def gen_bits(self, nbits):
        if self.bit_well_loc + nbits > 64:
            self.refill()
        out = self.bit_well & ((1 << nbits) - 1)
        self.bit_well >>= nbits
        self.bit_well_loc += nbits
        return out

    def random_inverval(self, max_value):
        if max_value == 1:
            return 0
        elif max_value == 2:
            return self.gen_bits(1)
        elif max_value < 4:
            bits = 2
        elif max_value < 8:
            bits = 3
        else:
            bits = 4
        while True:
            out = self.gen_bits(bits)
            if out < max_value:
                return out

    def shuffle(self, array):
        n = array.shape[0]
        for i in range(n - 1):
            j = i + self.random_inverval(n - i)
            tmp = array[i]
            array[i] = array[j]
            array[j] = tmp
        return array

    def next_word(self):
        return self.gen_bits(4)


# def generate_key(seed_seq:SeedSequence):
seed_seq = SeedSequence(0)
out = np.uint64(0)
tmp = np.empty(14, dtype=np.uint8)
val = seed_seq.generate_state(1, np.uint64)[0]
# 1. Choose a random byte from 1, 3, ..., 15, call a
loc = val & 0x7
val >>= np.uint64(3)
# 2. Store in byte 0
first = 2 * loc + 1
out |= first
# 3. Populate an array with bytes 1...15 ex a
pos = 0
for i in range(1, 16):
    if i != first:
        tmp[pos] = i
        pos += 1
ss = SequenceSampled(seed_seq, 20)
ss.shuffle(tmp)

# 4. Shuffle the array
# TODO
# 5. Store in Bytes 1...7 with first 7 locations form shuffled array
last_value = 0
for i in range(1, 8):
    out |= np.uint64(tmp[i - 1]) << np.uint64(4 * i)
    if i == 7:
        last_value = tmp[i - 1]
# 7. Choose a random number from 1,2,...,15 ex position 7
# 8. Store in byte 8
# 9. Populate bytes 9...15 with random bytes
#    using the rule that byte[i] must be different from byte[i-1]
#    and byte[i] != 0
for i in range(8, 16):
    next_val = 0
    while next_val == last_value or next_val == 0:
        next_val = ss.next_word()
    last_value = next_val
    out |= np.uint64(next_val) << np.uint64(4 * i)
print(out)
print(bin(out))
print(hex(out))
