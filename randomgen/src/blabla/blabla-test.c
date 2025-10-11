/*
	A C++ version of Jean-Philippe Aumasson's BlaBla PRNG (modified from Orson Peters 'chacha.h' original code)

	Changes Copyright (c) 2024 Gahtan Syarif Nahdi, licence as below.

	Changes compared to Orson Peters original code:
 		- fix formatting
		- Changed word size from 32-bits to 64-bits
		- Changed round loop increment from 2 to 1
		- Changed matrix and rotational constants
		- Changed rotation direction from left to right
		- Rename class, functions, and variables
		- Changed SIMD to support AVX2
		- Added header guards
		- Added default round value of 10
		- Added default seed value and changed the default stream
		- Added namespace
		- Changed starting block index from 0 to 1
		- Changed iterator variable type to uint
		- Revised keysetup and seed sequence generation

	Original version of the code can be found at https://gist.github.com/orlp/32f5d1b631ab092608b1
 */

/*
    Copyright (c) 2024 Orson Peters <orsonpeters@gmail.com>

    This software is provided 'as-is', without any express or implied warranty. In no event will the
    authors be held liable for any damages arising from the use of this software.

    Permission is granted to anyone to use this software for any purpose, including commercial
    applications, and to alter it and redistribute it freely, subject to the following restrictions:

    1. The origin of this software must not be misrepresented; you must not claim that you wrote the
       original software. If you use this software in a product, an acknowledgment in the product
       documentation would be appreciated but is not required.

    2. Altered source versions must be plainly marked as such, and must not be misrepresented as
       being the original software.

    3. This notice may not be removed or altered from any source distribution.
*/

#include <inttypes.h>
#include <stdbool.h>
#include <stdint.h>

#define R 20 /* Use 20 for now */
#define ALIGN_WINDOWS32 __declspec(align(32))
/*
#define ALIGN_GCC_CLANG32 __attribute__((aligned(32)))
*/
#define ALIGN_GCC_CLANG32

ALIGN_WINDOWS32 struct BLABLA_STATE_T {
    ALIGN_WINDOWS32 uint64_t block[16] ALIGN_GCC_CLANG32;
    uint64_t keysetup[4];
    uint64_t block_idx;
    uint64_t ctr;
    int rounds;
} ALIGN_GCC_CLANG;

typedef struct BLABLA_STATE_T blabla_state_t;

void seed(blabla_state_t *state, uint64_t seedval[2], uint64_t stream[2]);
uint64_t next(blabla_state_t* state);
inline void generate_block(blabla_state_t* state);
inline void blabla_core(blabla_state_t* state);

void seed(blabla_state_t* state, uint64_t seedval[2], uint64_t stream[2]) {
	state->ctr = 0;
	// ctr[1] = 0;
	state->block_idx = -1ULL;
	// block_idx[1] = -1ULL; // Block is assumed to be uninitialized.
	state->keysetup[0] = seedval[0];
	state->keysetup[1] = stream[0];
	state->keysetup[2] = seedval[1];
	state->keysetup[3] = stream[1];
}



inline void blabla_core(blabla_state_t* state) {
	#define rotate_right(x, n) ((x >> n) | (x << (64 - n)))

	#define mix_func(a, b, c, d) \
		state->block[a] += state->block[b]; state->block[d] ^= state->block[a]; state->block[d] = rotate_right(state->block[d], 32); \
		state->block[c] += state->block[d]; state->block[b] ^= state->block[c]; state->block[b] = rotate_right(state->block[b], 24); \
		state->block[a] += state->block[b]; state->block[d] ^= state->block[a]; state->block[d] = rotate_right(state->block[d], 16); \
		state->block[c] += state->block[d]; state->block[b] ^= state->block[c]; state->block[b] = rotate_right(state->block[b], 63);

	for (uint32_t i = 0; i < R; ++i) {
		mix_func(0, 4,  8, 12);
		mix_func(1, 5,  9, 13);
		mix_func(2, 6, 10, 14);
		mix_func(3, 7, 11, 15);
		mix_func(0, 5, 10, 15);
		mix_func(1, 6, 11, 12);
		mix_func(2, 7,  8, 13);
		mix_func(3, 4,  9, 14);
	}
	#undef mix_func
	#undef rotate_right
}

inline void generate_block(blabla_state_t* state) {
	uint64_t constants[4] = {0x6170786593810fab, 0x3320646ec7398aee, 0x79622d3217318274, 0x6b206574babadada};

	uint64_t input[16];
	for (uint32_t i = 0; i < 4; ++i) input[i] = constants[i];
	for (uint32_t i = 0; i < 4; ++i) input[4 + i] = state->keysetup[i];
	input[8] = 0x2ae36e593e46ad5f;
	input[9] = 0xb68f143029225fc9;
	input[10] = 0x8da1e08468303aa6;
	input[11] = 0xa48a209acd50a4a7;
	input[12] = 0x7fdc12f23f90778c;
	input[13] = state->block_idx + 1;
	input[14] = input[15] = 0; // Could be used for 192-bit counter.

	for (uint32_t i = 0; i < 16; ++i) state->block[i] = input[i];
	blabla_core(state);
	for (uint32_t i = 0; i < 16; ++i) state->block[i] += input[i];
}


uint64_t next(blabla_state_t* state){
    uint64_t next_block_idx = state->ctr / 16;
    uint64_t idx_in_block = state->ctr % 16;
	if (next_block_idx != state->block_idx) {
		state->block_idx = next_block_idx;
		generate_block(state);
	}
	state->ctr++;

    return state->block[idx_in_block];
}




int main(void) {
    blabla_state_t state;
    seed(&state, (uint64_t[]){15793235383387715774ULL, 12390638538380655177ULL}, (uint64_t[]){2361836109651742017ULL,3188717715514472916ULL});
    for (int i=0; i<1000; i++){
        printf("%" PRIu64 "\n", next(&state));
    }
    return 0;
}
