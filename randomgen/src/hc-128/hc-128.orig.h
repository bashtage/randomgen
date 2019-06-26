/* Author: Lucas Clemente Vella
 * Source code placed into public domain. */

#pragma once

#include <inttypes.h>

typedef struct
{
  uint32_t p[512];
  uint32_t q[512];
  uint16_t i;
} hc128_state;

/** Initialize HC-128 state with key and IV.
 *
 * Contrary to the other implemented algorithms, the key and IV are taken
 * in a single function to initialize the state. This approach was chosen
 * here because of the nature of the algorithm, that keeps no intermediate
 * state between the key setting and the IV setting.
 *
 * Notice: an IV should never be reused.
 *
 * @param state The uninitialized state, it will be ready to
 * encryption/decryption afterwards.
 * @param key 16 bytes buffer containing the 128-bit key. The buffer must
 * be aligned to at least 4 bytes (depending on the platform it may or may
 * not work with unaligned memory).
 * @param iv 16 bytes buffer containing the IV.
 */
void hc128_init(hc128_state *state, const uint8_t *key, const uint8_t *iv);

/** Performs one round of the algorithm.
 *
 * @param state The algorithm state.
 * @param stream A 4 byte buffer where the generated stream will be stored.
 * Must be aligned.
 */
void hc128_extract(hc128_state *state, uint8_t *stream);
