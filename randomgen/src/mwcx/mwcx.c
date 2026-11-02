#include "mwcx.h"

extern INLINE uint64_t mwcx_mul_add(uint64_t a, uint64_t x, uint64_t c,
                                    uint64_t *hi);

extern INLINE uint64_t mwc192x_next64(mwc192x_state_t *state);
extern INLINE uint32_t mwc192x_next32(mwc192x_state_t *state);

extern INLINE uint64_t mwc128x_next64(mwc128x_state_t *state);
extern INLINE uint32_t mwc128x_next32(mwc128x_state_t *state);

extern INLINE uint32_t mwc128x32_next32(mwc128x32_state_t *state);
extern INLINE uint64_t mwc128x32_next64(mwc128x32_state_t *state);
extern INLINE double mwc128x32_next_double(mwc128x32_state_t *state);

extern INLINE uint32_t mwc64x32_next32(mwc64x32_state_t *state);
extern INLINE uint64_t mwc64x32_next64(mwc64x32_state_t *state);
extern INLINE double mwc64x32_next_double(mwc64x32_state_t *state);
