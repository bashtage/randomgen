#ifndef RANDOMGEN_ENDIAN_H
#define RANDOMGEN_ENDIAN_H 1

#include <inttypes.h>

#ifdef _MSC_VER

#include <stdlib.h>
#define bswap_32(x) _byteswap_ulong(x)
#define bswap_64(x) _byteswap_uint64(x)

#elif defined(__APPLE__)

// Mac OS X / Darwin features
#include <libkern/OSByteOrder.h>
#define bswap_32(x) OSSwapInt32(x)
#define bswap_64(x) OSSwapInt64(x)

#elif defined(__sun) || defined(sun)

#include <sys/byteorder.h>
#define bswap_32(x) BSWAP_32(x)
#define bswap_64(x) BSWAP_64(x)

#elif defined(__FreeBSD__)

#include <sys/endian.h>
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)

#elif defined(__OpenBSD__)

#include <sys/types.h>
#define bswap_32(x) swap32(x)
#define bswap_64(x) swap64(x)

#elif defined(__NetBSD__)

#include <sys/types.h>
#include <machine/bswap.h>
#if defined(__BSWAP_RENAME) && !defined(__bswap_32)
#define bswap_32(x) bswap32(x)
#define bswap_64(x) bswap64(x)
#endif

#else

#include <byteswap.h>

#endif

#ifndef RANDOMGEN_LITTLE_ENDIAN
    #if defined(__BYTE_ORDER__)
        #if __BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__
            #define RANDOMGEN_LITTLE_ENDIAN 1
        #elif __BYTE_ORDER__ == __ORDER_BIG_ENDIAN__
            #define RANDOMGEN_LITTLE_ENDIAN 0
        #else
            #error __BYTE_ORDER__ does not match a standard endian, pick a side
        #endif
    #elif __LITTLE_ENDIAN__ || _LITTLE_ENDIAN
        #define RANDOMGEN_LITTLE_ENDIAN 1
    #elif __BIG_ENDIAN__ || _BIG_ENDIAN
        #define RANDOMGEN_LITTLE_ENDIAN 0
    #elif __x86_64 || __x86_64__ || _M_X64 || __i386 || __i386__ || _M_IX86
        #define RANDOMGEN_LITTLE_ENDIAN 1
    #elif __powerpc__ || __POWERPC__ || __ppc__ || __PPC__ \
          || __m68k__ || __mc68000__
        #define RANDOMGEN_LITTLE_ENDIAN 0
    #else
        #error Unable to determine target endianness
    #endif
#endif

static inline uint64_t ensure_le_u64(uint64_t val){
#if RANDOMGEN_LITTLE_ENDIAN
    return val;
#else
    return bswap_64(val);
#endif
}

#endif /* RANDOMGEN_ENDIAN_H */