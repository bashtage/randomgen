#include "cpu_features.h"

void feature_flags(int flags[32], int major)
{
    int i;
#if defined(HAVE_CPUID) && HAVE_CPUID
#if (defined(__clang__) || defined(__GNUC__)) && !defined(_MSC_VER)
    uint32_t num_ids = 0, reg  = 0, eax = 0, ebx = 0, ecx = 0, edx = 0;
    num_ids = __get_cpuid_max(0, &ebx);
    ebx = 0;
    if (num_ids >= 1)
    {
        __get_cpuid(1, &eax, &ebx, &ecx, &edx);
    }
#elif defined(_MSC_VER) && defined(_WIN32)
    int cpu_info[4] = {0};
    int num_ids, reg = 0, eax = 0, ebx = 0, ecx = 0, edx = 0;
    __cpuid(cpu_info, 0);
    num_ids = (int)cpu_info[0];
    if (num_ids >= 1)
    {
        __cpuidex(cpu_info, 1, 0);
        eax = cpu_info[0];
        ebx = cpu_info[1];
        ecx = cpu_info[2];
        edx = cpu_info[3];
    }
#endif
#else
    uint32_t reg, eax, ebx, ecx, edx;
    reg = 0; eax = 0; ebx = 0; ecx = 0; edx = 0;
#endif
    switch(major){
        case 0:
        reg = eax;
        break;

        case 1:
        reg = ebx;
        break;

        case 2:
        reg = ecx;
        break;

        case 3:
        reg = edx;
        break;
    }
    for (i = 0; i < 32; i++)
    {
        flags[i] = (reg >> i) & 0x1;
    }
}

int avx2_capable(void)
{
#if defined(__AVX2__) && __AVX2__
    int info[4]; // Array to hold EAX, EBX, ECX, EDX registers

    // --- 1. Check for XSAVE and OSXSAVE support (CPUID Leaf 1) ---
    // This confirms the CPU supports saving extended state and the OS has enabled it.
    __cpuidex(info, 1, 0);

    // Bit 26 (XSAVE): CPU supports XSAVE/XRSTOR instructions
    const int XSAVE_MASK = (1 << 26);
    // Bit 27 (OSXSAVE): OS has enabled support for extended state
    const int OSXSAVE_MASK = (1 << 27);

    if (!(info[2] & XSAVE_MASK) || !(info[2] & OSXSAVE_MASK)) {
        // If the CPU doesn't support XSAVE or the OS hasn't enabled OSXSAVE,
        // AVX2 cannot be used safely.
        return 0;
    }

    // --- 2. Check XCR0 register for YMM state enabling ---
    // The XCR0 register specifies which register states the OS supports saving.
    // Bit 1 (0x2) = XMM state (SSE/AVX/AVX2)
    // Bit 2 (0x4) = YMM state (AVX/AVX2)
    // We need both (0x6) for full AVX/AVX2 support.
    uint64_t xcr0 = _xgetbv(0);
    const uint64_t XCR0_AVX_MASK = 0x6;

    if ((xcr0 & XCR0_AVX_MASK) != XCR0_AVX_MASK) {
        // OS does not support saving the AVX/AVX2 (YMM) register state.
        return 0;
    }

    // --- 3. Check for AVX2 hardware feature flag (CPUID Leaf 7) ---
    // Check for the maximum supported EAX value for the main leaf (0x0)
    __cpuidex(info, 0, 0);
    if (info[0] < 7) {
        // CPU does not support Extended Features (EAX=7)
        return 0;
    }

    // Call cpuid with EAX=7 (Extended Features) and ECX=0 (Sub-leaf 0)
    __cpuidex(info, 7, 0);

    // AVX2 is indicated by bit 5 (0x20) of the EBX register (info[1])
    const int AVX2_BIT = 5;
    const int AVX2_MASK = 1 << AVX2_BIT;

    // Check if the AVX2 bit is set in EBX
    if (info[1] & AVX2_MASK) {
        // All checks passed: Hardware AVX2 + XSAVE/OSXSAVE + XCR0 YMM state
        return 1;
    } else {
        return 0; // AVX2 feature is not available
    }
#else
    return 0;
#endif

}