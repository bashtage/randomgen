/*
 * A simple file that can be used to read long double sizes on different
 * platforms
 */
#include <float.h>
#include <stdio.h>

int main(){
    printf("Mantissa: %d, ", LDBL_MANT_DIG+0);
    printf("Exponent %d\n", LDBL_MAX_EXP+0);
}

