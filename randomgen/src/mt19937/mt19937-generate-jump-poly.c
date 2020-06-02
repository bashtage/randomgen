/*
 * Requires a modern NTL and GMP.
 *
 * Assuming you have a standard config
 *
 * g++ mt19937-generate-jump-poly.c -I /usr/local/include/NTL -lntl -lm -pthread
 * -lgmp -o jump-poly
 *
 */

#include <NTL/GF2X.h>
#include <NTL/vec_GF2.h>

#include <cstdlib>
#include <fstream>

NTL_CLIENT

/* parameters for MT19937 */
#define N 624
#define M 397
#define MATRIX_A 0x9908b0dfUL   /* constant vector a */
#define UPPER_MASK 0x80000000UL /* most significant w-r bits */
#define LOWER_MASK 0x7fffffffUL /* least significant r bits */

static unsigned long mt[N]; /* the array for the state vector  */
static int mti = N + 1;     /* mti==N+1 means mt[N] is not initialized */

/* Parameter for computing the minimal polynomial */
#define MEXP 19937 /* the dimension of the state space */

GF2X phi; /* phi is the minimal polynomial */
GF2X g;   /* g(t) is used to store t^J mod phi(t) */

/* initializes mt[N] with a seed */
void init_genrand(unsigned long s)
{
  mt[0] = s & 0xffffffffUL;
  for (mti = 1; mti < N; mti++)
  {
    mt[mti] = (1812433253UL * (mt[mti - 1] ^ (mt[mti - 1] >> 30)) + mti);
    /* See Knuth TAOCP Vol2. 3rd Ed. P.106 for multiplier. */
    /* In the previous versions, MSBs of the seed affect   */
    /* only MSBs of the array mt[].                        */
    /* 2002/01/09 modified by Makoto Matsumoto             */
    mt[mti] &= 0xffffffffUL;
    /* for >32 bit machines */
  }
}

/* initialize by an array with array-length */
/* init_key is the array for initializing keys */
/* key_length is its length */
/* slight change for C++, 2004/2/26 */
void init_by_array(unsigned long init_key[], int key_length)
{
  int i, j, k;
  init_genrand(19650218UL);
  i = 1;
  j = 0;
  k = (N > key_length ? N : key_length);
  for (; k; k--)
  {
    mt[i] =
        (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1664525UL)) + init_key[j] + j; /* non linear */
    mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
    i++;
    j++;
    if (i >= N)
    {
      mt[0] = mt[N - 1];
      i = 1;
    }
    if (j >= key_length)
      j = 0;
  }
  for (k = N - 1; k; k--)
  {
    mt[i] = (mt[i] ^ ((mt[i - 1] ^ (mt[i - 1] >> 30)) * 1566083941UL)) - i; /* non linear */
    mt[i] &= 0xffffffffUL; /* for WORDSIZE > 32 machines */
    i++;
    if (i >= N)
    {
      mt[0] = mt[N - 1];
      i = 1;
    }
  }

  mt[0] = 0x80000000UL; /* MSB is 1; assuring non-zero initial array */
}

/* generates a random number on [0,0xffffffff]-interval */
unsigned long genrand_int32(void)
{
  unsigned long y;
  static unsigned long mag01[2] = {0x0UL, MATRIX_A};
  /* mag01[x] = x * MATRIX_A  for x=0,1 */

  if (mti >= N)
  { /* generate N words at one time */
    int kk;

    if (mti == N + 1)       /* if init_genrand() has not been called, */
      init_genrand(5489UL); /* a default initial seed is used */

    for (kk = 0; kk < N - M; kk++)
    {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + M] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    for (; kk < N - 1; kk++)
    {
      y = (mt[kk] & UPPER_MASK) | (mt[kk + 1] & LOWER_MASK);
      mt[kk] = mt[kk + (M - N)] ^ (y >> 1) ^ mag01[y & 0x1UL];
    }
    y = (mt[N - 1] & UPPER_MASK) | (mt[0] & LOWER_MASK);
    mt[N - 1] = mt[M - 1] ^ (y >> 1) ^ mag01[y & 0x1UL];

    mti = 0;
  }

  y = mt[mti++];

  /* Tempering */
  y ^= (y >> 11);
  y ^= (y << 7) & 0x9d2c5680UL;
  y ^= (y << 15) & 0xefc60000UL;
  y ^= (y >> 18);

  return y;
}

/* computes the minimal polynomial of the linear recurrence */
void comp_mini_poly(void)
{
  int i;
  vec_GF2 v(INIT_SIZE, 2 * MEXP);

  for (i = 0; i < 2 * MEXP; i++) v[i] = genrand_int32() & 0x01ul;

  MinPolySeq(phi, v, MEXP);
}

/* computes the t^J mod phi(t) */
void comp_jump_rem(ZZ jump_step)
{
  /* changed by saito 2013.1.25 */
  // GF2X f;
  //  SetCoeff (f, jump_step, 1);
  //  g = f % phi;
  PowerXMod(g, jump_step, phi);
  /* changed by saito 2013.1.25 */
}

int main(void)
{
  int k, i, a = 0;
  char steps[7][45] = {"1234567",
                       "340282366920938463463374607431768211456",
                       "5444517870735015415413993718908291383296",
                       "87112285931760246646623899502532662132736",
                       "1393796574908163946345982392040522594123776",
                       "22300745198530623141535718272648361505980416",
                       "210306068529402873165736369884012333109"};
  char file_names[7][14] = {"clist-mt.txt", "poly-128.txt", "poly-132.txt", "poly-136.txt",
                            "poly-140.txt", "poly-144.txt", "poly-phi.txt"};
  for (k = 0; k < 7; k++)
  {
    cout << k << ": " << file_names[k] << "\n";
    ZZ jump_step = conv<ZZ>(steps[k]); /* the number of steps of jumping ahead */
    unsigned long init[4] = {0x123, 0x234, 0x345, 0x456}, length = 4;
    ofstream fout;

    init_by_array(init, length);

    comp_mini_poly();
    comp_jump_rem(jump_step);

    fout.open(file_names[k], ios::out);
    if (!fout)
      return -1;
    for (i = MEXP - 1; i > -1; i--) {
      fout << coeff(g, i);
    }

    fout.close();
  }

  return 0;
}
