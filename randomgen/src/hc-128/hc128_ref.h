/* This program gives the reference implementation of stream cipher HC-128

   HC-128 is a final portfolio cipher of eSTREAM, of the European Network of
   Excellence for Cryptology (ECRYPT, 2004-2008).
   The docuement of HC-128 is available at:
   1) Hongjun Wu. ``The Stream Cipher HC-128.'' New Stream Cipher Designs -- The eSTREAM Finalists, LNCS 4986, pp. 39-47, Springer-Verlag, 2008.
   2) eSTREAM website:  http://www.ecrypt.eu.org/stream/hcp3.html

   ------------------------------------
   Performance of this non-optimized implementation:

   Microprocessor: Intel CORE 2 processor (Core 2 Duo Mobile P9400 2.53GHz)
   Operating System: 32-bit Debian 5.0 (Linux kernel 2.6.26-2-686)
   Speed of encrypting long message:
   1) 6.3 cycle/byte   compiler: Intel C++ compiler 11.1   compilation option: icc -O2
   2) 3.8 cycles/byte  compiler: gcc 4.3.2                 compilation option: gcc -O3

   Microprocessor: Intel CORE 2 processor (Core 2 Quad Q6600 2.4GHz)
   Operating System: 32-bit Windows Vista Business
   Speed of encrypting long message:
   1) 6.2 cycles/byte  compiler: Intel C++ compiler 11.1    compilation option: icl /O2
   2) 6.4 cycles/byte  compiler: Microsoft Visual C++ 2008  compilation option: release

   ------------------------------------
   Written by: Hongjun Wu
   Last Modified: December 15, 2009
*/

#include <string.h>

typedef unsigned char uint8;
typedef unsigned long long uint64;

/*for LP64, "int" is 32-bit integer, while "long" is 64-bit integer*/
#if defined(_LP64)
    typedef unsigned int uint32;
#else
    typedef unsigned long uint32;
#endif

typedef struct {
      uint32 P[512];
      uint32 Q[512];
      uint32 counter1024;     /*counter1024 = i mod 1024 */
      uint32 keystreamword;   /*a 32-bit keystream word*/
} HC128_State;

#define ROTR32(x,n)   ( ((x) >> (n))  | ((x) << (32 - (n))) )
#define ROTL32(x,n)   ( ((x) << (n))  | ((x) >> (32 - (n))) )

#define f1(x)    (ROTR32((x),7) ^ ROTR32((x),18) ^ ((x) >> 3))
#define f2(x)    (ROTR32((x),17) ^ ROTR32((x),19) ^ ((x) >> 10))

/*g1 and g2 functions as defined in the HC-128 document*/
#define g1(x,y,z)  ((ROTR32((x),10)^ROTR32((z),23))+ROTR32((y),8))
#define g2(x,y,z)  ((ROTL32((x),10)^ROTL32((z),23))+ROTL32((y),8))

/*function h1*/
uint32 h1(HC128_State *state, uint32 u) {
      uint32 tem; 			
      uint8  a,c;			
      a = (uint8) ((u));		
      c = (uint8) ((u) >> 16);	
      tem = state->Q[a]+state->Q[256+c];
      return (tem);
}

/*function h2*/
uint32 h2(HC128_State *state, uint32 u) {
      uint32 tem; 			
      uint8  a,c;			
      a = (uint8) ((u));		
      c = (uint8) ((u) >> 16);	
      tem = state->P[a]+state->P[256+c];
      return (tem);
}

/* one step of HC-128:
   state is updated;
   a 32-bit keystream word is generated and stored in "state->keystreamword";
*/
void OneStep(HC128_State *state)
{
      uint32 i,i3, i10, i12, i511;

      i   = state->counter1024 & 0x1ff;
      i3  = (i - 3) & 0x1ff;
      i10 = (i - 10) & 0x1ff;
      i12 = (i - 12) & 0x1ff;
      i511 = (i - 511) & 0x1ff;

      if (state->counter1024 < 512) {
            state->P[i] = state->P[i] + g1(state->P[i3],state->P[i10],state->P[i511]);
            state->keystreamword = h1(state,state->P[i12]) ^ state->P[i];
      }
      else {
            state->Q[i] = state->Q[i] + g2(state->Q[i3],state->Q[i10],state->Q[i511]);
            state->keystreamword = h2(state,state->Q[i12]) ^ state->Q[i];
      }
      state->counter1024 = (state->counter1024+1) & 0x3ff;
}


/* one step of HC-128 in the intitalization stage:
   a 32-bit keystream word is generated to update the state;
*/
void InitOneStep(HC128_State *state)
{
      uint32 i,i3, i10, i12, i511;

      i   = state->counter1024 & 0x1ff;
      i3  = (i - 3) & 0x1ff;
      i10 = (i - 10) & 0x1ff;
      i12 = (i - 12) & 0x1ff;
      i511 = (i - 511) & 0x1ff;

      if (state->counter1024 < 512) {
            state->P[i] = state->P[i] + g1(state->P[i3],state->P[i10],state->P[i511]);
            state->P[i] = h1(state,state->P[i12]) ^ state->P[i];
      }
      else {
            state->Q[i] = state->Q[i] + g2(state->Q[i3],state->Q[i10],state->Q[i511]);
            state->Q[i] = h2(state,state->Q[i12]) ^ state->Q[i];
      }
      state->counter1024 = (state->counter1024+1) & 0x3ff;
}


/*this function initialize the state using 128-bit key and 128-bit IV*/
void Initialization(HC128_State *state, uint8 *key, uint8 *iv)
{

      uint32 W[1024+256],i;

      /*expand the key and iv into the state*/

      for (i = 0; i < 4; i++) {W[i] = ((uint32*)key)[i]; W[i+4] = ((uint32*)key)[i];}
      for (i = 0; i < 4; i++) {W[i+8] = ((uint32*)iv)[i]; W[i+12] = ((uint32*)iv)[i];}

      for (i = 16; i < 1024+256; i++) W[i] = f2(W[i-2]) + W[i-7] + f1(W[i-15]) + W[i-16]+i;

      for (i = 0; i < 512; i++)  state->P[i] = W[i+256];
      for (i = 0; i < 512; i++)  state->Q[i] = W[i+256+512];

      state->counter1024 = 0;

      /*update the cipher for 1024 steps without generating output*/
      for (i = 0; i < 1024; i++)  InitOneStep(state);
}	


/* this function encrypts a message*/
void EncryptMessage(HC128_State *state, uint8 *message, uint8 *ciphertext, uint64 msglength)
{
      uint64 i;
      uint32 j;

      /*encrypt a message, each time 4 bytes are encrypted*/
      for (i = 0; (i+4) <= msglength; i += 4, message += 4, ciphertext += 4) {
            /*generate 32-bit keystream and store it in state.keystreamword*/
            OneStep(state);
            /*encrypt 32 bits of the message*/
            ((uint32*)ciphertext)[0] = ((uint32*)message)[0] ^ state->keystreamword;
      }
      /*encrypt the last message block if the message length is not multiple of 4 bytes*/
      if ((msglength & 3) != 0) {
            OneStep(state);
            for (j = 0; j < (msglength & 3); j++) {
                  *(ciphertext+j) = *(message+j) ^ ((uint8*)&state->keystreamword)[j];
            }
      }
}


/* this function encrypts a message,
   there are four inputs to this function: a 128-bit key, a 128-bit iv, a message, the message length in bytes
   one output from this function: ciphertext
*/
void HC128(uint8 *key, uint8 *iv, uint8 *message, uint8 *ciphertext, uint64 msglength)
{
      HC128_State state;

      /*initializing the state*/
      Initialization(&state,key,iv);

      /*encrypt a message*/
      EncryptMessage(&state,message,ciphertext,msglength);
}

