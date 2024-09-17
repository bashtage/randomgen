/*
 * clang++ speck-test-data-gen.cpp cryptoPP/libcryptopp.a -o speck-test-data-gen
 * ./speck-test-data-gen
 */

#include <iomanip>
#include <iostream>

#include "cryptoPP/cryptlib.h"
#include "cryptoPP/files.h"
#include "cryptoPP/filters.h"
#include "cryptoPP/hex.h"
#include "cryptoPP/modes.h"
#include "cryptoPP/speck.h"

using namespace std;
using namespace CryptoPP;

#define N 1000

int main(int argc, char *argv[]) {
  uint64_t seed, state, seeded_key[4], result[2], ctr[2] = {0, 0};
  seed = state = 0;
  int loc;
  /* SeedSequence(0).generate_state(4, dtype=np.uint64) */
  uint64_t seed_seq[4] = {15793235383387715774, 12390638538380655177, 2361836109651742017, 3188717715514472916};
  for (int i = 0; i < 4; i++) {
    seeded_key[i] = seed_seq[i];
  }
  CryptoPP::byte key[SPECK128::MAX_KEYLENGTH];
  CryptoPP::byte cipher[16], counter[16];
  memcpy(&key, &seeded_key, sizeof(key));
  memset(counter, 0x00, sizeof(cipher));
  memset(cipher, 0x00, sizeof(cipher));
  ECB_Mode<SPECK128>::Encryption speck;
  speck.SetKey(key, SPECK128::MAX_KEYLENGTH);

  std::ofstream ofile;
  ofile.open("speck-128-testset-1.csv");
  ofile << "seed, " << 0 << endl;
  loc = 0;
  for (int j = 0; j < N / 2; j++) {
    ctr[0] = j;
    memcpy(&counter, &ctr, sizeof(ctr));
    speck.ProcessData(cipher, counter, 16);
    memcpy(&result, &cipher, sizeof(result));
    for (int i = 0; i < 2; i++) {
      ofile << loc << ", " << result[i] << std::endl;
      loc++;
    }
  }
  ofile.close();

  seed = state = 0xDEADBEAF;
  /* SeedSequence(0xDEADBEAF).generate_state(4, dtype=np.uint64) */
  uint64_t seed_seq_deadbeaf[4] = {5778446405158232650, 4639759349701729399,
                          13222832537653397986, 2330059127936092250};
  for (int i = 0; i < 4; i++) {
    seeded_key[i] = seed_seq_deadbeaf[i];
  }
  memcpy(&key, &seeded_key, sizeof(key));
  speck.SetKey(key, SPECK128::MAX_KEYLENGTH);
  ofile.open("speck-128-testset-2.csv");
  ofile << "seed, " << seed << endl;
  loc = 0;
  for (int j = 0; j < N / 2; j++) {
    ctr[0] = j;
    memcpy(&counter, &ctr, sizeof(ctr));
    speck.ProcessData(cipher, counter, 16);
    memcpy(&result, &cipher, sizeof(result));
    for (int i = 0; i < 2; i++) {
      ofile << loc << ", " << result[i] << std::endl;
      loc++;
    }
  }
  ofile.close();

  return 0;
}
