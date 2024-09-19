// gcc -o2 pcg64-xsl_rr-test-data-gen.cpp -i pcg64_xslrr

#include <stdint.h>

#include <fstream>
#include <iostream>

#include "pcg_random.hpp"

using namespace std;

#define U128BIT_CONSTANT(high, low) (((__uint128_t)(high) << 64) + low)
#define N 1000

int main() {
  // First two values produced by SeedSequence(0)
  std::ofstream ofile, cm_ofile;
  uint64_t last = 0, cm_last = 0;
  __uint128_t seed =
      U128BIT_CONSTANT(15793235383387715774ULL, 12390638538380655177ULL);
  __uint128_t inc =
      U128BIT_CONSTANT(2361836109651742017ULL, 3188717715514472916ULL);
  pcg_engines::setseq_xsl_rr_128_64 rng(seed, inc);
  pcg_engines::cm_setseq_xsl_rr_128_64 cm_rng(seed, inc);
  cout << rng << "\n\n";
  cout << cm_rng << "\n\n";

  ofile.open("pcg64-xsl_rr-testset-1.csv");
  cm_ofile.open("pcg64-cm-xsl_rr-testset-1.csv");
  ofile << "seed, 0x" << 0 << endl;
  // cm_ofile << "seed, 0x" << 0 << endl;
  for (int i = 0; i < N; i++) {
    last = rng();
    cm_last = cm_rng();
    ofile << i << ", 0x" << std::hex << last << std::endl;
    cm_ofile << i << ", 0x" << std::hex << cm_last << std::endl;
  };
  ofile.close();
  cm_ofile.close();
  cout << std::dec << N - 1 << ", 0x" << std::hex << last << "\n";
  cout << std::dec << N - 1 << ", 0x" << std::hex << cm_last << "\n";

  // First two values produced by SeedSequence(0xDEADBEAF)
  seed = U128BIT_CONSTANT(5778446405158232650ULL, 4639759349701729399ULL);
  inc = U128BIT_CONSTANT(13222832537653397986ULL, 2330059127936092250ULL);

  rng = pcg_engines::setseq_xsl_rr_128_64(seed, inc);
  cm_rng = pcg_engines::cm_setseq_xsl_rr_128_64(seed, inc);

  ofile.open("pcg64-xsl_rr-testset-2.csv");
  cm_ofile.open("pcg64-cm-xsl_rr-testset-2.csv");
  ofile << "seed, 0x" << 0xDEADBEAF << endl;
  cm_ofile << "seed, 0x" << 0xDEADBEAF << endl;
  for (int i = 0; i < N; i++) {
    last = rng();
    cm_last = cm_rng();
    ofile << i << ", 0x" << std::hex << last << std::endl;
    cm_ofile << i << ", 0x" << std::hex << cm_last << std::endl;
  };
  ofile.close();
  cm_ofile.close();
  cout << std::dec << N - 1 << ", 0x" << std::hex << last << "\n";
  cout << std::dec << N - 1 << ", 0x" << std::hex << cm_last << "\n";
}