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
  pcg_engines::setseq_dxsm_128_64 rng(seed, 0U);
  pcg_engines::cm_setseq_dxsm_128_64 cm_rng(seed, 0U);
  cout << rng << "\n";
  cout << cm_rng << "\n";

  ofile.open("pcg64-dxsm-testset-1.csv");
  cm_ofile.open("pcg64-cm-dxsm-testset-1.csv");
  ofile << "seed, 0x" << 0 << endl;
  cm_ofile << "seed, 0x" << 0 << endl;
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

  // First two values produced by SeedSequence(0xDEADBEEF)
  seed = U128BIT_CONSTANT(10671498545779160169ULL, 17039977206943430958ULL);
  rng = pcg_engines::setseq_dxsm_128_64(seed, 0U);
  cm_rng = pcg_engines::cm_setseq_dxsm_128_64(seed, 0U);

  ofile.open("pcg64-dxsm-testset-2.csv");
  cm_ofile.open("pcg64-cm-dxsm-testset-2.csv");
  ofile << "seed, 0x" << 0xDEADBEEF << endl;
  cm_ofile << "seed, 0x" << 0xDEADBEEF << endl;
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