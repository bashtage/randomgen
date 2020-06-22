/*

Extact PractRand-pre0.95 and then compile using

g++ -std=c++14 efiix64-test-gen.cpp src/*.cpp src/RNGs/*.cpp
src/RNGs/other/*.cpp -O3 -Iinclude -pthread -o efiix64-test-gen

then

./efiix64-test-gen
*/

#include <stdint.h>

#include <fstream>
#include <iostream>

#include "include/PractRand.h"
#include "include/PractRand/RNGs/efiix64x48.h"
#include <iostream>

#define N 1000

int main() {
  std::ofstream ofile1, ofile2;
  uint64_t last;

  PractRand::RNGs::Polymorphic::efiix64x48 rng =
      PractRand::RNGs::Polymorphic::efiix64x48(0ULL);
  /* First 4 values from SeedSequence(0)*/
  rng.seed(15793235383387715774ULL, 12390638538380655177ULL,
           2361836109651742017ULL, 3188717715514472916ULL);
  ofile1.open("efiix64-testset-1.csv");
  ofile1 << "seed, 0x" << 0 << std::endl;
  for (int i = 0; i < N; i++) {
    last = rng.raw64();
    ofile1 << i << ", 0x" << std::hex << last << std::endl;
  };
  ofile1.close();
  std::cout << std::dec << N - 1 << ", 0x" << std::hex << last << "\n";

  /* First 4 values from SeedSequence(0xDEADBEEF)*/
  rng.seed(10671498545779160169ULL, 17039977206943430958ULL,
           8098813118336512226ULL, 451580776527170015ULL);
  ofile2.open("efiix64-testset-2.csv");
  ofile2 << "seed, 0x" << std::hex << 0xDEADBEEF << std::endl;
  for (int i = 0; i < N; i++) {
    last = rng.raw64();
    ofile2 << i << ", 0x" << std::hex << last << std::endl;
  };
  ofile2.close();
  std::cout << std::dec << N - 1 << ", 0x" << std::hex << last << "\n";
}
