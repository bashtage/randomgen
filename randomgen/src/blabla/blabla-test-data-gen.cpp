//
// cl blabla-test-data-gen.cpp /O2 /arch:AVX2 /EHsc
// or no avx2
// cl blabla-test-data-gen.cpp /O2 /arch:SSE2 /EHsc
//
// g++ blabla-test-data-gen.cpp -o blabla-test-data-gen -O2 -march=native
// or no avx2
// g++ blabla-test-data-gen.cpp -o blabla-test-data-gen -O2 -march=x86_64
#include "blabla-orig.h"
#include <fstream>
#include <cinttypes>
#include <iostream>
#include <string>
#include <iostream>
#include <ostream>
using namespace std;

#define N 1000

int main() {
  clock_t t1, t2;
  uint64_t store[N];
  uint64_t seedval[2] = {15793235383387715774ULL, 12390638538380655177ULL};
  uint64_t stream[2] = {2361836109651742017ULL,3188717715514472916ULL};
  auto blabla = BlaBlaPRNG::BlaBla<20>(seedval, stream);
  std::ofstream ofile;
  std::string filename = "blabla-testset-1";
  #if (defined(__AVX2__) && __AVX2__)
  filename += "-avx2";
  #endif
  filename += ".csv";
  ofile.open(filename);
  ofile << "seed, " << 0 <<endl;
  for (int i=0; i<N;i++){
      store[i] = blabla();
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  std::cout << "\n\nKeysetup: \n";
  for (int i=0; i<4; i++){
    std::cout << blabla.keysetup[i] << " ";
  }
  std::cout << std::endl;

  seedval[0] = 5778446405158232650ULL;
  seedval[1] = 4639759349701729399ULL;
  stream[0] =  13222832537653397986ULL;
  stream[1] = 2330059127936092250ULL;
  blabla = BlaBlaPRNG::BlaBla<20>(seedval, stream);
  filename = "blabla-testset-2";
  #if (defined(__AVX2__) && __AVX2__)
  filename += "-avx2";
  #endif
  filename += ".csv";
  ofile.open(filename);
  ofile << "seed, " << 0xDEADBEAF <<endl;
  for (int i=0; i<N;i++){
      store[i] = blabla();
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  std::cout << "\n\nKeysetup: \n";
  for (int i=0; i<4; i++){
    std::cout << blabla.keysetup[i] << " ";
  }
  std::cout << std::endl;

  return 0;
}
