#include "chacha.orig.h"
#include <fstream>
#include <cinttypes>
#include <iostream>
#include <string>

using namespace std;

#define N 1000

int main() {
  clock_t t1, t2;
  uint64_t store[N];
  uint64_t seedval[2] = {15793235383387715774ULL, 12390638538380655177ULL};
  uint64_t stream[2] = {2361836109651742017ULL,3188717715514472916ULL};
  auto chacha = ChaCha<20>(seedval, stream);
  std::ofstream ofile;
  ofile.open ("chacha-testset-1.csv");
  ofile << "seed, " << 0 <<endl;
  for (int i=0; i<N;i++){
      store[i] = chacha() | ((uint64_t)chacha())<<32;
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  for (int i=0; i<8; i++){
    cout << chacha.keysetup[i] << " ";
  }
  cout << std::endl;

  seedval[0] = 5778446405158232650;
  seedval[1] =   4639759349701729399;
  stream[0] =  13222832537653397986;
  stream[1] = 2330059127936092250;
  chacha = ChaCha<20>(seedval, stream);
  ofile.open ("chacha-testset-2.csv");
  ofile << "seed, " << 0xDEADBEAF <<endl;
  for (int i=0; i<N;i++){
      store[i] = chacha() | ((uint64_t)chacha())<<32;
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  for (int i=0; i<8; i++){
    cout << chacha.keysetup[i] << " ";
  }
  cout << std::endl;

  return 0;
}
