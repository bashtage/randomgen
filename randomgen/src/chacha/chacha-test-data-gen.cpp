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
  auto chacha = ChaCha<20>(0);
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


  chacha = ChaCha<20>(0xDEADBEEF);
  ofile.open ("chacha-testset-2.csv");
  ofile << "seed, " << 0xDEADBEEF <<endl;
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
