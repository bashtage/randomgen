#include "tyche.orig.h"
#include <fstream>
#include <cinttypes>
#include <iostream>
#include <string>

using namespace std;

#define N 1000

int main() {
  clock_t t1, t2;
  uint64_t store[N];
  uint64_t seedval = 15793235383387715774ULL;
  uint32_t idx = 2968811710UL;
  auto tyche = openrand::Tyche(seedval, idx);
  std::ofstream ofile;
  ofile.open ("tyche-testset-1.csv");
  ofile << "seed, " << 0 <<endl;
  for (int i=0; i<N;i++){
      store[i] = ((uint64_t)tyche.draw())<<32 | tyche.draw();
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  cout << std::endl;

  seedval = 5778446405158232650ULL;
  idx =  3575046730UL;
  tyche = openrand::Tyche(seedval, idx);
  ofile.open ("tyche-testset-2.csv");
  ofile << "seed, " << 0xDEADBEAF <<endl;
  for (int i=0; i<N;i++){
      store[i] = ((uint64_t)tyche.draw())<<32 | tyche.draw() ;
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  cout << std::endl;

  return 0;
}
