// cl /O2 tyche-gen-test-data.cpp -D__USE_ORIGINAL_TYCHE__  /EHsc
// cl /O2 tyche-gen-test-data.cpp -D__USE_OPENRAND_TYCHE__  /EHsc

#include "tyche.orig.h"
#include <fstream>
#include <cinttypes>
#include <iostream>
#include <string>

using namespace std;

#define N 1000

#if !defined(__USE_OPENRAND_TYCHE__) && !defined(__USE_ORIGINAL_TYCHE__)
error("No implementation selected for Tyche");
#endif

#ifdef __USE_OPENRAND_TYCHE__
#define OUTPUT_FILE "tyche-openrand"
#pragma message("Using openrand Tyche")
#else
#define OUTPUT_FILE "tyche"
#pragma message("Using original Tyche")
#endif

int main() {
  clock_t t1, t2;
  uint64_t store[N];
  uint64_t seedval = 15793235383387715774ULL;
  uint32_t idx = 745650761UL;
  auto tyche = openrand::Tyche(seedval, idx);
  std::ofstream ofile;
  ofile.open (OUTPUT_FILE "-testset-1.csv");
  ofile << "seed, " << 0 <<endl;
  for (int i=0; i<N;i++){
      store[i] = ((uint64_t)tyche.draw())<<32 | tyche.draw();
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  cout << std::endl;

  seedval = 5778446405158232650ULL;
  idx =  3393510519UL;
  tyche = openrand::Tyche(seedval, idx);
  ofile.open (OUTPUT_FILE "-testset-2.csv");
  ofile << "seed, " << 0xDEADBEAF <<endl;
  for (int i=0; i<N;i++){
      store[i] = ((uint64_t)tyche.draw())<<32 | tyche.draw() ;
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  cout << std::endl;

  return 0;
}
