#include "jsf.hpp"
#include <fstream>
#include <cinttypes>
#include <iostream>
#include <string>
#include "../splitmix64/splitmix64.h"

using namespace std;

#define N 1000

int main() {
  uint64_t state, seed_val, seed = 0;
  state = seed;
  seed_val = splitmix64_next(&state);
  cout << (uint32_t)seed_val << std::endl;
  auto gen32 = jsf32((uint32_t)seed_val);
  std::ofstream ofile;
  ofile.open ("jsf32-testset-1.csv");
  ofile << "seed, " << seed <<endl;
  for (int i=0; i<N;i++){
      ofile << i << ", " <<  gen32() << std::endl;
  }
  ofile.close();

  seed = 0xDEADBEAF;
  state = seed;
  seed_val = splitmix64_next(&state);
  cout << (uint32_t)seed_val << std::endl;
  gen32 = jsf32((uint32_t)seed_val);
  ofile.open ("jsf32-testset-2.csv");
  ofile << "seed, " << seed <<endl;
  for (int i=0; i<N;i++){
      ofile << i << ", " <<  gen32() << std::endl;
  }
  ofile.close();

  seed = 0;
  state = seed;
  seed_val = splitmix64_next(&state);
  cout << seed_val << std::endl;
  auto gen64 = jsf64(seed_val);
  ofile.open ("jsf64-testset-1.csv");
  ofile << "seed, " << seed <<endl;
  for (int i=0; i<N;i++){
      ofile << i << ", " << gen64() << std::endl;
  }
  ofile.close();

  seed = 0xDEADBEAF;
  state = seed;
  seed_val = splitmix64_next(&state);
  cout << seed_val << std::endl;
  gen64 = jsf64(seed_val);
  ofile.open ("jsf64-testset-2.csv");
  ofile << "seed, " << seed <<endl;
  for (int i=0; i<N;i++){
      ofile << i << ", " << gen64() << std::endl;
  }
  ofile.close();
}
