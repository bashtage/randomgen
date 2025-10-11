#include "blabla-orig.h"
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
  auto blabla = BlaBlaPRNG::BlaBla<20>(seedval, stream);
  std::ofstream ofile;
  ofile.open ("blabla-testset-1.csv");
  ofile << "seed, " << 0 <<endl;
  for (int i=0; i<N;i++){
      store[i] = blabla();
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  cout << "\n\nKeysetup: \n";
  for (int i=0; i<4; i++){
    cout << blabla.keysetup[i] << " ";
  }
  cout << std::endl;

  seedval[0] = 5778446405158232650;
  seedval[1] =   4639759349701729399;
  stream[0] =  13222832537653397986;
  stream[1] = 2330059127936092250;
  blabla = BlaBlaPRNG::BlaBla<20>(seedval, stream);
  ofile.open ("blabla-testset-2.csv");
  ofile << "seed, " << 0xDEADBEAF <<endl;
  for (int i=0; i<N;i++){
      store[i] = blabla();
      ofile << i << ", " << store[i] << std::endl;
  };
  ofile.close();
  cout << "\n\nKeysetup: \n";
  for (int i=0; i<4; i++){
    cout << blabla.keysetup[i] << " ";
  }
  cout << std::endl;

  return 0;
}
