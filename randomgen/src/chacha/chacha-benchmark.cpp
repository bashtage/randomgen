//
// g++ -benchmark.cpp -mssse3 -O2 -o -benchmark
// ./-benchmark
//
// cl /Ox /EHsc -benchmark.cpp
// ./-benchmark.exe
//

#include ".orig.h"
#include <time.h>
using namespace std;

#define N 1000000000

int main() {
  clock_t t1, t2;

  auto chacha = ChaCha<20>(0xDEADBEEF);
  uint64_t sum = 0;
  t1 = clock();
  for (int i = 0; i < N; i++) {
    sum += ((uint64_t)chacha()) << 32 | chacha();
  }
  t2 = clock();
  cout << sum << std::endl;
  double diff = (double)(t2) - (double)(t1);
  double seconds = diff / CLOCKS_PER_SEC;
  double num_per_second = (double)N / seconds;
  cout.precision(10);
  cout << num_per_second << " randoms per second\n";
  cout.precision(3);
  cout << 1000. * 1000000. / num_per_second << " ms to produce 1,000,000 draws \n";
}
