// @HEADER
// *******************************************************************************
//                                OpenRAND                                       *
//   A Performance Portable, Reproducible Random Number Generation Library       *
//                                                                               *
// Copyright (c) 2023, Michigan State University                                 *
//                                                                               *
// Permission is hereby granted, free of charge, to any person obtaining a copy  *
// of this software and associated documentation files (the "Software"), to deal *
// in the Software without restriction, including without limitation the rights  *
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell     *
// copies of the Software, and to permit persons to whom the Software is         *
// furnished to do so, subject to the following conditions:                      *
//                                                                               *
// The above copyright notice and this permission notice shall be included in    *
// all copies or substantial portions of the Software.                           *
//                                                                               *
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR    *
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,      *
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE   *
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER        *
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, *
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE *
// SOFTWARE.                                                                     *
//********************************************************************************
// @HEADER

#ifndef OPENRAND_BASE_STATE_H_
#define OPENRAND_BASE_STATE_H_

#include "util.h"

#include <cstdint>
#include <limits>
#include <type_traits>

namespace openrand {

/**
 * @brief Base class for random number generators.
 *
 * This class utilizes the CRTP pattern to inject some common functionalities
 * into random number generators.
 *
 * @tparam RNG Random number generator class. It must contain two public
 * methods, a constructor and a templated draw() function that returns the next
 * 32 or 64 bit random unsigned int from its stream.
 */
template <typename RNG>
class BaseRNG {
 public:
  using result_type = uint32_t;

  static constexpr result_type min() {
    return 0u;
  }
  static constexpr result_type max() {
    return ~((result_type)0);
  }

  /**
   * @brief Generates a 32 bit unsigned integer from a uniform distribution.
   *
   * This function is needed to conform to C++ engine interface
   *
   * @return uint32_t random number from a uniform distribution
   */
  OPENRAND_DEVICE result_type operator()() {
    return gen().template draw<uint32_t>();
  }

  /**
   * @brief Generates a random number from a uniform distribution between 0
   * and 1.
   *
   * @note Some generators may expose a more efficient version of this function
   * that returns multiple values at once.
   *
   * @tparam T Data type to be returned. Can be 32 or 64 bit integer, float or
   * double.
   * @return T random number from a uniform distribution between 0 and 1
   */
  template <typename T = float>
  OPENRAND_DEVICE T rand() {
    if constexpr (sizeof(T) <= 4) {
      const uint32_t x = gen().template draw<uint32_t>();
      if constexpr (std::is_integral_v<T>)
        return static_cast<T>(x);
      else
        return u01<float, uint32_t>(x);
    } else {
      const uint64_t x = gen().template draw<uint64_t>();
      if constexpr (std::is_integral_v<T>)
        return static_cast<T>(x);
      else
        return u01<double, uint64_t>(x);
    }
  }

  /**
   * @brief Generates a number from a uniform distribution between a and b.
   *
   * @Note For integer types, consider using @ref range for greater control.
   *
   * @tparam T Data type to be returned. Can be float or double.
   * double.
   * @param low lower bound of the uniform distribution
   * @param high upper bound of the uniform distribution
   * @return T random number from a uniform distribution between a and b
   */
  template <typename T = float>
  OPENRAND_DEVICE T uniform(const T low, const T high) {
    // TODO: Allow 64 bit integers
    static_assert(!(std::is_integral_v<T> && sizeof(T) > sizeof(int32_t)),
                  "64 bit int not yet supported");

    T r = high - low;

    if constexpr (std::is_floating_point_v<T>) {
      return low + r * rand<T>();
    } else if constexpr (std::is_integral_v<T>) {
      return low + range<true, T>(r);
    }
  }

  /*
   * @brief Fills an array with random numbers from a uniform distribution [0, 1)
   *
   * @Note This array is filled serially. `N` ideally should not be large.
   */
  template <typename T = float>
  OPENRAND_DEVICE void fill_random(T *array, const int N) {
    for (int i = 0; i < N; i++) array[i] = rand<T>();
  }

  /**
   * @brief Generates a random number from a normal distribution with mean 0 and
   * std 1.
   *
   * This function implements box-muller method. This method avoids branching,
   * and therefore more efficient on GPU. see:
   * https://developer.nvidia.com/gpugems/gpugems3/part-vi-gpu-computing/chapter-37-efficient-random-number-generation-and-application
   *
   * @tparam T floating point type to be returned
   * @return T random number from a normal distribution with mean 0 and std 1
   */
  template <typename T = float>
  OPENRAND_DEVICE T randn() {
    static_assert(std::is_floating_point_v<T>);
    constexpr T M_PI2 = 2 * static_cast<T>(M_PI);

    T u = rand<T>();
    T v = rand<T>();
    T r = openrand::sqrt(T(-2.0) * openrand::log(u));
    T theta = v * M_PI2;
    return r * openrand::cos(theta);
  }

  /**
   * @brief More efficient version of @ref randn, returns two values at once.
   *
   * @tparam T floating point type to be returned
   * @return T random number from a normal distribution with mean 0 and std 1
   */
  template <typename T = float>
  OPENRAND_DEVICE vec2<T> randn2() {
    // Implements box-muller method
    static_assert(std::is_floating_point_v<T>);
    constexpr T M_PI2 = 2 * static_cast<T>(M_PI);

    T u = rand<T>();
    T v = rand<T>();
    T r = sqrt(T(-2.0) * log(u));
    T theta = v * M_PI2;
    return {r * cos(theta), r * sin(theta)};
  }

  /**
   * @brief Generates a random number from a normal distribution with mean and
   * std.
   *
   * @tparam T floating point type to be returned
   * @param mean mean of the normal distribution
   * @param std_dev standard deviation of the normal distribution
   * @return T random number from a normal distribution with mean and std
   */
  template <typename T = float>
  OPENRAND_DEVICE T randn(const T mean, const T std_dev) {
    return mean + randn<T>() * std_dev;
  }

  /**
   * @brief Generates a random integer of certain range
   *
   * This uses the method described in [1] to generate a random integer
   * of range [0..N)
   *
   *
   * @attention if using non-biased version, please make sure that N is not
   * too large [2]
   *
   * @tparam biased if true, the faster, but slightly biased variant is used.
   * @tparam T integer type (<=32 bit) to be returned
   *
   * @param N A random integral of range [0..N) will be returned
   * @return T random number from a uniform distribution between 0 and N
   *
   *
   * @note [1] https://lemire.me/blog/2016/06/30/fast-random-shuffling/
   * @note [2] If N=2^b (b<32), pr(taking the branch) = p = 1/2^(32-b). For N=2^24,
   * this value is 1/2^8 = .4%, quite negligible. But GPU complicates this simple math.
   * Assuming a warp size of 32, the probability of a thread taking the branch becomes
   * 1 - (1-p)^32. For N=2^24, that value is 11.8%. For N=2^20, it's 0.8%.
   */
  template <bool biased = true, typename T = int>
  OPENRAND_DEVICE T range(const T N) {
    // static_assert(std::is_integral_v<T> && sizeof(T) <= sizeof(int32_t),
    //               "64 bit int not yet supported");

    uint32_t x = gen().template draw<uint32_t>();
    uint64_t res = static_cast<uint64_t>(x) * static_cast<uint64_t>(N);

    if constexpr (biased) {
      return static_cast<T>(res >> 32);
    } else {
      uint32_t leftover = static_cast<uint32_t>(res);
      if (leftover < N) {
        uint32_t threshold = -N % N;
        while (leftover < threshold) {
          x = gen().template draw<uint32_t>();
          res = static_cast<uint64_t>(x) * static_cast<uint64_t>(N);
          leftover = static_cast<uint32_t>(res);
        }
      }
      return static_cast<T>(res);
    }
  }

  /**
   * @brief Generates a random number from a gamma distribution with shape alpha
   * and scale b.
   *
   * Adapted from the following implementation:
   * https://www.hongliangjie.com/2012/12/19/how-to-generate-gamma-random-variables/
   *
   * @tparam T floating point type to be returned
   * @param alpha shape parameter of the gamma distribution
   * @param b scale parameter of the gamma distribution
   * @return T random number from a gamma distribution with shape alpha and
   * scale b
   */
  template <typename T = float>
  OPENRAND_DEVICE inline T gamma(T alpha, T b) {
    T d = alpha - T((1. / 3.));
    T c = T(1.) / sqrt(9.f * d);
    T v, x;
    while (true) {
      do {
        x = randn<T>();
        v = T(1.0) + c * x;
      } while (v <= T(0.));
      v = v * v * v;
      T u = rand<T>();

      const T x2 = x * x;
      if (u < 1.0f - 0.0331f * x2 * x2) return (d * v * b);

      if (log(u) < 0.5f * x2 + d * (1.0f - v + log(v))) return (d * v * b);
    }
  }

  /**
   * @brief Returns a new generator with the internal state forwarded by a given number
   *
   * The new generator's first output will be the n+1th output of the current
   * generator and so on. This is O(1) operation.
   *
   * @param n Number of steps to move the state forward
   * @return RNG A new RNG object with the state moved forward by n steps
   *
   */
  template <typename T = RNG>
  typename std::enable_if_t<has_counter<T>::value, RNG> forward_state(
      int n) const {
    RNG rng = *static_cast<const RNG *>(this);  // copy
    rng._ctr += n;
    return rng;
  }

 protected:
  /*
   * @brief Converts a random number integer to a floating point number between [0., 1.)
   */
  template <typename Ftype, typename Utype>
  inline OPENRAND_DEVICE Ftype u01(const Utype in) const {
    constexpr Ftype factor =
        Ftype(1.) / (Ftype(~static_cast<Utype>(0)) + Ftype(1.));
    constexpr Ftype halffactor = Ftype(0.5) * factor;
    return static_cast<Ftype>(in) * factor + halffactor;
  }

 private:
  /**
   * @brief Returns a reference to the random number generator.
   */
  //OPENRAND_DEVICE __inline__ RNG &gen() {
 //   return *static_cast<RNG *>(this);
 // }

};  // class BaseRNG

}  // namespace openrand

#endif  // OPENRAND_BASE_STATE_H_
