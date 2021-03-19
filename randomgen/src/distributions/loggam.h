#ifndef _RANDOMDGEN__LOGGAM_H_
#define _RANDOMDGEN__LOGGAM_H_

/*
 * log-gamma function to support some of these distributions. The
 * algorithm comes from SPECFUN by Shanjie Zhang and Jianming Jin and their
 * book "Computation of Special Functions", 1996, John Wiley & Sons, Inc.
 */
static double loggam(double x) {
  double x0, x2, lg2pi, gl, gl0;
  RAND_INT_TYPE k, n;

  static double a[10] = {8.333333333333333e-02, -2.777777777777778e-03,
                         7.936507936507937e-04, -5.952380952380952e-04,
                         8.417508417508418e-04, -1.917526917526918e-03,
                         6.410256410256410e-03, -2.955065359477124e-02,
                         1.796443723688307e-01, -1.39243221690590e+00};

  if ((x == 1.0) || (x == 2.0)) {
    return 0.0;
  } else if (x < 7.0) {
    n = (RAND_INT_TYPE)(7 - x);
  } else {
    n = 0;
  }
  x0 = x + n;
  x2 = (1.0 / x0) * (1.0 / x0);
  /* log(2 * M_PI) */
  lg2pi = 1.8378770664093453e+00;
  gl0 = a[9];
  for (k = 8; k >= 0; k--) {
    gl0 *= x2;
    gl0 += a[k];
  }
  gl = gl0 / x0 + 0.5 * lg2pi + (x0 - 0.5) * log(x0) - x0;
  if (x < 7.0) {
    for (k = 1; k <= n; k++) {
      gl -= log(x0 - 1.0);
      x0 -= 1.0;
    }
  }
  return gl;
}

#endif
