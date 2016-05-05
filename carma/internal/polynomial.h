#ifndef _CARMA_INTERNAL_POLYNOMIAL_H_
#define _CARMA_INTERNAL_POLYNOMIAL_H_

#include <cmath>
#include <vector>
#include <Eigen/Core>

namespace carma {
namespace internal {

template <typename T>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> roots_from_params (
    const Eigen::Matrix<T, Eigen::Dynamic, 1>& params
) {
    typedef Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> ctype;

    size_t n = params.rows();
    std::complex<T> b, c, arg;
    ctype roots(n);

    if (n == 0) return roots;
    if (n % 2 == 1) roots(n - 1) = -exp(params(n - 1));
    for (size_t i = 0; i < n-1; i += 2) {
        b = exp(params(i+1));
        c = exp(params(i));
        arg = sqrt(b*b - T(4.0)*c);
        roots(i)   = T(0.5) * (-b + arg);
        roots(i+1) = T(0.5) * (-b - arg);
    }
    return roots;
}


template <typename T>
Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> poly_from_roots (
    const Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1>& roots
) {
    typedef Eigen::Matrix<std::complex<T>, Eigen::Dynamic, 1> ctype;
    size_t n = roots.rows() + 1;

    if (n == 1) return ctype::Ones(1);

    ctype poly = ctype::Zero(n);
    poly(0) = -roots(0);
    poly(1) = T(1.0);
    for (size_t i = 1; i < n-1; ++i) {
        for (size_t j = n-1; j >= 1; --j)
            poly(j) = poly(j - 1) - roots(i) * poly(j);
        poly(0) *= -roots(i);
    }
    return poly;
}

}; // namespace internal
}; // namespace carma

#endif // _CARMA_INTERNAL_POLYNOMIAL_H_
