#include <iostream>
#include <complex>
#include "carma.h"

int main ()
{
    double areal[] = {-0.5, -0.5, -1.0, -1.0},
           aimag[] = {0.01, -0.01, 0.1, -0.1},
           breal[] = {-0.5, -0.5},
           bimag[] = {0.1, -0.1},
           x[] = {0.5, 0.6, 10.0},
           y[] = {0.05, 0.06, 1.0},
           yerr[] = {0.1, 0.1, 0.1};

    std::cout << carma::compute_log_likelihood(0.1, 4, areal, aimag,
        2, breal, bimag, 3, x, y, yerr) << std::endl;


    /* Eigen::VectorXcd a(4), b(2); */

    /* a(0) = std::complex<double>(-0.5, 0.01); */
    /* a(1) = std::complex<double>(-0.5, -0.01); */
    /* a(2) = std::complex<double>(-1.0, 0.1); */
    /* a(3) = std::complex<double>(-1.0, -0.1); */

    /* b(0) = std::complex<double>(-0.5, 0.1); */
    /* b(1) = std::complex<double>(-0.5, -0.1); */

    /* unsigned n = 3; */
    /* Eigen::VectorXd x(n), y(n), yerr = 0.1 * Eigen::VectorXd::Ones(n); */

    /* x(0) = 0.5; */
    /* x(1) = 0.6; */
    /* x(2) = 10.0; */
    /* y = 0.1 * x; */

    /* carma::CARMASolver solver (0.1, a, b); */
    /* std::cout << solver.log_likelihood(x, y, yerr) << std::endl; */

    return 0;
}
