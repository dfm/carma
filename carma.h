#ifndef _CARMA_CARMA_H_
#define _CARMA_CARMA_H_

#include <cmath>
#include <complex>

#include <Eigen/Dense>

namespace carma {

#define _CARMA_SOLVER_UNSTABLE_ 1

//
// Get the polynomial representation from a list of roots.
//
Eigen::VectorXcd poly_from_roots (const Eigen::VectorXcd& roots) {
    unsigned n = roots.rows() + 1;
    if (n == 1) return Eigen::VectorXcd::Ones(1);
    Eigen::VectorXcd poly = Eigen::VectorXcd::Zero(n);
    poly(0) = -roots(0);
    poly(1) = 1.0;
    for (unsigned i = 1; i < n-1; ++i) {
        for (unsigned j = n-1; j >= 1; --j)
            poly(j) = poly(j - 1) - roots(i) * poly(j);
        poly(0) *= -roots(i);
    }
    return poly;
}

class CARMASolver {
public:

    CARMASolver (double sigma, Eigen::VectorXcd arroots, Eigen::VectorXcd maroots)
    : sigma_(sigma), arroots_(arroots), maroots_(maroots),
      p_(arroots.rows()), q_(maroots.rows()),
      b_(Eigen::MatrixXcd::Zero(1, p_)), x_(p_), lambda_base_(p_), P_(p_, p_),
      expect_y_(0.0), var_y_(-1.0)
    {
        // Pre-compute the base lambda vector.
        for (unsigned i = 0; i < p_; ++i)
            lambda_base_(i) = exp(arroots_(i));

        // Construct the rotation matrix for the diagonalized space.
        Eigen::MatrixXcd U(p_, p_);
        for (unsigned i = 0; i < p_; ++i)
            for (unsigned j = 0; j < p_; ++j)
                U(i, j) = pow(arroots_(j), i);

        // Compute the polynomial coefficients and rotate into the diagonalized space.
        Eigen::VectorXcd beta = poly_from_roots(maroots_);
        beta /= beta(0);
        b_.head(q_ + 1) = beta;
        b_ = b_ * U;

        // Compute V.
        Eigen::VectorXcd e = Eigen::VectorXcd::Zero(p_);
        e(p_ - 1) = sigma;

        // J = U \ e
        Eigen::FullPivLU<Eigen::MatrixXcd> lu(U);
        Eigen::VectorXcd J = lu.solve(e);

        // V_ij = -J_i J_j^* / (r_i + r_j^*)
        V_ = -J * J.adjoint();
        for (unsigned i = 0; i < p_; ++i)
            for (unsigned j = 0; j < p_; ++j)
                V_(i, j) /= arroots_(i) + std::conj(arroots_(j));
    };

    void reset () {
        // Step 2 from Kelly et al.
        for (unsigned i = 0; i < p_; ++i) {
            x_(i) = 0.0;
            for (unsigned j = 0; j < p_; ++j)
                P_(i, j) = V_(i, j);
        }
    };

    void predict (double yerr) {
        // Steps 3 and 9 from Kelly et al.
        std::complex<double> E = b_ * x_;
        expect_y_ = E.real();

        std::complex<double> V = b_ * P_ * b_.adjoint();
        var_y_ = yerr * yerr + V.real();
    };

    void update (double y) {
        // Steps 4-6 and 10-12 from Kelly et al.
        Eigen::VectorXcd K = P_ * b_.adjoint() / var_y_;
        x_ += (y - expect_y_) * K;
        P_ -= var_y_ * K * K.adjoint();
    };

    void advance (double dt) {
        // Steps 7 and 8 from Kelly et al.
        Eigen::MatrixXcd lam = pow(lambda_base_, dt).matrix();
        for (unsigned i = 0; i < p_; ++i) x_(i) *= lam(i);
        P_ = lam.asDiagonal() * (P_ - V_) * lam.conjugate().asDiagonal() + V_;
    };

    double log_likelihood (Eigen::VectorXd x, Eigen::VectorXd y, Eigen::VectorXd yerr) {
        unsigned n = x.rows();
        double r, ll = n * log(2.0 * M_PI);

        reset();
        for (unsigned i = 0; i < n; ++i) {
            predict(yerr(i));
            if (var_y_ < 0.0) throw _CARMA_SOLVER_UNSTABLE_;
            r = y(i) - expect_y_;
            ll += r * r / var_y_ + log(var_y_);

            update(y(i));
            if (i < n - 1) advance(x(i+1) - x(i));
        }

        return -0.5 * ll;
    };


private:

    unsigned p_, q_;
    double sigma_;
    Eigen::VectorXcd arroots_, maroots_;

    Eigen::Matrix<std::complex<double>, 1, Eigen::Dynamic> b_;
    Eigen::VectorXcd x_;
    Eigen::MatrixXcd V_, P_;
    Eigen::ArrayXcd lambda_base_;
    double expect_y_, var_y_;

}; // class CARMASolver

}; // namespace carma

#endif // _CARMA_KALMAN_H_
