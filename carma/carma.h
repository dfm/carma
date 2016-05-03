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


struct Prediction {
    double expectation;
    double variance;
};


struct State {
    double time;
    Eigen::VectorXcd x;
    Eigen::MatrixXcd P;
};


//
// This class evaluates the log likelihood of a CARMA model using a Kalman filter.
//
class CARMASolver {
public:

    CARMASolver (std::complex<double> sigma,
                 Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> arroots,
                 Eigen::Matrix<std::complex<double>, Eigen::Dynamic, 1> maroots)
    : sigma_(sigma), p_(arroots.rows()), q_(maroots.rows()),
      arroots_(arroots), maroots_(maroots),
      b_(Eigen::MatrixXcd::Zero(1, p_)), lambda_base_(p_)
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
        alpha_ = poly_from_roots(arroots);
        beta_ = poly_from_roots(maroots);
        beta_ /= beta_(0);
        b_.head(q_ + 1) = beta_;
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

    void reset (double t) {
        // Step 2 from Kelly et al.
        state_.time = t;
        state_.x = Eigen::VectorXcd::Zero(p_);
        state_.P = V_;
    };

    Prediction predict (double yerr) const {
        // Steps 3 and 9 from Kelly et al.
        Prediction pred;
        std::complex<double> tmp = b_ * state_.x;
        pred.expectation = tmp.real();
        tmp = b_ * state_.P * b_.adjoint();
        pred.variance = yerr * yerr + tmp.real();

        // Check the variance value for instability.
        if (pred.variance < 0.0) throw _CARMA_SOLVER_UNSTABLE_;
        return pred;
    };

    void update_state (const Prediction& pred, double y) {
        // Steps 4-6 and 10-12 from Kelly et al.
        Eigen::VectorXcd K = state_.P * b_.adjoint() / pred.variance;
        state_.x += (y - pred.expectation) * K;
        state_.P -= pred.variance * K * K.adjoint();
    };

    void advance_time (double dt) {
        // Steps 7 and 8 from Kelly et al.
        Eigen::VectorXcd lam = pow(lambda_base_, dt).matrix();
        state_.time += dt;
        for (unsigned i = 0; i < p_; ++i) state_.x(i) *= lam(i);
        state_.P = lam.asDiagonal() * (state_.P - V_) * lam.conjugate().asDiagonal() + V_;
    };

    double log_likelihood (Eigen::VectorXd t, Eigen::VectorXd y, Eigen::VectorXd yerr) {
        unsigned n = t.rows();
        double r, ll = n * log(2.0 * M_PI);
        Prediction pred;

        reset(t(0));
        for (unsigned i = 0; i < n; ++i) {
            // Integrate the Kalman filter.
            pred = predict(yerr(i));
            update_state(pred, y(i));
            if (i < n - 1) advance_time(t(i+1) - t(i));

            // Update the likelihood evaluation.
            r = y(i) - pred.expectation;
            ll += r * r / pred.variance + log(pred.variance);
        }

        return -0.5 * ll;
    };

    double psd (double f) const {
        std::complex<double> w(0.0, 2.0 * M_PI * f), num = 0.0, denom = 0.0;
        for (unsigned i = 0; i < q_+1; ++i)
            num += beta_(i) * pow(w, i);
        for (unsigned i = 0; i < p_+1; ++i)
            denom += alpha_(i) * pow(w, i);
        return std::norm(sigma_) * std::norm(num) / std::norm(denom);
    };

    double covariance (double tau) const {
        std::complex<double> n1, n2, norm, value = 0.0;

        for (unsigned k = 0; k < p_; ++k) {
            n1 = 0.0;
            n2 = 0.0;
            for (unsigned l = 0; l < q_+1; ++l) {
                n1 += beta_(l) * pow(arroots_(k), l);
                n2 += beta_(l) * pow(-arroots_(k), l);
            }
            norm = n1 * n2 / arroots_(k).real();
            for (unsigned l = 0; l < p_; ++l) {
                if (l != k)
                    norm /= (arroots_(l) - arroots_(k)) * (std::conj(arroots_(l)) + arroots_(k));
            }
            value += norm * exp(arroots_(k) * tau);
        }

        return -0.5 * std::norm(sigma_) * value.real();
    };


private:

    std::complex<double> sigma_;
    unsigned p_, q_;
    Eigen::VectorXcd arroots_, maroots_;
    Eigen::VectorXcd alpha_, beta_;
    Eigen::RowVectorXcd b_;

    Eigen::MatrixXcd V_;
    Eigen::ArrayXcd lambda_base_;
    State state_;

}; // class CARMASolver


//
// C-type wrappers around the CARMASolver functions.
//
double log_likelihood (double sigma,
                       unsigned p, std::complex<double>* ar,
                       unsigned q, std::complex<double>* ma,
                       unsigned n, double* t, double* y, double* yerr)
{
    Eigen::Map<Eigen::VectorXcd> arroots(ar, p), maroots(ma, q);
    CARMASolver solver(sigma, arroots, maroots);

    Eigen::Map<Eigen::VectorXd> tvec(t, n),
                                yvec(y, n),
                                yerrvec(yerr, n);

    return solver.log_likelihood(tvec, yvec, yerrvec);
};

void psd (double sigma,
          unsigned p, std::complex<double>* ar,
          unsigned q, std::complex<double>* ma,
          unsigned n, double* f, double* out)
{
    Eigen::Map<Eigen::VectorXcd> arroots(ar, p), maroots(ma, q);
    CARMASolver solver(sigma, arroots, maroots);
    for (unsigned i = 0; i < n; ++i)
        out[i] = solver.psd(f[i]);
};

void covariance (double sigma,
                 unsigned p, std::complex<double>* ar,
                 unsigned q, std::complex<double>* ma,
                 unsigned n, double* tau, double* out)
{
    Eigen::Map<Eigen::VectorXcd> arroots(ar, p), maroots(ma, q);
    CARMASolver solver(sigma, arroots, maroots);
    for (unsigned i = 0; i < n; ++i)
        out[i] = solver.covariance(tau[i]);
};


}; // namespace carma

#endif // _CARMA_CARMA_H_
