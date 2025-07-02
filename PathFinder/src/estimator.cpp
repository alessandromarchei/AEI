#include "estimator.hpp"

void Estimator::initialize(const std::vector<double> &init_state, const std::vector<double> &init_var)
{
    dim = init_state.size();
    state = init_state;
    variance = init_var;
}

void Estimator::predict(std::vector<double> process_var)
{
    if (process_var.size() != dim)
    {
        throw std::runtime_error("Process variance size does not match state dimension.");
    }
    for (size_t i = 0; i < dim; ++i)
    {
        variance[i] += process_var[i];
    }
}

void Estimator::update(const std::vector<double> &measurement, const std::vector<double> &measurement_var)
{
    if (measurement.size() != dim || measurement_var.size() != dim)
    {
        std::cerr << "Measurement size: " << measurement.size() << ", Measurement variance size: " << measurement_var.size() << ", State dimension: " << dim << std::endl;
        throw std::runtime_error("Measurement or measurement variance size does not match state dimension.");
    }

    for (size_t i = 0; i < dim; ++i)
    {
        double prior_mean = state[i];
        double prior_var = variance[i];
        double meas = measurement[i];
        double meas_var = measurement_var[i];

        // Kalman gain
        double K = prior_var / (prior_var + meas_var);

        // Posterior estimate
        state[i] = prior_mean + K * (meas - prior_mean);

        // Posterior variance
        variance[i] = (1.0 - K) * prior_var;
    }
}

const std::vector<double> &Estimator::getState() const { return state; }

const std::vector<double> &Estimator::getVariance() const { return variance; }