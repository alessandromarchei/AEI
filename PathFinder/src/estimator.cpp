#include "estimator.hpp"

void Estimator::initialize(const std::vector<double> &init_state, const std::vector<double> &init_var)
{
    dim = init_state.size();
    state = init_state;
    variance = init_var;
}

void Estimator::predict(const std::vector<double> &delta, const std::vector<double> &process_var)
{
    if (delta.size() != state.size() || process_var.size() != state.size())
        throw std::runtime_error("Size mismatch in predict step");

    for (size_t i = 0; i < state.size(); ++i)
    {
        state[i] += delta[i];          // Mean update: m0p = m0 + delta
        variance[i] += process_var[i]; // Variance update: v0p = v0 + Q
    }
}

void Estimator::update(const std::vector<double> &measurement,
                       const std::vector<double> &measurement_var)
{
    for (size_t i = 0; i < state.size(); ++i)
    {
        double v0p = variance[i];       // predicted variance
        double v1 = measurement_var[i]; // measurement variance
        double m0p = state[i];          // predicted mean
        double m1 = measurement[i];     // measurement

        // New variance: V2 = (V0p * V1) / (V0p + V1)
        double v2 = (v0p * v1) / (v0p + v1);
        // New mean: M2 = (M0p * V1 + M1 * V0p) / (V0p + V1)
        double m2 = (m0p * v1 + m1 * v0p) / (v0p + v1);

        // Store updated values
        state[i] = m2;
        variance[i] = v2;
    }
}

const std::vector<double> &Estimator::getState() const { return state; }

const std::vector<double> &Estimator::getVariance() const { return variance; }