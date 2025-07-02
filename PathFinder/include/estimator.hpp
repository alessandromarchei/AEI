#pragma once

#include <vector>
#include <iostream>

class Estimator
{
private:
    size_t dim;                   // dimensionality
    std::vector<double> state;    // mean estimate for each variable
    std::vector<double> variance; // variance for each variable (diagonal covariance)
public:
    void initialize(const std::vector<double> &init_state, const std::vector<double> &init_var);
    void predict(const std::vector<double> &delta, const std::vector<double> &process_var);
    void update(const std::vector<double> &measurement, const std::vector<double> &measurement_var);
    const std::vector<double> &getState() const;
    const std::vector<double> &getVariance() const;
};