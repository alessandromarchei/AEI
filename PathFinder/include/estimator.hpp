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
    void predict(std::vector<double> process_var); // Add process noise to variance
    void update(const std::vector<double> &measurement, const std::vector<double> &measurement_var);
    const std::vector<double> &getState() const;
    const std::vector<double> &getVariance() const;
};