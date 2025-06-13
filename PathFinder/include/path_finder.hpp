#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

void loadLanesFromYaml(const std::string &filename,
                       std::vector<std::vector<std::array<double, 2>>> &lanes2d,
                       std::vector<std::vector<std::array<double, 2>>> &lanesBEV);
void loadHFromYaml(const std::string &filename,
                   std::array<std::array<double, 4>, 3> &H);
cv::Mat drawLanePoints(const std::vector<std::vector<std::array<double, 2>>> &lanes2d);
cv::Mat drawBEV(const std::vector<std::vector<std::array<double, 2>>> &lanesBEV);
void fitQuadraticPolynomial(const std::vector<std::array<double, 2>> &points, std::vector<double> &coefficients);

#endif // PATH_FINDER_HPP