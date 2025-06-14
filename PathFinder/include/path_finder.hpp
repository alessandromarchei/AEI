#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

class Lane
{
public:
    int id;
    // std::array<double, 3> coeff;
    // double cte;       // meters
    // double yaw_error; // radians
    // double curvature; // meters-1
    std::vector<cv::Point2f> BevPoints; // meters
    std::vector<std::array<double, 2>> GtPoints;  // meters
    Lane(int id,
         std::vector<std::array<double, 2>> GtPoints,
         std::vector<cv::Point2f> BevPoints) : id(id), GtPoints(GtPoints), BevPoints(BevPoints) {}
};

void drawLanes(const std::vector<Lane> &lanes);
std::vector<Lane> loadLanesFromYaml(const std::string &filename);


cv::Mat loadHFromYaml(const std::string &filename);
void cameraView(const std::vector<std::vector<cv::Point2f>> &lanes2d);

// cv::Mat drawLanePoints(const std::vector<std::vector<cv::Point2f>> &lanes2d);
// cv::Mat drawBEV(const std::vector<std::vector<cv::Point2f>> &lanesBEV);
// void fitQuadraticPolynomial(const cv::Point2f &points, std::vector<double> &coefficients);

#endif // PATH_FINDER_HPP