#ifndef PATH_FINDER_HPP
#define PATH_FINDER_HPP

#include <opencv2/opencv.hpp>
#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <yaml-cpp/yaml.h>
#include <Eigen/Dense>

struct LanePts
{
    int id;
    std::vector<cv::Point2f> BevPoints; // meters
    std::vector<cv::Point2f> GtPoints;  // meters
    LanePts(int id,
            std::vector<cv::Point2f> GtPoints,
            std::vector<cv::Point2f> BevPoints);
};

struct fittedCurve
{
    std::array<double, 3> coeff; // Coefficients for the quadratic polynomial
    double cte;       // Cross-track error in meters
    double yaw_error; // Yaw error in radians
    double curvature; // Curvature in meters^-1
    fittedCurve(const std::array<double, 3> &coeff);
};

struct roadLane
{
    int id;
    fittedCurve egoLeftLane;
    fittedCurve egoRightLane;
    fittedCurve egoPath;
};

void drawLanes(const std::vector<LanePts> &lanes,
               const std::vector<fittedCurve> &egoLanes,
               const fittedCurve &egoPath);
std::vector<LanePts> loadLanesFromYaml(const std::string &filename);
std::array<double, 2> generatePixelNoise(double max_noise);
std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points);
cv::Mat loadHFromYaml(const std::string &filename);
void cameraView(const std::vector<std::vector<cv::Point2f>> &lanes2d);

#endif // PATH_FINDER_HPP