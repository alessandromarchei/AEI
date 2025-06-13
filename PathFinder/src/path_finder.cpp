#include "path_finder.hpp"

void loadLanesFromYaml(const std::string &filename,
                       std::vector<std::vector<std::array<double, 2>>> &lanes2d,
                       std::vector<std::vector<std::array<double, 2>>> &lanesBEV)
{
    YAML::Node root = YAML::LoadFile(filename);
    lanes2d.clear();
    int i = 0;
    for (const auto &laneNode : root["lanes2d"])
    {
        i++;
        std::cout << "Lane " << i << ":\n";
        std::vector<std::array<double, 2>> lane;
        for (const auto &ptNode : laneNode)
        {
            {
                std::cout << "  [" << ptNode[0].as<double>() << ", " << ptNode[1].as<double>() << "]\n";
                lane.push_back({ptNode[0].as<double>(), ptNode[1].as<double>()});
            }
            lanes2d.push_back(lane);
        }
    }

    lanesBEV.clear();
    i = 0;
    for (const auto &laneNode : root["lanes3d"])
    {
        i++;
        std::cout << "Lane " << i << ":\n";
        std::vector<std::array<double, 2>> lane;
        for (const auto &ptNode : laneNode)
        {
            {
                std::cout << "  [" << ptNode[0].as<double>() << ", "
                          << ptNode[1].as<double>() << ", "
                          << ptNode[2].as<double>() << "]\n";
                lane.push_back({ptNode[0].as<double>(), ptNode[2].as<double>()});
            }
            lanesBEV.push_back(lane);
        }
    }
}

void loadHFromYaml(const std::string &filename,
                   std::array<std::array<double, 4>, 3> &H)
{
    YAML::Node root = YAML::LoadFile(filename);
    const auto &camera_intri = root["camera_intri"];
    for (int i = 0; i < camera_intri.size(); i++)
    {
        for (int j = 0; j < camera_intri[i].size(); j++)
        {
            H[i][j] = camera_intri[i][j].as<double>();
            std::cout << H[i][j] << " ";
        }
        std::cout << std::endl;
    }
}

cv::Mat drawLanePoints(const std::vector<std::vector<std::array<double, 2>>> &lanes2d)
{
    // Create blank image
    int width = 1920, height = 1020;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // Draw each lane
    for (const auto &lane : lanes2d)
    {
        for (size_t i = 0; i < lane.size(); ++i)
        {
            cv::Point pt(lane[i][0], lane[i][1]);
            cv::circle(image, pt, 3, cv::Scalar(0, 255, 0), -1); // green dots

            // Optionally connect points with a line
            if (i > 0)
            {
                cv::Point prev(lane[i - 1][0], lane[i - 1][1]);
                cv::line(image, prev, pt, cv::Scalar(255, 0, 0), 1); // blue line
            }
        }
    }

    return image;
}

cv::Mat drawBEV(const std::vector<std::vector<std::array<double, 2>>> &lanesBEV)
{
    // Create blank image
    int width = 1000, height = 1000, scale = 20;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::line(image, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255), 1); // white centerline

    // Draw each lane
    for (const auto &lane : lanesBEV)
    {
        std::vector<double> coeffs;
        fitQuadraticPolynomial(lane, coeffs); // Fit a polynomial to the lane points
        // std::cout<<coeffs[0]<<" "<<coeffs[1]<<" "<<coeffs[2]<<std::endl;


        int prev_u, prev_v;
        for (size_t i = 0; i < lane.size(); ++i)
        {
            int u = lane[i][0] * scale + width / 2;
            int v = height - lane[i][1] * scale;
            cv::Point pt(u, v);
            cv::circle(image, pt, 3, cv::Scalar(0, 255, 0), -1); // green dots

            if (i > 0)
            {
                cv::Point prev(prev_u, prev_v);
                cv::line(image, prev, pt, cv::Scalar(255, 0, 0), 1); // blue line
            }
            prev_u = u;
            prev_v = v;
        }


    }

    return image;
}

void fitQuadraticPolynomial(const std::vector<std::array<double, 2>> &points, std::vector<double> &coefficients)
{
    const int degree = 2;
    const size_t N = points.size();

    Eigen::MatrixXd A(N, degree + 1);
    Eigen::VectorXd b(N);

    for (size_t i = 0; i < N; ++i)
    {
        double x = points[i][0];
        double y = points[i][1];

        A(i, 0) = 1.0;
        A(i, 1) = x;
        A(i, 2) = x * x;
        b(i) = y;
    }

    // Solve using least squares (robust to overdetermined systems)
    Eigen::VectorXd coeffs = A.colPivHouseholderQr().solve(b);

    coefficients.resize(degree + 1);
    for (int i = 0; i <= degree; ++i)
    {
        coefficients[i] = coeffs(i);
    }
}

int main()
{
    std::vector<std::vector<std::array<double, 2>>> lanes2d;
    std::vector<std::vector<std::array<double, 2>>> lanesBEV;
    loadLanesFromYaml("test.yaml", lanes2d, lanesBEV);
    cv::imshow("Camera", drawLanePoints(lanes2d));
    cv::imshow("BEV Lanes",     drawBEV(lanesBEV));

    cv::waitKey(0);
    return 0;
}