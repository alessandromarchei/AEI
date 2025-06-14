#include "path_finder.hpp"

void drawLanes(const std::vector<Lane> &lanes)
{
    int width = 800, height = 1000, scale = 20, height_offset = 50;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::line(image, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255), 1); // white centerline
    cv::circle(image, cv::Point(width / 2, height - height_offset), 5, cv::Scalar(0, 0, 255), -1);        // red dot

    for (const auto &lane : lanes)
    {
        int prev_u, prev_v;
        for (int i = 0; i < lane.GtPoints.size(); i++)
        {
            const auto pt_m = lane.GtPoints[i];
            int u = pt_m[0] * scale + width / 2;
            int v = height - height_offset - pt_m[1] * scale;
            cv::Point pt_pix(u, v);
            cv::circle(image, pt_pix, 3, cv::Scalar(0, 255, 255), -1);
            if (i > 0)
            {
                cv::Point prev(prev_u, prev_v);
                cv::line(image, prev, pt_pix, cv::Scalar(255, 0, 255), 1);
            }
            prev_u = u;
            prev_v = v;
        }
        for (int i = 0; i < lane.BevPoints.size(); i++)
        {
            const auto pt_m = lane.BevPoints[i];
            int u = pt_m.x * scale + width / 2;
            int v = height - height_offset - pt_m.y * scale;
            cv::Point pt_pix(u, v);
            cv::circle(image, pt_pix, 3, cv::Scalar(0, 255, 0), -1); // green dots
            if (i > 0)
            {
                cv::Point prev(prev_u, prev_v);
                cv::line(image, prev, pt_pix, cv::Scalar(255, 0, 0), 1); // blue line
            }
            prev_u = u;
            prev_v = v;
        }
    }
    cv::imshow("BEV Lanes", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

std::vector<Lane> loadLanesFromYaml(const std::string &filename)
{
    cv::Mat H = loadHFromYaml("image_to_world_transform.yaml");

    std::vector<Lane> lanes;
    YAML::Node root = YAML::LoadFile(filename);
    int i = 0;
    for (const auto &lane3d : root["lanes3d"])
    {
        const auto &lane2d = root["lanes2d"][i++];
        std::cout << "Lane " << i << ":\n";
        std::vector<cv::Point2f> lane_pixels;
        for (const auto &pt2d : lane2d)
        {
            std::cout << "  [" << pt2d[0].as<double>() << ", " << pt2d[1].as<double>() << "]\n";
            lane_pixels.emplace_back(cv::Point2f(pt2d[0].as<double>(), pt2d[1].as<double>()));
        }
        std::vector<cv::Point2f> bev_pixels;
        cv::perspectiveTransform(lane_pixels, bev_pixels, H);

        std::vector<std::array<double, 2>> gt_pts;
        for (const auto &pt3d : lane3d)
        {
            std::cout << "  [" << pt3d[0].as<double>() << ", "
                      << pt3d[1].as<double>() << ", "
                      << pt3d[2].as<double>() << "]\n";
            gt_pts.push_back({pt3d[0].as<double>(), pt3d[2].as<double>()});
        }
        lanes.emplace_back(i, gt_pts, bev_pixels);
    }
    return lanes;
}

cv::Mat loadHFromYaml(const std::string &filename)
{
    YAML::Node root = YAML::LoadFile(filename);
    const auto &mat = root["H"];

    if (!mat || mat.size() != 9)
    {
        throw std::runtime_error("Invalid or missing homography matrix (expecting 9 values).");
    }

    cv::Mat H = cv::Mat(3, 3, CV_64F); // Create 3x3 matrix of type double

    for (int i = 0; i < 9; ++i)
    {
        H.at<double>(i / 3, i % 3) = mat[i].as<double>();
    }

    std::cout << "Loaded H:\n"
              << H << std::endl;
    return H;
}

void cameraView(const std::vector<std::vector<cv::Point2f>> &lanes2d) // camera perspective
{
    // Create blank image
    int width = 1920, height = 1020;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);

    // Draw each lane
    for (const auto &lane : lanes2d)
    {
        for (size_t i = 0; i < lane.size(); ++i)
        {
            cv::circle(image, lane[i], 3, cv::Scalar(0, 255, 0), -1); // green dots

            // Optionally connect points with a line
            if (i > 0)
            {
                cv::line(image, lane[i - 1], lane[i], cv::Scalar(255, 0, 0), 1); // blue line
            }
        }
    }
    cv::imshow("Camera View", image);
    cv::waitKey(0);
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

void estimateH()
{
    //   [975.577, 567.689]    //   [1.657, 1.973, 38.649]
    //   [1150.05, 747.249]    //   [1.469, 1.553, 6.53]
    //   [904.635, 567.044]    //   [-1.217, 1.968, 39.064]
    //   [706.126, 741.3]      //   [-1.583, 1.539, 6.644]

    std::vector<cv::Point2f> imagePixels = {
        {1150.05f, 747.249f},
        {706.126f, 741.3f},
        {904.635f, 567.044f},
        {975.577f, 567.689f}};
    std::vector<cv::Point2f> BevPixels = {
        {1.469f, 6.53f},
        {-1.583f, 6.644f},
        {-1.217f, 39.064f},
        {1.657f, 38.649f}};
    cv::Mat H = cv::findHomography(imagePixels, BevPixels);
    std::cout << "Estimated Homography Matrix H:\n" << H << std::endl;
    // cv::FileStorage fs("image_to_world_transform.yaml", cv::FileStorage::WRITE);
}

int main()
{
    estimateH();
    auto lanes = loadLanesFromYaml("test.yaml");
    drawLanes(lanes);
    return 0;
}