#include "path_finder.hpp"

bool gt = false; // true for ground truth, false for BEV points

LanePts::LanePts(int id,
                 std::vector<cv::Point2f> GtPoints,
                 std::vector<cv::Point2f> BevPoints) : id(id), GtPoints(GtPoints), BevPoints(BevPoints) {}

fittedCurve::fittedCurve(const std::array<double, 3> &coeff) : coeff(coeff)
{
    cte = -coeff[2];
    yaw_error = -atan2(coeff[1], 1.0);
    curvature = 2 * coeff[0] / std::pow(1 + coeff[1] * coeff[1], 1.5);
}

void drawLanes(const std::vector<LanePts> &lanes,
               const std::vector<fittedCurve> &egoLanes,
               const fittedCurve &egoPath)
{
    int width = 800, height = 1000, scale = 20, height_offset = 50;
    cv::Mat image = cv::Mat::zeros(height, width, CV_8UC3);
    cv::line(image, cv::Point(width / 2, 0), cv::Point(width / 2, height), cv::Scalar(255, 255, 255), 1); // white centerline
    cv::circle(image, cv::Point(width / 2, height - height_offset), 5, cv::Scalar(0, 0, 255), -1);        // red dot

    int color = 0;
    for (const auto &lane : lanes)
    {
        int prev_u, prev_v;
        color += 20;
        std::vector<cv::Point2f> points;
        if (gt)
            points = lane.GtPoints;
        else
            points = lane.BevPoints;

        for (int i = 0; i < points.size(); i++)
        {
            const auto pt_m = points[i];
            int u = pt_m.x * scale + width / 2;
            int v = height - height_offset - pt_m.y * scale;
            cv::Point pt_pix(u, v);
            cv::circle(image, pt_pix, 3, cv::Scalar(255 - color, color, 0), -1); // green dots
            if (i > 0)
            {
                cv::Point prev(prev_u, prev_v);
                // cv::line(image, prev, pt_pix, cv::Scalar(255, 0, 0), 1); // blue line
            }
            prev_u = u;
            prev_v = v;
        }
    }

    color = 0;
    for (auto egoLane : egoLanes)
    {
        color += 20;
        cv::Point2f prev_pt(0, 0);
        for (double y = 0.0; y <= 50.0; y += 0.1) // y range in meters
        {
            double x = egoLane.coeff[0] * y * y + egoLane.coeff[1] * y + egoLane.coeff[2];
            int u = static_cast<int>(x * scale + width / 2);
            int v = static_cast<int>(height - height_offset - y * scale);
            if (y > 0.0) // skip the first point
            {
                cv::line(image, prev_pt, cv::Point2f(u, v), cv::Scalar(0, color, 255), 1); // yellow
            }
            prev_pt.x = u;
            prev_pt.y = v;
        }
    }

    // Draw ego path
    cv::Point2f prev_pt(0, 0);
    for (double y = 0.0; y <= 50.0; y += 0.1) // y range in meters
    {
        double x = egoPath.coeff[0] * y * y + egoPath.coeff[1] * y + egoPath.coeff[2];
        int u = static_cast<int>(x * scale + width / 2);
        int v = static_cast<int>(height - height_offset - y * scale);
        if (y > 0.0) // skip the first point
        {
            cv::line(image, prev_pt, cv::Point2f(u, v), cv::Scalar(255, 0, 0), 2);
        }
        prev_pt.x = u;
        prev_pt.y = v;
    }

    cv::imshow("BEV Lanes", image);
    // cv::imwrite("../test/bev_lanes.png", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
}

std::vector<LanePts> loadLanesFromYaml(const std::string &filename)
{
    cv::Mat H = loadHFromYaml("../test/image_to_world_transform.yaml");

    std::vector<LanePts> lanes;
    YAML::Node root = YAML::LoadFile(filename);
    int i = 0;
    for (const auto &lane3d : root["lanes3d"])
    {
        const auto &lane2d = root["lanes2d"][i++];
        // std::cout << "LanePts " << i << ":\n";
        std::vector<cv::Point2f> lane_pixels;
        for (const auto &pt2d : lane2d)
        {
            // std::cout << "  [" << pt2d[0].as<double>() << ", " << pt2d[1].as<double>() << "]\n";
            auto noise = generatePixelNoise(5.0);
            double u = pt2d[0].as<double>() + noise[0];
            double v = pt2d[1].as<double>() + noise[1];
            lane_pixels.emplace_back(cv::Point2f(u, v));
        }
        std::vector<cv::Point2f> bev_pixels;
        cv::perspectiveTransform(lane_pixels, bev_pixels, H);

        std::vector<cv::Point2f> gt_pts;
        for (const auto &pt3d : lane3d)
        {
            // std::cout << "  [" << pt3d[0].as<double>() << ", "
            //           << pt3d[1].as<double>() << ", "
            //           << pt3d[2].as<double>() << "]\n";
            gt_pts.emplace_back(cv::Point2f(pt3d[0].as<double>(), pt3d[2].as<double>()));
        }
        lanes.emplace_back(i, gt_pts, bev_pixels);
    }
    return lanes;
}

// Function to generate uv noise in pixel units
std::array<double, 2> generatePixelNoise(double max_noise = 10.0)
{
    static std::random_device rd;
    static std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist_pix(-max_noise, max_noise);
    double du = dist_pix(gen);
    double dv = dist_pix(gen);
    return {du, dv}; // Pixel-space noise to be added to (u, v)
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

    // std::cout << "Loaded H:\n"
    //           << H << std::endl;
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

std::array<double, 3> fitQuadPoly(const std::vector<cv::Point2f> &points)
{
    const int degree = 2;
    const size_t N = points.size();

    if (points.size() == 2)
    {
        double y1 = points[0].y, x1 = points[0].x;
        double y2 = points[1].y, x2 = points[1].x;

        double b = (x2 - x1) / (y2 - y1);
        double c = x1 - b * y1;

        return {0.0, b, c};
    }

    Eigen::MatrixXd A(N, degree + 1);
    Eigen::VectorXd b(N);

    for (size_t i = 0; i < N; ++i)
    {
        double x = points[i].x;
        double y = points[i].y;

        A(i, 0) = y * y;
        A(i, 1) = y;
        A(i, 2) = 1.0;
        b(i) = x;
    }
    std::array<double, 3> coeffs;
    Eigen::VectorXd res = A.colPivHouseholderQr().solve(b);
    for (int i = 0; i <= degree; ++i)
    {
        coeffs[i] = res(i);
    }
    return coeffs;
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
    // std::cout << "Estimated Homography Matrix H:\n"
    //           << H << std::endl;
    // cv::FileStorage fs("image_to_world_transform.yaml", cv::FileStorage::WRITE);
}

fittedCurve calculateEgoPath(const fittedCurve &leftLane, const fittedCurve &rightLane)
{
    return fittedCurve({(leftLane.coeff[0] + rightLane.coeff[0]) / 2.0,
                        (leftLane.coeff[1] + rightLane.coeff[1]) / 2.0,
                        (leftLane.coeff[2] + rightLane.coeff[2]) / 2.0});
}

void Estimator::initialize(const std::vector<double> &init_state, const std::vector<double> &init_var)
{
    dim = init_state.size();
    state = init_state;
    variance = init_var;
    // std::cout << "Filter initialized with: " <<
    //             "dim: "<< dim <<
    //             "state: "<< state[0] <<
    //             "variance" <<std::endl;
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

const std::vector<std::pair<int, int>> lanePairs = {
    {2, 1},
    {2, 0},
    {0, 2},
    {0, 1},
    {2, 0},
    {2, 0},
    {1, 2},
    {0, 3}};

int main()
{
    namespace fs = std::filesystem;
    std::vector<std::pair<std::string, std::string>> file_pairs;
    for (const auto &entry : fs::directory_iterator("../test/000001"))
    {
        if (entry.is_regular_file() && entry.path().extension() == ".yaml")
        {
            std::string stem = entry.path().stem().string();
            std::string yaml_path = entry.path().string();
            std::string img_path = "../test/000001/img/" + stem + ".jpg";

            std::cout << "Found YAML file: " << stem << std::endl;
            file_pairs.emplace_back(yaml_path, img_path);
        }
    }

    // Sort numerically by ID (stem of filename)
    std::sort(file_pairs.begin(), file_pairs.end(),
              [](const auto &a, const auto &b)
              {
                  return std::stoll(fs::path(a.first).stem().string()) <
                         std::stoll(fs::path(b.first).stem().string());
              });

    // Now use sorted img_files
    for (const auto &pair : file_pairs)
    {
        const std::string &img_path = pair.second;

        cv::Mat img = cv::imread(img_path, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cerr << "Failed to load image: " << img_path << std::endl;
            continue;
        }

        cv::imshow("Camera View", img);
        cv::waitKey(1000); // 1 frame per second
    }

    std::vector<fittedCurve> egoPaths, egoPathsGT;

    int i = 0;
    std::cout << "Found " << file_pairs.size() << " YAML files." << std::endl;
    // TODO: show image camera view
    for (const auto &file_pair : file_pairs)
    {
        auto egoLanesPts = loadLanesFromYaml(file_pair.first);
        if (egoLanesPts.size() < 2)
        {
            std::cerr << "Not enough lanes to calculate ego path. Loop id: " << i << std::endl;
            continue;
        }
        // std::cout << "Loaded " << egoLanesPts.size() << " lanes from " << yaml_path << std::endl;
        auto egoLanesPtsLR = {egoLanesPts[lanePairs[i].first], egoLanesPts[lanePairs[i].second]};
        std::vector<fittedCurve> egoLanes, egoLanesGT;

        for (auto lanePts : egoLanesPtsLR)
        {
            std::array<double, 3> coeff_gt = fitQuadPoly(lanePts.GtPoints);
            std::array<double, 3> coeff = fitQuadPoly(lanePts.BevPoints);
            egoLanesGT.emplace_back(fittedCurve(coeff_gt));
            egoLanes.emplace_back(fittedCurve(coeff));
        }

        auto egoPath = calculateEgoPath(egoLanes[0], egoLanes[1]);
        auto egoPathGT = calculateEgoPath(egoLanesGT[0], egoLanesGT[1]);
        i++;

        egoPaths.push_back(egoPath);
        egoPathsGT.push_back(egoPathGT);

        // std::cout << "egoPath: "
        //           << egoPath.cte << " "
        //           << egoPath.yaw_error << " "
        //           << egoPath.curvature << std::endl;

        for (auto &egoLane : egoLanes)
        {
            // std::cout << "egoLane: "
            //           << egoLane.cte << " "
            //           << egoLane.yaw_error << " "
            //           << egoLane.curvature << std::endl;
        }

        drawLanes(egoLanesPts, egoLanes, egoPath);
    }

    // ----------------------
    auto bayesFilter = Estimator();
    const double proc_SD = 0.5;
    const double meas_SD = 0.1;

    std::cout << "process standard dev: " << proc_SD << std::endl;
    std::cout << "measurement standard dev: " << meas_SD << std::endl;

    const std::vector<double> process_var = {proc_SD * proc_SD, proc_SD * proc_SD, proc_SD * proc_SD};
    const std::vector<double> measurement_var = {meas_SD * meas_SD, meas_SD * meas_SD, meas_SD * meas_SD}; // Measurement variance for cte, yaw_error, curvature

    for (int i = 0; i < egoPaths.size(); i++)
    {
        std::vector<double> measurement = {
            egoPaths[i].cte,
            egoPaths[i].yaw_error,
            egoPaths[i].curvature};

        if (i == 0)
            bayesFilter.initialize(measurement, measurement_var);
        else
            bayesFilter.update(measurement, measurement_var);

        const auto &state = bayesFilter.getState();
        const auto &variance = bayesFilter.getVariance();

        std::cout << std::fixed << std::setprecision(6); // uniform precision
        std::cout << "Frame " << i << ":\n";
        std::cout << std::setw(18) << "Ground Truth:" << "\t"
                  << std::setw(12) << egoPathsGT[i].cte << "\t"
                  << std::setw(12) << egoPathsGT[i].yaw_error << "\t"
                  << std::setw(12) << egoPathsGT[i].curvature << "\n";

        std::cout << std::setw(18) << "Measurement:" << "\t"
                  << std::setw(12) << measurement[0] << "\t"
                  << std::setw(12) << measurement[1] << "\t"
                  << std::setw(12) << measurement[2] << "\n";

        std::cout << std::setw(18) << "Filter Estimate:" << "\t"
                  << std::setw(12) << state[0] << "\t"
                  << std::setw(12) << state[1] << "\t"
                  << std::setw(12) << state[2] << "\n";

        std::cout << std::setw(18) << "Filter Variance:" << "\t"
                  << std::setw(12) << variance[0] << "\t"
                  << std::setw(12) << variance[1] << "\t"
                  << std::setw(12) << variance[2] << "\n";

        std::cout << std::setw(18) << "Error:" << "\t"
                  << std::setw(12) << egoPathsGT[i].cte - state[0] << "\t"
                  << std::setw(12) << egoPathsGT[i].yaw_error - state[1] << "\t"
                  << std::setw(12) << egoPathsGT[i].curvature - state[2]  << "\n";


        std::cout << "--------------------------------------------------------\n";
        bayesFilter.predict(process_var);
    }
    // ----------------------

    return 0;
}