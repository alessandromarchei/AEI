#include <iostream>
#include <vector>
#include <string>
#include <stdexcept>

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

using namespace cv; 
using namespace std; 

int main(int argc, char* argv[]) {
    if (argc != 2) {
        std::cerr << "Usage: ./video_visualization <input_video.mp4>" << std::endl;
        return -1;
    }

    const std::string input_video_path = argv[1];

    try {
        // Initialize video capture
        cv::VideoCapture cap(input_video_path);
        if (!cap.isOpened()) {
            throw std::runtime_error("Error opening video stream or file: " + input_video_path);
        }

        const double fps = cap.get(cv::CAP_PROP_FPS);
        std::cout << "Frame rate of input video: " << fps << " FPS" << std::endl;

        // Display video frames
        std::cout << "Processing video... Press ESC to stop." << std::endl;
        cv::Mat frame;
        while (cap.read(frame)) {
            cv::imshow("Play video", frame);
            if (cv::waitKey(1) == 27) { // Stop if 'ESC' is pressed
                std::cout << "Processing stopped by user." << std::endl;
                break;
            }
        }
        
        // Clean up
        cap.release();
        cv::destroyAllWindows();
    } catch (const std::exception& e) {
        std::cerr << "Standard error: " << e.what() << std::endl;
        return -1;
    }

    return 0;
} 