#include <vector>
#include "../types/types.h"
#include <opencv2/opencv.hpp>

void draw_box(
		const std::string &image_path,
		const std::string &output_path,
		const std::vector<YoloDetectionResult> &detections)
{
	cv::Mat image = cv::imread(image_path);

	if (image.empty())
	{
		std::cerr << "Failed to load image: " << image_path << std::endl;
		return;
	}

	for (const auto &detection : detections)
	{
		cv::Point top_left(detection.x, detection.y);
		cv::Point bottom_right(detection.x + detection.width, detection.y + detection.height);
		cv::rectangle(image, top_left, bottom_right, cv::Scalar(0, 255, 0), 2); // Green box, thickness 2
	}

	cv::imwrite(output_path, image);
}