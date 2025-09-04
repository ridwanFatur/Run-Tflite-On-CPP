#include "../types/types.h"
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>

void draw_image_keypoints(
		const std::string &image_path,
		const std::string &output_path,
		const std::vector<DetectionResult> &detections)
{
	cv::Mat image = cv::imread(image_path);

	std::vector<cv::Scalar> colors = {
			cv::Scalar(255, 0, 0),
			cv::Scalar(0, 255, 0),
			cv::Scalar(0, 0, 255),
			cv::Scalar(255, 255, 0),
			cv::Scalar(255, 0, 255),
			cv::Scalar(0, 255, 255)};

	for (const auto &detection : detections)
	{
		const auto &bbox = detection.location_data.relative_bounding_box;

		int x = static_cast<int>(bbox.xmin);
		int y = static_cast<int>(bbox.ymin);
		int w = static_cast<int>(bbox.width);
		int h = static_cast<int>(bbox.height);

		cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

		for (size_t i = 0; i < detection.location_data.relative_keypoints.size() && i < 6; ++i)
		{
			const auto &kp = detection.location_data.relative_keypoints[i];

			int kp_x = static_cast<int>(kp.x);
			int kp_y = static_cast<int>(kp.y);

			cv::rectangle(image, cv::Rect(kp_x - 2, kp_y - 2, 5, 5), colors[i], -1);
		}
	}

	cv::imwrite(output_path, image);
}

void draw_image_resized_keypoints(
		const std::string &image_path,
		const std::string &output_path,
		const std::vector<DetectionResult> &detections,
		int request_size_width,
		int request_size_height)
{
	cv::Mat image = cv::imread(image_path);

	cv::resize(image, image, cv::Size(request_size_width, request_size_height));

	std::vector<cv::Scalar> colors = {
			cv::Scalar(255, 0, 0),
			cv::Scalar(0, 255, 0),
			cv::Scalar(0, 0, 255),
			cv::Scalar(255, 255, 0),
			cv::Scalar(255, 0, 255),
			cv::Scalar(0, 255, 255)};

	for (const auto &detection : detections)
	{
		const auto &bbox = detection.location_data.relative_bounding_box;

		int x = static_cast<int>(bbox.xmin);
		int y = static_cast<int>(bbox.ymin);
		int w = static_cast<int>(bbox.width);
		int h = static_cast<int>(bbox.height);

		cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 2);

		for (size_t i = 0; i < detection.location_data.relative_keypoints.size() && i < 6; ++i)
		{
			const auto &kp = detection.location_data.relative_keypoints[i];

			int kp_x = static_cast<int>(kp.x);
			int kp_y = static_cast<int>(kp.y);

			cv::rectangle(image, cv::Rect(kp_x - 2, kp_y - 2, 5, 5), colors[i], -1);
		}
	}

	cv::imwrite(output_path, image);
}