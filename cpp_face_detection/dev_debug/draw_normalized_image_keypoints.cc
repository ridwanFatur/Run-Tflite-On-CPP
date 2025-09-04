#include "../types/types.h"
#include <algorithm>
#include <string>
#include <opencv2/opencv.hpp>
#include <iostream>
#include "revert_normalize_image.h"

void draw_normalized_image_keypoints(
		const Image &input_image,
		int original_image_width,
		int original_image_height,
		const std::string &output_path,
		const std::vector<DetectionResult> &detections)
{
	Image *resized_image = revert_normalize_image(
			input_image,
			-1,
			1);

	cv::Mat image(resized_image->height, resized_image->width, CV_8UC3);

	for (int y = 0; y < resized_image->height; y++)
	{
		for (int x = 0; x < resized_image->width; x++)
		{
			int index = (y * resized_image->width + x) * 3;
			int r = resized_image->pixels[index];
			int g = resized_image->pixels[index + 1];
			int b = resized_image->pixels[index + 2];

			r = std::max(0, std::min(255, r));
			g = std::max(0, std::min(255, g));
			b = std::max(0, std::min(255, b));

			image.at<cv::Vec3b>(y, x) = cv::Vec3b(
					static_cast<uchar>(b),
					static_cast<uchar>(g),
					static_cast<uchar>(r));
		}
	}

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
		int x = static_cast<int>(bbox.xmin * 128);
		int y = static_cast<int>(bbox.ymin * 128);
		int w = static_cast<int>(bbox.width * 128);
		int h = static_cast<int>(bbox.height * 128);
		cv::rectangle(image, cv::Rect(x, y, w, h), cv::Scalar(0, 0, 255), 1);
	}

	cv::imwrite(output_path, image);
}