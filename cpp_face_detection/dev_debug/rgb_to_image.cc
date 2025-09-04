#include <opencv2/opencv.hpp>
#include <string>
#include "../types/types.h"

bool rgb_to_image(const Image *image, const std::string &image_path)
{
	if (image == nullptr || image->pixels == nullptr)
	{
		return false;
	}

	cv::Mat cv_image(image->height, image->width, CV_8UC3);

	for (int y = 0; y < image->height; y++)
	{
		for (int x = 0; x < image->width; x++)
		{
			int index = (y * image->width + x) * 3;
			int r = image->pixels[index];
			int g = image->pixels[index + 1];
			int b = image->pixels[index + 2];

			r = std::max(0, std::min(255, r));
			g = std::max(0, std::min(255, g));
			b = std::max(0, std::min(255, b));

			cv_image.at<cv::Vec3b>(y, x) = cv::Vec3b(
					static_cast<uchar>(b),
					static_cast<uchar>(g),
					static_cast<uchar>(r));
		}
	}

	bool success = cv::imwrite(image_path, cv_image);
	return success;
}