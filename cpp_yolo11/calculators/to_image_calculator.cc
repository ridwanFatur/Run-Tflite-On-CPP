#include <opencv2/opencv.hpp>
#include <string>
#include "../types/types.h"

Image *to_image_calculator(const std::string &image_path)
{
	cv::Mat image = cv::imread(image_path, cv::IMREAD_COLOR);
	if (image.empty())
	{
		return nullptr;
	}

	int width = image.cols;
	int height = image.rows;

	Image *result = new Image(width, height);

	for (int y = 0; y < height; y++)
	{
		for (int x = 0; x < width; x++)
		{
			cv::Vec3b pixel = image.at<cv::Vec3b>(y, x);

			int index = (y * width + x) * 3;
			result->pixels[index] = pixel[2];			/** R (from BGR) */ 
			result->pixels[index + 1] = pixel[1]; /** G */ 
			result->pixels[index + 2] = pixel[0]; /** B */
		}
	}

	return result;
}