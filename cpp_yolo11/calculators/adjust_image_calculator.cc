#include "../types/types.h"
#include <vector>
#include <cmath>

std::vector<YoloDetectionResult> yolo11_adjust_image_calculator(
		int request_size_width,
		int request_size_height,
		int original_image_width,
		int original_image_height,
		const std::vector<YoloDetectionResult> &detections)
{
	std::vector<YoloDetectionResult> results;
	float scale_x = (float)request_size_width / original_image_width;
	float scale_y = (float)request_size_height / original_image_height;

	for (const auto &det : detections)
	{
		float x = det.x * scale_x;
		float y = det.y * scale_y;
		float w = det.width * scale_x;
		float h = det.height * scale_y;

		YoloDetectionResult result;
		result.confidence = det.confidence;
		result.label_class = det.label_class;
	 	result.x = x;
	 	result.y = y;
	 	result.width = w;
	 	result.height = h;
		
		results.push_back(result);
	}

	return results;
}