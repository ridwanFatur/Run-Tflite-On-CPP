#include "../types/types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

std::vector<YoloDetectionResult> yolo11_convert_detection_calculator(
		int image_width,
		int image_height,
		float padding_vertical,
		float padding_horizontal,
		int tensor_width,
		int tensor_height,
		const std::vector<YoloDetectionResult> &detections)
{
	int iw = image_width;
	int ih = image_height;

	float cw = static_cast<float>(tensor_width) - 2 * padding_horizontal;
	float ch = static_cast<float>(tensor_height) - 2 * padding_vertical;

	float scale = std::min(cw / static_cast<float>(iw), ch / static_cast<float>(ih));
	float sw = static_cast<float>(iw) * scale;
	float sh = static_cast<float>(ih) * scale;
	float ox = (cw - sw) / 2.0f;
	float oy = (ch - sh) / 2.0f;

	std::vector<YoloDetectionResult> results;

	for (const auto &det : detections)
	{
		float x = det.x;
		float y = det.y;
		float w = det.width;
		float h = det.height;

		float x1 = ((x)-padding_horizontal - ox) / scale;
		float y1 = ((y)-padding_vertical - oy) / scale;
		float x2 = (((x + w)) - padding_horizontal - ox) / scale;
		float y2 = (((y + h)) - padding_vertical - oy) / scale;

		x1 = std::max(0.0f, x1);
		y1 = std::max(0.0f, y1);
		x2 = std::min(static_cast<float>(iw), x2);
		y2 = std::min(static_cast<float>(ih), y2);

		YoloDetectionResult result;
		result.confidence = det.confidence;
		result.label_class = det.label_class;
		result.x = x1;
		result.y = y1;
		result.width = x2 - x1;
		result.height = y2 - y1;

		results.push_back(result);
	}

	return results;
}