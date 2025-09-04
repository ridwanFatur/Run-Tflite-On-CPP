#ifndef YOLO11_CONVERT_DETECTION_CALCULATOR_H
#define YOLO11_CONVERT_DETECTION_CALCULATOR_H

#include <vector>
#include "../types/types.h"

std::vector<YoloDetectionResult> yolo11_convert_detection_calculator(
		int image_width,
		int image_height,
		float padding_horizontal,
		float padding_vertical,
		int tensor_width,
		int tensor_height,
		const std::vector<YoloDetectionResult> &detections);

#endif
