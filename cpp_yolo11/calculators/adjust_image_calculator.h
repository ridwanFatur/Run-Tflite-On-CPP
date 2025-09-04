#ifndef YOLO11_ADJUST_IMAGE_CALCULATOR_H
#define YOLO11_ADJUST_IMAGE_CALCULATOR_H

#include <vector>
#include "../types/types.h"

std::vector<YoloDetectionResult> yolo11_adjust_image_calculator(
		int request_size_width,
		int request_size_height,
		int original_image_width,
		int original_image_height,
		const std::vector<YoloDetectionResult> &detections);

#endif // YOLO11_ADJUST_IMAGE_CALCULATOR_H