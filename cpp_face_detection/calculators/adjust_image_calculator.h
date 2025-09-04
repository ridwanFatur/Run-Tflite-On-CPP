#ifndef ADJUST_IMAGE_CALCULATOR_H
#define ADJUST_IMAGE_CALCULATOR_H

#include <vector>
#include "../types/types.h"

std::vector<DetectionResult> adjust_image_calculator(
		int request_size_width,
		int request_size_height,
		int original_image_width,
		int original_image_height,
		const std::vector<DetectionResult> &detections);

#endif // ADJUST_IMAGE_CALCULATOR_H