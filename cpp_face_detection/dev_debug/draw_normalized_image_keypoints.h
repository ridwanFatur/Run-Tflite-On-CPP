#ifndef DRAW_NORMALIZED_IMAGE_KEYPOINTS_H
#define DRAW_NORMALIZED_IMAGE_KEYPOINTS_H

#include <string>
#include <vector>
#include "../types/types.h"

void draw_normalized_image_keypoints(
		const Image &input_image,
		int original_image_width,
		int original_image_height,
		const std::string &output_path,
		const std::vector<DetectionResult> &detections);

#endif
