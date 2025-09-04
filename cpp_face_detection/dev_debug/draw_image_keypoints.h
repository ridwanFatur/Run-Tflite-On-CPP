#ifndef DRAW_IMAGE_KEYPOINTS_H
#define DRAW_IMAGE_KEYPOINTS_H

#include <string>
#include <vector>
#include "../types/types.h"

void draw_image_keypoints(
		const std::string &image_path,
		const std::string &output_path,
		const std::vector<DetectionResult> &detections);

void draw_image_resized_keypoints(
		const std::string &image_path,
		const std::string &output_path,
		const std::vector<DetectionResult> &detections,
		int request_size_width,
		int request_size_height);

#endif
