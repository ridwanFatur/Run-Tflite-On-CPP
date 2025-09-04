#include "../types/types.h"
#include <vector>
#include <cmath>

std::vector<DetectionResult> adjust_image_calculator(
		int request_size_width,
		int request_size_height,
		int original_image_width,
		int original_image_height,
		const std::vector<DetectionResult> &detections)
{
	std::vector<DetectionResult> results;
	float scale_x = (float)request_size_width / original_image_width;
	float scale_y = (float)request_size_height / original_image_height;

	for (const auto &det : detections)
	{
		const auto &bbox = det.location_data.relative_bounding_box;
		float xmin = bbox.xmin * scale_x;
		float ymin = bbox.ymin * scale_y;
		float w = bbox.width * scale_x;
		float h = bbox.height * scale_y;

		std::vector<Keypoint> new_keypoints;

		for (const auto &keypoint : det.location_data.relative_keypoints)
		{
			float kp_x = keypoint.x * scale_x;
			float kp_y = keypoint.y * scale_y;

			new_keypoints.push_back({kp_x, kp_y});
		}

		DetectionResult result;
		result.score = det.score;
		result.location_data.relative_bounding_box = {xmin, ymin, w, h};
		result.location_data.relative_keypoints = new_keypoints;
		results.push_back(result);
	}

	return results;
}