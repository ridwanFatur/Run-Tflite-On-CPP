#include "../types/types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>
#include <iomanip>

std::vector<DetectionResult> convert_detection_calculator(
		int image_width,
		int image_height,
		float padding[4],
		int tensor_width,
		int tensor_height,
		const std::vector<DetectionResult> &detections)
{
	int iw = image_width;
	int ih = image_height;

	float pt = padding[0];
	float pb = padding[1];
	float pl = padding[2];
	float pr = padding[3];

	float cw = static_cast<float>(tensor_width) - pl - pr;
	float ch = static_cast<float>(tensor_height) - pt - pb;

	float scale = std::min(cw / static_cast<float>(iw), ch / static_cast<float>(ih));
	float sw = static_cast<float>(iw) * scale;
	float sh = static_cast<float>(ih) * scale;
	float ox = (cw - sw) / 2.0f;
	float oy = (ch - sh) / 2.0f;

	std::vector<DetectionResult> results;

	for (const auto &det : detections)
	{
		const auto &bbox = det.location_data.relative_bounding_box;
		float xmin = bbox.xmin;
		float ymin = bbox.ymin;
		float w = bbox.width;
		float h = bbox.height;

		float x1 = ((xmin * static_cast<float>(tensor_width)) - pl - ox) / scale;
		float y1 = ((ymin * static_cast<float>(tensor_height)) - pt - oy) / scale;
		float x2 = (((xmin + w) * static_cast<float>(tensor_width)) - pl - ox) / scale;
		float y2 = (((ymin + h) * static_cast<float>(tensor_height)) - pt - oy) / scale;

		x1 = std::max(0.0f, x1);
		y1 = std::max(0.0f, y1);
		x2 = std::min(static_cast<float>(iw), x2);
		y2 = std::min(static_cast<float>(ih), y2);

		std::vector<Keypoint> new_keypoints;
		for (const auto &keypoint : det.location_data.relative_keypoints)
		{
			float kp_x = keypoint.x;
			float kp_y = keypoint.y;

			float kp_x_transformed = ((kp_x * static_cast<float>(tensor_width)) - pl - ox) / scale;
			float kp_y_transformed = ((kp_y * static_cast<float>(tensor_height)) - pt - oy) / scale;

			kp_x_transformed = std::max(0.0f, std::min(static_cast<float>(iw), kp_x_transformed));
			kp_y_transformed = std::max(0.0f, std::min(static_cast<float>(ih), kp_y_transformed));

			new_keypoints.push_back({kp_x_transformed, kp_y_transformed});
		}

		DetectionResult result;
		result.score = det.score;
		result.location_data.relative_bounding_box = {x1, y1, x2 - x1, y2 - y1};
		result.location_data.relative_keypoints = new_keypoints;

		results.push_back(result);
	}

	return results;
}