#include "../types/types.h"
#include <vector>
#include <cmath>
#include <algorithm>

std::vector<DetectionResult> tensors_to_detections_calculator(
		const InferenceOutput &inference_output,
		const std::vector<Anchor> &anchors,
		float min_score_thresh,
		float x_scale,
		float y_scale,
		float w_scale,
		float h_scale,
		int num_detections)
{
	std::vector<DetectionResult> results;
	for (int i = 0; i < num_detections; i++)
	{
		float score = inference_output.classificators[i];
		score = std::max(-100.0f, std::min(100.0f, score));
		score = 1.0f / (1.0f + std::exp(-score));

		if (score <= min_score_thresh)
		{
			continue;
		}

		Anchor anchor = anchors[i];
		float anchor_cx = anchor.x;
		float anchor_cy = anchor.y;
		float anchor_w = anchor.w;
		float anchor_h = anchor.h;

		float x_center = inference_output.regressors[i * 16 + 0];
		float y_center = inference_output.regressors[i * 16 + 1];
		float w = inference_output.regressors[i * 16 + 2];
		float h = inference_output.regressors[i * 16 + 3];

		x_center = x_center / x_scale * anchor_w + anchor_cx;
		y_center = y_center / y_scale * anchor_h + anchor_cy;

		h = h / h_scale * anchor_h;
		w = w / w_scale * anchor_w;

		float ymin = y_center - h / 2.0f;
		float xmin = x_center - w / 2.0f;
		float ymax = y_center + h / 2.0f;
		float xmax = x_center + w / 2.0f;

		std::vector<Keypoint> keypoints;
		for (int j = 0; j < 6; j++)
		{
			float keypoint_x = inference_output.regressors[i * 16 + 4 + j * 2 + 0];
			float keypoint_y = inference_output.regressors[i * 16 + 4 + j * 2 + 1];

			keypoint_x = keypoint_x / x_scale * anchor_w + anchor_cx;
			keypoint_y = keypoint_y / y_scale * anchor_h + anchor_cy;

			keypoints.push_back({keypoint_x, keypoint_y});
		}

		DetectionResult result;
		result.score = score;
		result.location_data.relative_bounding_box.xmin = xmin;
		result.location_data.relative_bounding_box.ymin = ymin;
		result.location_data.relative_bounding_box.width = xmax - xmin;
		result.location_data.relative_bounding_box.height = ymax - ymin;
		result.location_data.relative_keypoints = keypoints;

		results.push_back(result);
	}

	return results;
}