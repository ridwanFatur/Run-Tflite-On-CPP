#include "../types/types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <iostream>

void non_max_suppression_calculator(
		std::vector<YoloDetectionResult> &detections,
		float conf_threshold,
		float iou_threshold)
{
	std::vector<int> valid_indices;
	valid_indices.reserve(detections.size());

	for (int i = 0; i < detections.size(); ++i)
	{
		if (detections[i].confidence >= conf_threshold)
		{
			valid_indices.push_back(i);
		}
	}

	if (valid_indices.empty())
	{
		detections.clear();
		return;
	}

	std::sort(valid_indices.begin(), valid_indices.end(),
						[&detections](int a, int b)
						{ return detections[a].confidence > detections[b].confidence; });

	std::vector<int> keep_indices;
	std::vector<bool> suppressed(valid_indices.size(), false);

	for (int i = 0; i < valid_indices.size(); ++i)
	{
		if (suppressed[i])
			continue;

		int current_idx = valid_indices[i];
		keep_indices.push_back(current_idx);

		const auto &current_det = detections[current_idx];
		float current_x1 = current_det.x;
		float current_y1 = current_det.y;
		float current_x2 = current_det.x + current_det.width;
		float current_y2 = current_det.y + current_det.height;

		for (int j = i + 1; j < valid_indices.size(); ++j)
		{
			if (suppressed[j])
				continue;

			int other_idx = valid_indices[j];
			const auto &other_det = detections[other_idx];
			float other_x1 = other_det.x;
			float other_y1 = other_det.y;
			float other_x2 = other_det.x + other_det.width;
			float other_y2 = other_det.y + other_det.height;

			float x1 = std::max(current_x1, other_x1);
			float y1 = std::max(current_y1, other_y1);
			float x2 = std::min(current_x2, other_x2);
			float y2 = std::min(current_y2, other_y2);

			float w = std::max(0.0f, x2 - x1);
			float h = std::max(0.0f, y2 - y1);
			float intersection = w * h;

			float current_area = current_det.width * current_det.height;
			float other_area = other_det.width * other_det.height;
			float union_area = current_area + other_area - intersection;

			float iou = (union_area > 0) ? intersection / union_area : 0.0f;

			if (iou > iou_threshold)
			{
				suppressed[j] = true;
			}
		}
	}

	std::vector<YoloDetectionResult> filtered_detections;
	filtered_detections.reserve(keep_indices.size());
	for (int idx : keep_indices)
	{
		filtered_detections.push_back(detections[idx]);
	}

	detections = std::move(filtered_detections);
}

std::vector<YoloDetectionResult> yolo11_tensors_to_detections_calculator(
		const YoloInferenceOutput &inference_output,
		float conf_threshold,
		float iou_threshold,
		int num_classes,
		int image_size)
{
	const auto &output = inference_output.identity;
	std::vector<YoloDetectionResult> detections;

	for (int i = 0; i < 8400; ++i)
	{
		int max_class_idx = 0;
		float max_prob = output[4 * 8400 + i]; 

		for (int j = 1; j < num_classes; ++j)
		{
			float prob = output[(4 + j) * 8400 + i];
			if (prob > max_prob)
			{
				max_prob = prob;
				max_class_idx = j;
			}
		}

		float confidence = max_prob;

		if (confidence > conf_threshold)
		{
			float center_x = output[0 * 8400 + i];
			float center_y = output[1 * 8400 + i];
			float w = output[2 * 8400 + i];
			float h = output[3 * 8400 + i];

			float x_min = (center_x - w / 2) * image_size;
			float y_min = (center_y - h / 2) * image_size;
			float x_max = (center_x + w / 2) * image_size;
			float y_max = (center_y + h / 2) * image_size;

			YoloDetectionResult detection;
			detection.x = x_min;
			detection.y = y_min;
			detection.width = x_max - x_min;
			detection.height = y_max - y_min;
			detection.label_class = max_class_idx;
			detection.confidence = confidence;

			detections.push_back(detection);
		}
	}

	non_max_suppression_calculator(detections, conf_threshold, iou_threshold);

	return detections;
}