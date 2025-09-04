#include "../types/types.h"
#include <vector>
#include <cmath>
#include <algorithm>
#include <string>

DetectionResult create_weighted_detection(const std::vector<DetectionResult> &candidates)
{
	if (candidates.empty())
	{
		return {};
	}
	if (candidates.size() == 1)
	{
		return candidates[0];
	}

	float w_xmin = 0.0f;
	float w_ymin = 0.0f;
	float w_xmax = 0.0f;
	float w_ymax = 0.0f;
	float total_score = 0.0f;

	const DetectionResult &first_detection = candidates[0];
	const std::vector<Keypoint> &keypoints = first_detection.location_data.relative_keypoints;
	int num_keypoints = keypoints.size();

	std::vector<Keypoint> weighted_keypoints;
	if (num_keypoints > 0)
	{
		weighted_keypoints.resize(num_keypoints);
		for (int i = 0; i < num_keypoints; i++)
		{
			weighted_keypoints[i] = {0.0f, 0.0f};
		}
	}

	for (const auto &candidate : candidates)
	{
		float score = candidate.score;
		total_score += score;
		const RelativeBoundingBox &bbox = candidate.location_data.relative_bounding_box;
		w_xmin += bbox.xmin * score;
		w_ymin += bbox.ymin * score;
		w_xmax += (bbox.xmin + bbox.width) * score;
		w_ymax += (bbox.ymin + bbox.height) * score;

		const std::vector<Keypoint> &candidate_keypoints = candidate.location_data.relative_keypoints;
		for (int i = 0; i < candidate_keypoints.size() && i < weighted_keypoints.size(); i++)
		{
			weighted_keypoints[i].x += candidate_keypoints[i].x * score;
			weighted_keypoints[i].y += candidate_keypoints[i].y * score;
		}
	}

	if (total_score > 0)
	{
		w_xmin /= total_score;
		w_ymin /= total_score;
		w_xmax /= total_score;
		w_ymax /= total_score;
		for (auto &keypoint : weighted_keypoints)
		{
			keypoint.x /= total_score;
			keypoint.y /= total_score;
		}
	}

	DetectionResult weighted_detection;
	weighted_detection.score = candidates[0].score;
	weighted_detection.location_data.relative_bounding_box = {
			w_xmin,
			w_ymin,
			w_xmax - w_xmin,
			w_ymax - w_ymin};
	weighted_detection.location_data.relative_keypoints = weighted_keypoints;

	return weighted_detection;
}

float overlap_similarity(const DetectionResult &box1, const DetectionResult &box2, const std::string &overlap_type)
{
	const RelativeBoundingBox &bbox1 = box1.location_data.relative_bounding_box;
	const RelativeBoundingBox &bbox2 = box2.location_data.relative_bounding_box;

	float x1_min = bbox1.xmin;
	float y1_min = bbox1.ymin;
	float x1_max = x1_min + bbox1.width;
	float y1_max = y1_min + bbox1.height;

	float x2_min = bbox2.xmin;
	float y2_min = bbox2.ymin;
	float x2_max = x2_min + bbox2.width;
	float y2_max = y2_min + bbox2.height;

	float inter_xmin = std::max(x1_min, x2_min);
	float inter_ymin = std::max(y1_min, y2_min);
	float inter_xmax = std::min(x1_max, x2_max);
	float inter_ymax = std::min(y1_max, y2_max);

	if (inter_xmin >= inter_xmax || inter_ymin >= inter_ymax)
	{
		return 0.0f;
	}

	float intersection_area = (inter_xmax - inter_xmin) * (inter_ymax - inter_ymin);
	float area1 = bbox1.width * bbox1.height;
	float area2 = bbox2.width * bbox2.height;

	float normalization;
	if (overlap_type == "INTERSECTION_OVER_UNION" || overlap_type == "JACCARD")
	{
		normalization = area1 + area2 - intersection_area;
	}
	else if (overlap_type == "MODIFIED_JACCARD")
	{
		normalization = area2;
	}
	else
	{
		return 0.0f;
	}

	return normalization > 0 ? intersection_area / normalization : 0.0f;
}

std::vector<DetectionResult> weighted_non_max_suppression(
		std::vector<DetectionResult> sorted_detections,
		const std::string &overlap_type,
		float min_suppression_threshold)
{
	std::vector<DetectionResult> output_detections;
	std::vector<DetectionResult> remaining_detections = sorted_detections;

	while (!remaining_detections.empty())
	{
		DetectionResult current_detection = remaining_detections[0];

		std::vector<DetectionResult> candidates;
		std::vector<DetectionResult> remaining;

		for (const auto &detection : remaining_detections)
		{
			float similarity = overlap_similarity(current_detection, detection, overlap_type);
			if (similarity > min_suppression_threshold)
			{
				candidates.push_back(detection);
			}
			else
			{
				remaining.push_back(detection);
			}
		}

		DetectionResult weighted_detection = create_weighted_detection(candidates);
		output_detections.push_back(weighted_detection);

		if (remaining.size() == remaining_detections.size())
		{
			break;
		}
		remaining_detections = remaining;
	}

	return output_detections;
}

std::vector<DetectionResult> non_max_suppression_calculator(
		const std::vector<DetectionResult> &detections,
		const std::string &overlap_type,
		const std::string &algorithm,
		float min_suppression_threshold)
{
	std::vector<DetectionResult> sorted_detections = detections;
	std::sort(sorted_detections.begin(), sorted_detections.end(),
						[](const DetectionResult &a, const DetectionResult &b)
						{
							return a.score > b.score;
						});

	return weighted_non_max_suppression(sorted_detections, overlap_type, min_suppression_threshold);
}