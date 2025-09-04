#ifndef CONFIG_H
#define CONFIG_H

#include <vector>
#include <string>

struct Config
{
	int tensor_width = 128;
	int tensor_height = 128;
	float tensor_float_min = -1.0f;
	float tensor_float_max = 1.0f;

	int num_layers = 4;
	std::vector<int> strides = {8, 16, 16, 16};
	float min_scale = 0.1484375f;
	float max_scale = 0.75f;
	std::vector<float> option_aspect_ratios = {1.0f};
	float interpolated_scale_aspect_ratio = 1.0f;
	float anchor_offset_x = 0.5f;
	float anchor_offset_y = 0.5f;

	float x_scale = 128.0f;
	float y_scale = 128.0f;
	float h_scale = 128.0f;
	float w_scale = 128.0f;
	float min_score_thresh = 0.5f;
	int num_detections = 896;

	std::string overlap_type = "INTERSECTION_OVER_UNION";
	std::string algorithm = "WEIGHTED";
	float min_suppression_threshold = 0.3f;

	int request_size_width = 1080;
	int request_size_height = 1737;
};

#endif // CONFIG_H