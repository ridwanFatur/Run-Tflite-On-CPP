#include "../types/types.h"
#include <vector>
#include <cmath>

float calculate_scale(float min_scale, float max_scale, int layer_index, int num_layers)
{
	if (num_layers == 1)
	{
		return (min_scale + max_scale) * 0.5f;
	}
	else
	{
		return min_scale + (max_scale - min_scale) * layer_index / (num_layers - 1);
	}
}

std::vector<Anchor> ssd_anchors_calculator(
		int num_layers,
		const std::vector<int> &strides,
		float min_scale,
		float max_scale,
		const std::vector<float> &option_aspect_ratios,
		float interpolated_scale_aspect_ratio,
		int input_size_height,
		int input_size_width,
		float anchor_offset_x,
		float anchor_offset_y)
{
	std::vector<Anchor> anchors;
	int layer_id = 0;
	while (layer_id < num_layers)
	{
		std::vector<float> anchor_height;
		std::vector<float> anchor_width;
		std::vector<float> aspect_ratios;
		std::vector<float> scales;

		int last_same_stride_layer = layer_id;
		while (last_same_stride_layer < strides.size() && strides[last_same_stride_layer] == strides[layer_id])
		{
			float scale = calculate_scale(min_scale, max_scale, last_same_stride_layer, strides.size());
			for (int aspect_ratio_id = 0; aspect_ratio_id < option_aspect_ratios.size(); aspect_ratio_id++)
			{
				aspect_ratios.push_back(option_aspect_ratios[aspect_ratio_id]);
				scales.push_back(scale);
			}

			float scale_next = (last_same_stride_layer == strides.size() - 1) ? 1.0f : calculate_scale(min_scale, max_scale, last_same_stride_layer + 1, strides.size());

			scales.push_back(sqrt(scale * scale_next));
			aspect_ratios.push_back(interpolated_scale_aspect_ratio);
			last_same_stride_layer++;
		}

		for (int i = 0; i < aspect_ratios.size(); i++)
		{
			float ratio_sqrts = sqrt(aspect_ratios[i]);
			anchor_height.push_back(scales[i] / ratio_sqrts);
			anchor_width.push_back(scales[i] * ratio_sqrts);
		}

		int feature_map_height = 0;
		int feature_map_width = 0;

		int stride = strides[layer_id];
		feature_map_height = ceil(1.0f * input_size_height / stride);
		feature_map_width = ceil(1.0f * input_size_width / stride);

		for (int y = 0; y < feature_map_height; y++)
		{
			for (int x = 0; x < feature_map_width; x++)
			{
				for (size_t anchor_id = 0; anchor_id < anchor_height.size(); anchor_id++)
				{
					float x_center = (x + anchor_offset_x) * 1.0f / feature_map_width;
					float y_center = (y + anchor_offset_y) * 1.0f / feature_map_height;
					anchors.push_back({x_center, y_center, 1.0f, 1.0f});
				}
			}
		}

		layer_id = last_same_stride_layer;
	}

	return anchors;
}
