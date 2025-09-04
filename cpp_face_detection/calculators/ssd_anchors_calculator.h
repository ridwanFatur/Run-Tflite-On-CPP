#ifndef SSD_ANCHORS_CALCULATOR_H
#define SSD_ANCHORS_CALCULATOR_H

#include <vector>
#include "../types/types.h"

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
    float anchor_offset_y
);

float calculate_scale(
    float min_scale,
    float max_scale,
    int stride_index,
    int total_strides
);

#endif
