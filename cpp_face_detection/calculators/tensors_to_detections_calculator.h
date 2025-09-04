#ifndef TENSORS_TO_DETECTIONS_CALCULATOR_H
#define TENSORS_TO_DETECTIONS_CALCULATOR_H

#include "../types/types.h"
#include <vector>

std::vector<DetectionResult> tensors_to_detections_calculator(
		const InferenceOutput &inference_output,
		const std::vector<Anchor> &anchors,
		float min_score_thresh,
		float x_scale,
		float y_scale,
		float w_scale,
		float h_scale,
		int num_detections);

#endif
