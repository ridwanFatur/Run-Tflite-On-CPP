#include <iostream>
#include <string>
#include <vector>
#include "types/types.h"
#include "config/config.h"

#include "calculators/to_image_calculator.h"
#include "calculators/image_to_tensor_calculator.h"
#include "calculators/inference_calculator.h"
#include "calculators/ssd_anchors_calculator.h"
#include "calculators/tensors_to_detections_calculator.h"
#include "calculators/convert_detection_calculator.h"
#include "calculators/non_max_suppression_calculator.h"
#include "calculators/adjust_image_calculator.h"

#include "types/types.h"
#include "config/config.h"

#include "dev_debug/rgb_to_image.h"
#include "dev_debug/revert_normalize_image.h"
#include "dev_debug/draw_image_keypoints.h"
#include "dev_debug/draw_normalized_image_keypoints.h"

int main(int argc, char **argv)
{
	/** Param */
	Config config;
	std::string image_path;
	if (argc > 1)
	{
		image_path = argv[1];
	}
	else
	{
		image_path = "../images/face_detection_image_example.jpg";
	}
	std::string output_normalized_result_path = "normalized_result.jpg";
	std::string output_original_result_path = "result.jpg";
	std::string output_adjusted_result_path = "adjusted_result.jpg";

	/** ToImageCalculator */
	Image *image = to_image_calculator(image_path);
	if (image == nullptr)
	{
		std::cerr << "Failed to load image." << std::endl;
		return 1;
	}

	/** ImageToTensorCalculator */
	ImageWithPadding image_with_padding = image_to_tensor_calculator(
			*image,
			config.tensor_width,
			config.tensor_height,
			config.tensor_float_min,
			config.tensor_float_max);

	/** InferenceCalculator */
	InferenceOutput inference_output = inference_calculator(*image_with_padding.image);

	/** SSDAnchorsCalculator */
	std::vector<Anchor> anchors = ssd_anchors_calculator(
			config.num_layers,
			config.strides,
			config.min_scale,
			config.max_scale,
			config.option_aspect_ratios,
			config.interpolated_scale_aspect_ratio,
			config.tensor_height,
			config.tensor_width,
			config.anchor_offset_x,
			config.anchor_offset_y);

	/** TensorsToDetectionsCalculator */
	std::vector<DetectionResult> detections = tensors_to_detections_calculator(
			inference_output,
			anchors,
			config.min_score_thresh,
			config.x_scale,
			config.y_scale,
			config.w_scale,
			config.h_scale,
			config.num_detections);

	/** NonMaxSuppressionCalculator */
	std::vector<DetectionResult> filtered_detections = non_max_suppression_calculator(
			detections,
			config.overlap_type,
			config.algorithm,
			config.min_suppression_threshold);

	/** [Debug] NonMaxSuppressionCalculator */
	draw_normalized_image_keypoints(
			*image_with_padding.image,
			image->width,
			image->height,
			output_normalized_result_path,
			filtered_detections);

	/** ConvertDetectionCalculator */
	std::vector<DetectionResult> final_detections = convert_detection_calculator(
			image->width,
			image->height,
			image_with_padding.padding,
			config.tensor_width,
			config.tensor_height,
			filtered_detections);

	/** [Debug] ConvertDetectionCalculator */
	draw_image_keypoints(
			image_path,
			output_original_result_path,
			final_detections);

	/** Adjust Image Calculator */
	std::vector<DetectionResult> adjusted_detections = adjust_image_calculator(
			config.request_size_width,
			config.request_size_height,
			image->width,
			image->height,
			final_detections);

	/** [Debug] Adjust Image Calculator */
	draw_image_resized_keypoints(
			image_path,
			output_adjusted_result_path,
			adjusted_detections,
			config.request_size_width,
			config.request_size_height);

    return 0;
}