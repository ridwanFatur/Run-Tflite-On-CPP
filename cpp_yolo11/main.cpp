#include <string>
#include "config/config.h"
#include "types/types.h"
#include <iostream>

#include "calculators/to_image_calculator.h"
#include "calculators/image_to_tensor_calculator.h"
#include "calculators/inference_calculator.h"
#include "calculators/tensors_to_detections_calculator.h"
#include "calculators/convert_detection_calculator.h"
#include "calculators/adjust_image_calculator.h"
#include "dev_debug/draw_box.h"

void printDetections(const std::vector<YoloDetectionResult> &detections)
{
	for (size_t i = 0; i < detections.size(); ++i)
	{
		const YoloDetectionResult &det = detections[i];
		std::cout << "Detection " << i << ":\n";
		std::cout << "  x: " << det.x << "\n";
		std::cout << "  y: " << det.y << "\n";
		std::cout << "  width: " << det.width << "\n";
		std::cout << "  height: " << det.height << "\n";
		std::cout << "  label_class: " << det.label_class << "\n";
		std::cout << "  confidence: " << det.confidence << "\n";
	}
}

int main(int argc, char **argv)
{
	/** Param */
	Yolo11Config config;
	std::string image_path;
	if (argc > 1)
	{
		image_path = argv[1];
	}
	else
	{
		image_path = "../images/yolo_image_example.jpg";
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

	ImageWithPadding image_with_padding = yolo11_image_to_tensor_calculator(
			*image,
			config.tensor_width,
			config.tensor_height,
			config.tensor_float_min,
			config.tensor_float_max);

	YoloInferenceOutput inference_output = yolo11_inference_calculator(
			image_with_padding.image);

	std::vector<YoloDetectionResult> detections = yolo11_tensors_to_detections_calculator(
			inference_output,
			config.conf_threshold,
			config.iou_threshold,
			config.num_classes,
			config.image_size);
	std::vector<YoloDetectionResult> final_detections = yolo11_convert_detection_calculator(
			image->width,
			image->height,
			image_with_padding.padding_vertical,
			image_with_padding.padding_horizontal,
			config.tensor_width,
			config.tensor_height,
			detections);
	std::vector<YoloDetectionResult> adjusted_detections = yolo11_adjust_image_calculator(
			config.request_size_width,
			config.request_size_height,
			image->width,
			image->height,
			final_detections);

	printDetections(adjusted_detections);
	draw_box(image_path, output_adjusted_result_path, adjusted_detections);

	return 0;
}
