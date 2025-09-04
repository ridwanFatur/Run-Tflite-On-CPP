#ifndef YOLO11_INFERENCE_CALCULATOR_H
#define YOLO11_INFERENCE_CALCULATOR_H

#include "../types/types.h"
#include "tensorflow/lite/c/c_api.h"

YoloInferenceOutput yolo11_inference_calculator(
		const Image &input_image);

#endif
