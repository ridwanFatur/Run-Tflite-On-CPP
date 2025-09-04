#include "../types/types.h"
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include <cstring>
#include <chrono>

YoloInferenceOutput yolo11_inference_calculator(
		const Image &input_image)
{
	YoloInferenceOutput output;
	TfLiteModel *model = TfLiteModelCreateFromFile("../models/yolo11n_float16.tflite");
	TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();

	TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
	TfLiteInterpreterAllocateTensors(interpreter);
	TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
	TfLiteTensorCopyFromBuffer(input_tensor, input_image.pixels.data(),
														 640 * 640 * 3 * sizeof(float));

	auto start_time = std::chrono::high_resolution_clock::now();

	if (TfLiteInterpreterInvoke(interpreter) != kTfLiteOk)
	{
		TfLiteInterpreterDelete(interpreter);
		TfLiteInterpreterOptionsDelete(options);
		TfLiteModelDelete(model);
		return output;
	}

	auto end_time = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> inference_time = end_time - start_time;
	std::cout << "Inference time: " << inference_time.count() << " ms" << std::endl;
	
	const TfLiteTensor *identity = TfLiteInterpreterGetOutputTensor(interpreter, 0);
	std::memcpy(output.identity.data(), TfLiteTensorData(identity), 1 * 84 * 8400 * sizeof(float));

	TfLiteInterpreterDelete(interpreter);
	TfLiteInterpreterOptionsDelete(options);
	TfLiteModelDelete(model);

	return output;
}