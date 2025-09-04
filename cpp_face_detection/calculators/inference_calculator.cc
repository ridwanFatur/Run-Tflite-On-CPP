#include "../types/types.h"
#include <iostream>
#include "tensorflow/lite/c/c_api.h"
#include <cstring>
#include "tensorflow/lite/delegates/gpu/delegate.h"
#include <chrono>

InferenceOutput inference_calculator(const Image &input_image)
{
	InferenceOutput output;
	TfLiteModel *model = TfLiteModelCreateFromFile("../models/face_detection_short_range.tflite");
	TfLiteInterpreterOptions *options = TfLiteInterpreterOptionsCreate();

	TfLiteGpuDelegateOptionsV2 gpu_opts = TfLiteGpuDelegateOptionsV2Default();
	gpu_opts.inference_preference = TFLITE_GPU_INFERENCE_PREFERENCE_SUSTAINED_SPEED;
	gpu_opts.experimental_flags = TFLITE_GPU_EXPERIMENTAL_FLAGS_GL_ONLY;
	TfLiteDelegate *delegate = TfLiteGpuDelegateV2Create(&gpu_opts);
	TfLiteInterpreterOptionsAddDelegate(options, delegate);

	TfLiteInterpreter *interpreter = TfLiteInterpreterCreate(model, options);
	TfLiteInterpreterAllocateTensors(interpreter);
	TfLiteTensor *input_tensor = TfLiteInterpreterGetInputTensor(interpreter, 0);
	TfLiteTensorCopyFromBuffer(input_tensor, input_image.pixels, 128 * 128 * 3 * sizeof(float));

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

	const TfLiteTensor *regressors = TfLiteInterpreterGetOutputTensor(interpreter, 0);
	const TfLiteTensor *classificators = TfLiteInterpreterGetOutputTensor(interpreter, 1);

	output.regressors = new float[896 * 16];
	output.classificators = new float[896 * 1];

	std::memcpy(output.regressors, TfLiteTensorData(regressors), 896 * 16 * sizeof(float));
	std::memcpy(output.classificators, TfLiteTensorData(classificators), 896 * 1 * sizeof(float));

	TfLiteInterpreterDelete(interpreter);
	TfLiteInterpreterOptionsDelete(options);
	TfLiteModelDelete(model);

	return output;
}