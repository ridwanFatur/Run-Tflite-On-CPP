#ifndef YOLO11_IMAGE_TO_TENSOR_CALCULATOR_H
#define YOLO11_IMAGE_TO_TENSOR_CALCULATOR_H

#include "../types/types.h"

ImageWithPadding yolo11_image_to_tensor_calculator(
        const Image &input_image,
        int output_tensor_width,
        int output_tensor_height,
        float output_tensor_float_min,
        float output_tensor_float_max
);

#endif
