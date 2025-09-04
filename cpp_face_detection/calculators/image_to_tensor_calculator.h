#ifndef IMAGE_TO_TENSOR_CALCULATOR_H
#define IMAGE_TO_TENSOR_CALCULATOR_H

#include "../types/types.h"

ImageWithPadding image_to_tensor_calculator(
    const Image &input_image,
    int output_tensor_width,
    int output_tensor_height,
    float output_tensor_float_min,
    float output_tensor_float_max
);

#endif
