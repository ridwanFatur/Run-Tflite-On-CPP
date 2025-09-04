#ifndef REVERT_NORMALIZE_IMAGE_H
#define REVERT_NORMALIZE_IMAGE_H

#include "../types/types.h"

Image *revert_normalize_image(
    const Image &input_image,
    float output_tensor_float_min,
    float output_tensor_float_max);

#endif
