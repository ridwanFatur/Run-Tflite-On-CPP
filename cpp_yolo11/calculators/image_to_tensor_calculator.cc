#include "../types/types.h"
#include <algorithm>
#include <cmath>

ImageWithPadding yolo11_image_to_tensor_calculator(
        const Image &input_image,
        int output_tensor_width,
        int output_tensor_height,
        float output_tensor_float_min,
        float output_tensor_float_max) {
    float scale_x = static_cast<float>(output_tensor_width) / input_image.width;
    float scale_y = static_cast<float>(output_tensor_height) / input_image.height;
    float scale = std::min(scale_x, scale_y);

    int new_width = static_cast<int>(input_image.width * scale);
    int new_height = static_cast<int>(input_image.height * scale);

    float padding_x = static_cast<float>((output_tensor_width - new_width) / 2);
    float padding_y = static_cast<float>((output_tensor_height - new_height) / 2);

    int padding_x_int = static_cast<int>(padding_x);
    int padding_y_int = static_cast<int>(padding_y);

    ImageWithPadding result(output_tensor_width, output_tensor_height);

    std::fill(result.image.pixels.begin(), result.image.pixels.end(), output_tensor_float_min);

    const float *input_ptr = input_image.pixels.data();
    float *output_ptr = result.image.pixels.data();
    const int input_width = input_image.width;
    const float inv_scale = 1.0f / scale;
    const float norm_factor = (output_tensor_float_max - output_tensor_float_min) / 255.0f;

    for (int y = 0; y < new_height; y++) {
        const float orig_y = y * inv_scale;
        const int y0 = static_cast<int>(orig_y);
        const int y1 = std::min(y0 + 1, input_image.height - 1);
        const float fy = orig_y - y0;
        const float fy_inv = 1.0f - fy;

        const int out_y = y + padding_y_int;
        const int out_row_offset = out_y * output_tensor_width;

        for (int x = 0; x < new_width; x++) {
            const float orig_x = x * inv_scale;
            const int x0 = static_cast<int>(orig_x);
            const int x1 = std::min(x0 + 1, input_width - 1);
            const float fx = orig_x - x0;
            const float fx_inv = 1.0f - fx;

            const int out_x = x + padding_x_int;
            const int out_idx = (out_row_offset + out_x) * 3;

            const int in_idx_00 = (y0 * input_width + x0) * 3;
            const int in_idx_01 = (y0 * input_width + x1) * 3;
            const int in_idx_10 = (y1 * input_width + x0) * 3;
            const int in_idx_11 = (y1 * input_width + x1) * 3;

            const float w00 = fx_inv * fy_inv;
            const float w01 = fx * fy_inv;
            const float w10 = fx_inv * fy;
            const float w11 = fx * fy;

            output_ptr[out_idx] = (input_ptr[in_idx_00] * w00 +
                                   input_ptr[in_idx_01] * w01 +
                                   input_ptr[in_idx_10] * w10 +
                                   input_ptr[in_idx_11] * w11) *
                                  norm_factor +
                                  output_tensor_float_min;

            output_ptr[out_idx + 1] = (input_ptr[in_idx_00 + 1] * w00 +
                                       input_ptr[in_idx_01 + 1] * w01 +
                                       input_ptr[in_idx_10 + 1] * w10 +
                                       input_ptr[in_idx_11 + 1] * w11) *
                                      norm_factor +
                                      output_tensor_float_min;

            output_ptr[out_idx + 2] = (input_ptr[in_idx_00 + 2] * w00 +
                                       input_ptr[in_idx_01 + 2] * w01 +
                                       input_ptr[in_idx_10 + 2] * w10 +
                                       input_ptr[in_idx_11 + 2] * w11) *
                                      norm_factor +
                                      output_tensor_float_min;
        }
    }

    result.padding_horizontal = padding_x / output_tensor_width;
    result.padding_vertical = padding_y / output_tensor_height;

    return result;
}