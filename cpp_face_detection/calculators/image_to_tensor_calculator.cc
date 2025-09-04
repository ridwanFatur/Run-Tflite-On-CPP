#include "../types/types.h"
#include <algorithm>
#include <cmath>

ImageWithPadding image_to_tensor_calculator(
		const Image &input_image,
		int output_tensor_width,
		int output_tensor_height,
		float output_tensor_float_min,
		float output_tensor_float_max)
{
	float scale_x = static_cast<float>(output_tensor_width) / input_image.width;
	float scale_y = static_cast<float>(output_tensor_height) / input_image.height;
	float scale = std::min(scale_x, scale_y);

	int new_width = static_cast<int>(input_image.width * scale);
	int new_height = static_cast<int>(input_image.height * scale);

	float padding_x = static_cast<float>((output_tensor_width - new_width) / 2);
	float padding_y = static_cast<float>((output_tensor_height - new_height) / 2);

	int padding_x_int = static_cast<int>(padding_x);
	int padding_y_int = static_cast<int>(padding_y);

	Image *output_image = new Image(output_tensor_width, output_tensor_height);

	for (int i = 0; i < output_tensor_width * output_tensor_height * 3; i++)
	{
		output_image->pixels[i] = output_tensor_float_min;
	}

	for (int y = 0; y < new_height; y++)
	{
		for (int x = 0; x < new_width; x++)
		{
			float orig_x = x / scale;
			float orig_y = y / scale;

			int x0 = static_cast<int>(orig_x);
			int y0 = static_cast<int>(orig_y);
			int x1 = std::min(x0 + 1, input_image.width - 1);
			int y1 = std::min(y0 + 1, input_image.height - 1);

			float fx = orig_x - x0;
			float fy = orig_y - y0;

			int out_x = x + padding_x_int;
			int out_y = y + padding_y_int;

			for (int c = 0; c < 3; c++)
			{
				float p00 = input_image.pixels[(y0 * input_image.width + x0) * 3 + c];
				float p01 = input_image.pixels[(y0 * input_image.width + x1) * 3 + c];
				float p10 = input_image.pixels[(y1 * input_image.width + x0) * 3 + c];
				float p11 = input_image.pixels[(y1 * input_image.width + x1) * 3 + c];

				float interpolated = p00 * (1 - fx) * (1 - fy) +
														 p01 * fx * (1 - fy) +
														 p10 * (1 - fx) * fy +
														 p11 * fx * fy;

				float normalized = (interpolated / 255.0f) * (output_tensor_float_max - output_tensor_float_min) + output_tensor_float_min;

				output_image->pixels[(out_y * output_tensor_width + out_x) * 3 + c] = normalized;
			}
		}
	}

	ImageWithPadding result;
	result.image = output_image;
	result.padding[0] = padding_x / output_tensor_width;
	result.padding[1] = padding_y / output_tensor_height;
	result.padding[2] = padding_x / output_tensor_width;
	result.padding[3] = padding_y / output_tensor_height;

	return result;
}