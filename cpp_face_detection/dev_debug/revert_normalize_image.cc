#include "../types/types.h"
#include <algorithm>

Image *revert_normalize_image(
		const Image &input_image,
		float output_tensor_float_min,
		float output_tensor_float_max)
{
	Image *output_image = new Image(input_image.width, input_image.height);

	float range = output_tensor_float_max - output_tensor_float_min;

	for (int i = 0; i < input_image.width * input_image.height * 3; i++)
	{
		float normalized_value = input_image.pixels[i];

		float original_value = (normalized_value - output_tensor_float_min) * 255.0f / range;

		float clamped_value = std::max(0.0f, std::min(255.0f, original_value));

		output_image->pixels[i] = clamped_value;
	}

	return output_image;
}