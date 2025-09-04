#ifndef YOLO11_TYPES_H
#define YOLO11_TYPES_H

#include <vector>
#include <memory>

struct Image
{
	int width;
	int height;
	std::vector<float> pixels;

	Image(int w, int h)
			: width(w), height(h),
				pixels(w * h * 3)
	{
		pixels.reserve(w * h * 3);
	}
};

struct ImageWithPadding
{
	Image image;
	float padding_horizontal;
	float padding_vertical;

	ImageWithPadding(int w, int h) : image(w, h) {}
};

struct YoloInferenceOutput
{
	std::vector<float> identity;

	YoloInferenceOutput() : identity(1 * 84 * 8400) {}
};

/**  DetectionResult */
struct YoloDetectionResult
{
	float x;
	float y;
	float width;
	float height;
	int label_class;
	float confidence;
};

#endif