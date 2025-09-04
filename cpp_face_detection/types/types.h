#ifndef TYPES_H
#define TYPES_H
#include <vector>
#pragma once

struct Image
{
	int width;
	int height;
	float *pixels;

	Image(int w, int h)
			: width(w), height(h)
	{
		pixels = new float[width * height * 3];
	}

	~Image()
	{
		delete[] pixels;
	}
};

struct ImageWithPadding
{
	Image *image;
	/** left, top, right, bottom */
	float padding[4];
};

struct InferenceOutput
{
	float *regressors;
	float *classificators;

	InferenceOutput()
			: regressors(nullptr), classificators(nullptr)
	{
	}

	~InferenceOutput()
	{
		delete[] regressors;
		delete[] classificators;
	}
};

struct Anchor
{
	float x;
	float y;
	float w;
	float h;
};

/**  DetectionResult */
struct Keypoint
{
	float x;
	float y;
};

struct RelativeBoundingBox
{
	float xmin;
	float ymin;
	float width;
	float height;
};

struct LocationData
{
	RelativeBoundingBox relative_bounding_box;
	std::vector<Keypoint> relative_keypoints;
};

struct DetectionResult
{
	float score;
	LocationData location_data;
};

#endif
