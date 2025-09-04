#ifndef YOLO11_CONFIG_H
#define YOLO11_CONFIG_H

#include <vector>
#include <string>

struct Yolo11Config
{
	int tensor_width = 640;
	int tensor_height = 640;
	float tensor_float_min = 0.0f;
	float tensor_float_max = 1.0f;
	float conf_threshold = 0.25f;
	float iou_threshold = 0.45;
	int num_classes = 80;
	int image_size = 640;
	int request_size_width = 1024;
	int request_size_height = 682;
};

static Yolo11Config config;

#endif // YOLO11_CONFIG_H