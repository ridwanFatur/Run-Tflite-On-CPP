#ifndef CONVERT_DETECTION_CALCULATOR_H
#define CONVERT_DETECTION_CALCULATOR_H

#include <vector>
#include "../types/types.h"

std::vector<DetectionResult> convert_detection_calculator(
    int image_width,
    int image_height,
    float padding[4],
    int tensor_width,
    int tensor_height,
    const std::vector<DetectionResult> &detections
);

#endif 
