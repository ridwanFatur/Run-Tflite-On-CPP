#ifndef YOLO11_TENSORS_TO_DETECTIONS_CALCULATOR_H
#define YOLO11_TENSORS_TO_DETECTIONS_CALCULATOR_H

#include <vector>
#include "../types/types.h"

std::vector<YoloDetectionResult> yolo11_tensors_to_detections_calculator(
    const YoloInferenceOutput &inference_output,
    float conf_threshold,
    float iou_threshold,
    int num_classes,
    int image_size);

#endif // YOLO11_TENSORS_TO_DETECTIONS_CALCULATOR_H