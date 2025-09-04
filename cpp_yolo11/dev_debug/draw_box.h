#ifndef DRAW_BOX_H
#define DRAW_BOX_H

#include <string>
#include <vector>
#include "../types/types.h"

void draw_box(
    const std::string &image_path,
    const std::string &output_path,
    const std::vector<YoloDetectionResult> &detections);

#endif // DRAW_BOX_H
