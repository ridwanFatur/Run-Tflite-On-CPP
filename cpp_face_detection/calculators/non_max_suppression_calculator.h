#ifndef NON_MAX_SUPPRESSION_CALCULATOR_H
#define NON_MAX_SUPPRESSION_CALCULATOR_H

#include <vector>
#include <string>
#include "../types/types.h"

std::vector<DetectionResult> non_max_suppression_calculator(
    const std::vector<DetectionResult> &detections,
    const std::string &overlap_type,
    const std::string &algorithm,
    float min_suppression_threshold
);

#endif
