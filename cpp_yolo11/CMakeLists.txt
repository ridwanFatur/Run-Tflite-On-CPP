cmake_minimum_required(VERSION 3.10)
project(MyProject)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -O2")

set(TENSORFLOW_SRC "$ENV{HOME}/programs/tensorflow")
set(TENSORFLOW_BUILD_RESULT "$ENV{HOME}/programs/tensorflow_build_result")
set(TENSORFLOW_LIB "${TENSORFLOW_BUILD_RESULT}/linux_x86_64")
set(TENSORFLOW_LIB_GPU_DELEGATE "${TENSORFLOW_BUILD_RESULT}/linux_x86_64")
include_directories(${TENSORFLOW_SRC})

# Find required packages
find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)

# Force use of EGL and GLES2 - no fallback
find_package(PkgConfig REQUIRED)
pkg_check_modules(EGL REQUIRED egl)
pkg_check_modules(GLES2 REQUIRED glesv2)

set(SOURCES
    main.cpp

		calculators/to_image_calculator.cc
		calculators/image_to_tensor_calculator.cc
		calculators/inference_calculator.cc
		calculators/tensors_to_detections_calculator.cc
		calculators/convert_detection_calculator.cc
		calculators/adjust_image_calculator.cc

		dev_debug/draw_box.cc

		types/types.h
    config/config.h 
)

add_executable(main ${SOURCES})

target_link_libraries(
    main 
    ${OpenCV_LIBS} 
    ${TENSORFLOW_LIB}/libtensorflowlite_c.so
    ${TENSORFLOW_LIB_GPU_DELEGATE}/libtensorflowlite_gpu_delegate.so
    ${OPENGL_LIBRARIES}
    ${EGL_LIBRARIES}
    ${GLES2_LIBRARIES}
)