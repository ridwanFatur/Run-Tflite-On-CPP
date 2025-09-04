# Run-Tflite-On-CPP

This project demonstrates how to run **TensorFlow Lite** models directly in **C++** using the prebuilt `libtensorflowlite_c.so` library on Linux.  
It includes two sample implementations:

- Face Detection → Using MediaPipe's `face_detection_short_range.tflite` model.  
- Object Detection (YOLOv11) → Using Ultralytics YOLOv11 model in TensorFlow Lite format.

## 📂 Project Structure

Run-Tflite-On-CPP/
├── cpp_face_detection/
│   ├── build/
│   └── CMakeLists.txt
├── cpp_yolo11/
│   ├── build/
│   └── CMakeLists.txt
└── README.md

## 🚀 Build & Run

1. Clone this repository.  
2. Build the Face Detection example:

   cd cpp_face_detection/build  
   cmake ..  
   make  
   ./main  

3. Build the YOLOv11 example:

   cd cpp_yolo11/build  
   cmake ..  
   make  
   ./main  

## 📸 Output

- The detection results (images with bounding boxes and keypoints) will be saved inside each `build/` folder.  
- Example:  

  cpp_face_detection/build/result.jpg  
  cpp_yolo11/build/result.jpg  
