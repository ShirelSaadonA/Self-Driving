<p align="center">
  <a href="" rel="noopener">
 <img width=200px height=200px src="https://github.com/ShirelSaadonA/Self-Driving/blob/main/test/Screenshot%20from%202024-12-04%2017-44-37.png" alt="Project logo"></a>
</p>



<h3 align="center">SELF DRIVING</h3>

<div align="center">

[![Status](https://img.shields.io/badge/status-active-success.svg)]()
[![GitHub Issues](https://img.shields.io/github/issues/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/issues)
[![GitHub Pull Requests](https://img.shields.io/github/issues-pr/kylelobo/The-Documentation-Compendium.svg)](https://github.com/kylelobo/The-Documentation-Compendium/pulls)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>




# Self-Driving Assistance System

This project is focused on creating a **real-time driver assistance system** using advanced deep learning techniques. The system is designed to enhance road safety by detecting lanes, monitoring deviations, and identifying various road objects.

---

## **Features**

### **1. Lane Detection and Deviation Warning**
- **Purpose:** Identify the current driving lane boundaries in real time and calculate the vehicle's distance from the left and right lane edges.
- **Goal:** Alert the driver when the vehicle deviates from its lane, improving safety.
- **Dataset:** Trained on the **CULane dataset**, ensuring robust performance in diverse driving conditions.

### **2. Object Detection and Classification**
- **Purpose:** Detect and classify road objects to improve situational awareness:
  - **Vehicles:** Cars, buses, trucks, motorcycles, bicycles.
  - **Pedestrians:** Recognize and track people on the road.
  - **Traffic Signs:** Detect and identify traffic lights and stop signs.
- **Technology:** The object detection model is based on **YOLOv5**, with weights converted to TensorRT (TRT) for optimized performance on the NVIDIA Jetson Nano.

---

## **Technologies and Tools**

### **Frameworks**
- **YOLOv5**: Advanced object detection framework used for identifying vehicles, pedestrians, traffic signs, and other objects with high accuracy.  
- **TensorRT**: NVIDIA's high-performance deep learning inference library, used to optimize YOLOv5 and lane detection models for deployment on edge devices.  
- **NVIDIA DeepStream**: End-to-end streaming analytics framework, enabling efficient processing of video feeds and integration of multiple AI models.  
- **ONNX**: Open Neural Network Exchange format, used to convert and integrate pre-trained models like the lane detection model for cross-platform compatibility.  

### **Datasets**
- **Lane Detection**: Leveraged the **CULane dataset**, which includes diverse scenarios such as urban roads, highways, and challenging weather conditions. The dataset was used to train and test the lane detection model.  
- **Object Detection**: Fine-tuned YOLOv5 using a **custom dataset**, tailored to recognize specific classes such as cars, buses, motorcycles, bicycles, and traffic signs. The dataset includes annotated images for optimal performance in real-world applications.  

### **Hardware**
- **NVIDIA Jetson Nano**: Designed for edge computing applications, offering a compact and power-efficient platform for deploying AI models.  
- **CUDA**: NVIDIA's parallel computing platform and API, used to accelerate deep learning computations on the Jetson Nano.  
- **Hardware Optimization**: TensorRT and DeepStream ensure that models run efficiently, maximizing inference speed and accuracy while minimizing latency on the Jetson Nano device.


---




## Howto

### Download ONNX Model

### Clone PINTO_model_zoo repository and download Ultra-Fast-Lane-Detection

### Convert ONNX Model to TensorRT Serialize engine file.

### Run 
```
python3 main.py \
    --model ultra_falst_lane_detection_culane_288x800.trt
    --model_config culane --videopath ... --output test.mp4

```


## TEST

![Image](frame_4213.png)
