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
### CUDA 
- **Frameworks:** YOLOv5, TensorRT, NVIDIA DeepStream, ONNX.
- **Dataset:**
  - **Lane Detection:** CULane dataset.
  - **Object Detection:** Custom dataset fine-tuned on YOLOv5.
- **Hardware:** Optimized for NVIDIA Jetson Nano to enable edge computing.

---




