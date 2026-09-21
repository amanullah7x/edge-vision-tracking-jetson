# Edge Vision Multi-Object Tracking Pipeline (NVIDIA Jetson AGX Orin)

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Platform](https://img.shields.io/badge/Platform-NVIDIA%20Jetson%20AGX%20Orin-green.svg)]()
[![Inference](https://img.shields.io/badge/Engine-TensorRT%20INT8-orange.svg)]()
[![Telemetry](https://img.shields.io/badge/MAVLink-v2.0-red.svg)]()

> A deterministic computer vision deployment pipeline for high-speed target detection and continuous tracking on resource-constrained embedded companion hardware. Couples **TensorRT INT8 quantized YOLO** with **BoT-SORT** Kalman association and hardware-accelerated GStreamer ingestion to achieve sub-22ms end-to-end loop latency.

---

## 📽️ Demo & Visual Output

The pipeline running real-time aerial target detection and tracking (`tank 0.9235` confidence) from a dynamic drone perspective:

![Aerial Target Tracking Demo](assets/demo.gif)

---

## 🏗️ System Architecture

```
[ Sony IMX415 / MIPI-CSI ]
            │ (Raw Bayer Frames via CSI-2)
            ▼
[ Hardware ISP (Jetson NVMM) ] ──(Zero-Copy DMA Memory)
            │
            ▼
[ TensorRT INT8 Engine ] ───────(Sub-20ms YOLO Inference)
            │
            ▼
[ BoT-SORT Tracker ] ───────────(Kalman Filter + ReID Association)
            │
            ▼
[ Target Guidance Logic ] ──────(LOS Angular Offset & Velocity Vector)
            │
            ▼ (UART / 115200 Baud @ 50 Hz)
[ Pixhawk FCU (PX4/ArduPilot) ]
```

---

## ⚙️ Key Technical Challenges & Solutions

### 1. Thermal Throttling & Inference Latency
* **Problem:** Standard FP32 detection models exceeded the 25W companion compute thermal ceiling and ran at only 15.6 FPS (~64ms/frame), causing severe control loop instability.
* **Solution:** Quantized the network down to INT8 using TensorRT post-training calibration with an embedded validation dataset. Achieved an **18.2ms inference latency** (54.9 FPS) with less than a 1.2% drop in mAP@50.

| Precision Mode | Inference Latency | Throughput (FPS) | VRAM Allocation | Power Draw |
|---|---|---|---|---|
| **FP32** | 64.1 ms | 15.6 FPS | 3.4 GB | 23.8 W |
| **FP16** | 28.4 ms | 35.2 FPS | 1.8 GB | 16.4 W |
| **INT8 (Quantized)** | **18.2 ms** | **54.9 FPS** | **1.1 GB** | **11.2 W** |

### 2. Memory Copy Overhead & Ingestion Jitter
* **Problem:** Ingesting 1080p frames through standard V4L2/OpenCV memory buffers incurred 15–20ms in userspace CPU memory copying before inference even started.
* **Solution:** Architected a zero-copy GStreamer pipeline using `nvarguscamerasrc` and `nvvidconv`. Image buffers are delivered directly to unified Jetson DMA memory pointers (`memory:NVMM`), allowing TensorRT to execute directly without CPU intervention.

---

## 🛠️ Stack & Dependencies
* **Compute:** NVIDIA Jetson AGX Orin / Xavier NX (JetPack 5.1+ / 6.0)
* **Core Languages:** Python 3.10+, C++17
* **Inference & Vision:** TensorRT, Ultralytics YOLOv8/v11, OpenCV 4.8+
* **Tracking:** BoT-SORT / ByteTrack
* **Autopilot Comms:** pymavlink, pyzmq

---

## 🚀 Quickstart & Reproduction

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/amanullah7x/edge-vision-tracking-jetson.git
cd edge-vision-tracking-jetson
pip install -r requirements.txt
```

### 2. Run in Demo / Benchmark Mode
```bash
python run_pipeline.py --max-frames 120 --target-class tank
```

### 3. Deploy on Jetson Companion with Live Camera & MAVLink
```bash
python run_pipeline.py \
  --source "csi://0" \
  --weights "weights/yolo_custom_int8.engine" \
  --target-class "tank" \
  --mavlink "/dev/ttyTHS0" \
  --baud 115200
```
