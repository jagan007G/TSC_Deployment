# TSC Deployment on ESP32

> A lightweight, real-time environmental sound classification system for smart bathroom activity recognition using TinyML and the ESP32 platform.

![Platform](https://img.shields.io/badge/Platform-ESP32-blue)
![Framework](https://img.shields.io/badge/Framework-ESP--IDF-green)
![Language](https://img.shields.io/badge/Language-C%2FC%2B%2B-orange)
![TinyML](https://img.shields.io/badge/TinyML-TensorFlow_Lite_Micro-red)
![Status](https://img.shields.io/badge/Status-Active-success)

---

# Overview

This project implements a **real-time sound-driven activity recognition system** that detects common bathroom activities using only environmental audio captured by an onboard microphone.

Unlike cloud-based audio recognition systems, all signal processing and machine learning inference are performed **locally on an ESP32 microcontroller**, enabling:

- Low latency
- Privacy-preserving operation
- Low power consumption
- Offline functionality
- Cost-effective deployment

The system continuously records audio, preprocesses the signal, extracts meaningful acoustic information, performs noise reduction, and classifies activities using a TensorFlow Lite Micro (TFLM) neural network.

---

# Project Motivation

Human activity recognition traditionally relies on wearable sensors or cameras. These approaches suffer from several limitations:

- Privacy concerns
- User inconvenience
- High deployment cost
- Dependency on lighting conditions
- Limited accessibility monitoring

Environmental sound provides a natural, non-invasive alternative.

Bathroom activities such as:

- Toilet flushing
- Water usage
- Showering
- Door operation
- Walking with assistive devices

produce unique acoustic signatures that can be recognized without recording speech or requiring wearable devices.

This project demonstrates that embedded TinyML models can accurately classify these sounds directly on resource-constrained hardware.

---

# Features

- Real-time sound classification
- Fully offline operation
- TensorFlow Lite Micro inference
- Audio preprocessing pipeline
- Noise reduction
- Novelty detection
- Sliding window inference
- Circular audio buffering
- Automatic WAV recording
- SD card logging
- OLED status display
- ESP-IDF implementation
- Optimized for embedded deployment

---

# Target Activities

The current model classifies the following seven sound categories:

| Class ID | Activity |
|-----------|----------|
| 0 | Flush |
| 1 | Door |
| 2 | Basin Tap |
| 3 | Walker / Crutch |
| 4 | Bathroom Tap |
| 5 | No Activity |
| 6 | Shower |

---

# Model: DPNet

The classifier used in this project is **DPNet**, a deep learning model for sound classification. DPNet is a convolutional neural network that learns directly from the **raw audio waveform**, without a spectrogram or MFCC front end.

The model's design and training are maintained in a separate repository. This repository covers only its **deployment** on the ESP32.

🔗 **Model repository:** [DPNet](https://github.com/debolina-34/DPNet)

---

# Dataset

DPNet is trained on **SnaanGhar7**, a custom bathroom sound dataset covering the seven activity classes listed above.

## 🔗 Accessing the Dataset

To access the SnaanGhar7 dataset, please fill out the [Dataset Access Request Form](https://docs.google.com/forms/d/e/1FAIpQLSeG6SI1RTFGJUxEYZn49DJKLeNmeOQVrni4vGBuXhI87fPw9w/viewform?usp=dialog).

**Steps:**

1. Complete the access request form with your details and research purpose
2. Your request will be reviewed within 2-3 business days
3. Upon approval, you will receive a download link via email

**License:** CC BY-NC 4.0 (Non-commercial academic use only)

---

# Hardware

The system is designed for ESP32-based embedded platforms.

## Main Components

- ESP32-P4 Nano
- Onboard microphone (ES8311 codec)
- SD Card
- OLED Display
- External Flash
- PSRAM

---

# Software Stack

- ESP-IDF
- TensorFlow Lite Micro
- ESP-DL
- ESP-NN
- FATFS
- FreeRTOS

---

# System Architecture

```
Microphone
     │
     ▼
Audio Capture
     │
     ▼
Preprocessing
     │
     ▼
Noise Reduction
     │
     ▼
Novelty Detection
     │
     ▼
Feature Preparation
     │
     ▼
DPNet Neural Network
     │
     ▼
Prediction
     │
     ▼
Display + Logging + WAV Storage
```

---

# Audio Processing Pipeline

The embedded DSP pipeline consists of several stages designed to improve robustness in noisy bathroom environments.

## 1. Audio Acquisition

- Sampling Rate: 20 kHz
- 16-bit PCM
- Continuous streaming
- Ring buffer storage

---

## 2. Preprocessing

The incoming audio undergoes several preprocessing steps:

- DC offset removal
- High-pass filtering
- Low-pass filtering
- Signal normalization
- Gain adjustment

---

## 3. Noise Reduction

To improve recognition accuracy:

- Background noise estimation
- Wiener filtering
- Spectral subtraction
- Adaptive gain control

---

## 4. Novelty Detection

Before running inference, the system determines whether the incoming audio contains significant acoustic events.

Benefits:

- Reduces unnecessary inference
- Saves computation
- Improves efficiency

---

## 5. Sliding Window Processing

The system processes audio using overlapping windows.

Advantages:

- Continuous recognition
- Reduced missed events
- Improved temporal stability

---

## 6. Deep Learning Inference

The processed audio is fed into a TensorFlow Lite Micro implementation of **DPNet** (see [Model: DPNet](#model-dpnet)), which predicts the most probable activity class.

---

# Real-Time Processing Flow

```
Capture Audio
      │
      ▼
Ring Buffer
      │
      ▼
Noise Reduction
      │
      ▼
Novelty Detection
      │
      ▼
Prepare Input Tensor
      │
      ▼
TensorFlow Lite Micro
      │
      ▼
Classification
      │
      ├── OLED Display
      ├── Serial Monitor
      └── SD Card Logging
```

---

# Repository Structure

```
.
├── main/
|   ├── CMakeLists.txt
|   ├── DPNet_ori_a_hat_55x3.cc
|   ├── DPNet_ori_a_hat_55x3.h
|   ├── DPNet_ori_a_hat_55x3.tflite
|   ├── Kconfig.projbuild
|   ├── audio_driver.c
|   ├── audio_driver.h
|   ├── audio_features.c
|   ├── audio_features.h
|   ├── audio_pin_config.h
|   ├── bathroom_rf.pkl
|   ├── i2c_manager.c
|   ├── i2c_manager.h
|   ├── idf_component.yml
|   ├── main.cpp
|   ├── microphone.c
|   ├── microphone.h
|   ├── oled.c
|   ├── oled.h
|   ├── rf_model.h
|   ├── sdcard.c
|   ├── sdcard.h
|   ├── wav_header.h
│
├── components/
│
├── assets/
│
├── docs/
│
└── README.md
```

---

# Building the Project

Clone the repository

```bash
git clone https://github.com/jagan007G/Sound_Driven_Activity_Recognition.git
```

Move into the project

```bash
cd Sound_Driven_Activity_Recognition
```

Configure ESP-IDF

```bash
idf.py menuconfig
```

Build

```bash
idf.py build
```

Flash

```bash
idf.py flash
```

Monitor

```bash
idf.py monitor
```

---

# Example Output

```
Listening...

Detected Activity:
--------------------------
Class : Flush
Confidence : 96.8%

Inference Time : 124 ms

Noise Level : Low

Saving WAV...
Logging Prediction...
```

---

# Applications

This project can be extended for:

- Smart healthcare
- Elderly monitoring
- Assisted living
- Smart homes
- Occupancy detection
- Hospital automation
- Ambient intelligence
- IoT edge devices
- TinyML research
- Environmental sound classification

---

# Advantages

- No wearable devices
- No cameras
- Privacy preserving
- Offline operation
- Low computational cost
- Real-time inference
- Embedded deployment
- Easy scalability

---

# Future Improvements

- Multi-label activity recognition
- Additional sound classes
- Adaptive threshold learning
- Automatic noise profile estimation
- Model compression
- Quantization-aware training
- OTA model updates
- Multi-microphone support
- Event timeline visualization

---

# Research Focus

This work explores the integration of:

- TinyML
- Embedded AI
- Environmental Sound Classification (ESC)
- Digital Signal Processing
- Edge Computing
- Human Activity Recognition (HAR)

to build practical intelligent sensing systems for real-world environments.

---

# Citation

If you use this work in your research, please cite the repository.

```
@misc{sound_activity_recognition,
  author = {Jagan G},
  title = {TSC Deployment on ESP32},
  year = {2026},
  publisher = {GitHub},
  url = {https://github.com/jagan007G/TSC_Deployment}
}
```

---

# License

This project is released under the MIT License.

---
