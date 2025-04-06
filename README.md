# ⚕️ WSNet: A Lightweight Dual-Tree Wavelet CNN for Medical Image Analysis on IoT Devices

> 🚀 Optimized for Raspberry Pi 4 & Jetson Nano | 🏥 Supports Kidney, Colon, Malaria Datasets  
> 📉 83× Fewer Parameters | ⚡ 24× Lower FLOPs | 📦 Real-Time Performance (10 FPS)

## 🧠 Abstract

Nowadays, the Internet of Things (IoT) plays a crucial role in medical applications, offering valuable support in both rural and urban areas where access to medical professionals may be limited. Recent deep learning-based techniques for medical image analysis have shown great promise, especially in early disease detection. However, deployment on low-resource devices remains a challenge.

To address this, we propose **WSNet**, a lightweight deep learning model designed for efficient medical image analysis on IoT platforms. WSNet enhances **ShuffleNetv2** using **dual-tree wavelet transforms**, enabling the model to focus on key image features while drastically reducing computational complexity.

- 🔬 **Parameter Reduction:** From 1.4M → ~17K (83.33× smaller)
- ⚙️ **FLOPs Reduction:** From 41M → ~1.7M (24.12× lower)
- ⏱️ **Real-time Speed:** Achieves 10 FPS on devices like **Raspberry Pi 4** and **Jetson Nano**

Extensive experiments on datasets such as **kidney stones**, **colon cancer**, and **malaria cells** show that WSNet outperforms existing lightweight models while maintaining real-time inference capability.

---

## 📊 Performance Summary

| Dataset       | Accuracy (%) | Params | FLOPs | Inference Speed |
|---------------|--------------|--------|-------|-----------------|
| Kidney Stones | 99.44        | ~17K   | ~1.7M | 10 FPS (Raspberry Pi 4) |
| Colon Cancer  | 99.30        | ~17K   | ~1.7M | 10 FPS (Jetson Nano) |
| Malaria Cells | 96.08        | ~17K   | ~1.7M | 10 FPS (Raspberry Pi 4) |
