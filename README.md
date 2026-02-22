# PS-HK01_TRAILBLAZER_Sign-Language-Capture-and-Illustration-System-with-Sensor-Integration


# MudraVaani 🤟  
### A Hybrid Indian Sign Language (ISL) Translation System

MudraVaani is a **hybrid assistive communication system** designed to translate **Indian Sign Language (ISL)** gestures into **real-time text and speech** by combining **sensor-based wearable input** with **camera-based AI vision using the Gemini API**.

This project was developed as part of a hackathon to address real-world accessibility and inclusion challenges faced by hearing- and speech-impaired individuals.

---

## 📌 Problem Statement

Millions of people in India rely on Indian Sign Language for daily communication, yet most of society does not understand ISL.  
Existing solutions are mostly **camera-only**, which are sensitive to lighting, background, privacy concerns, and computational cost.  
Purely **sensor-based systems**, while reliable, may struggle with visually expressive gestures.

---

## 💡 Our Solution

MudraVaani uses a **hybrid approach** that integrates:

- **Sensor-based smart glove** for precise finger movement detection  
- **Camera-based gesture interpretation** powered by the **Gemini API**  
- **Machine Learning–based classification** for reliable gesture recognition  

This combination improves robustness, accuracy, and usability in real-world environments.

---

## 🧠 System Architecture

### 1. Sensor-Based Module
- Smart glove with **flex sensors** (one per finger)
- **ESP32** microcontroller for data acquisition and communication
- Captures real-time finger bend values

### 2. Camera-Based Module
- Camera input analyzed using **Gemini Vision API**
- Helps interpret hand posture and visual context
- Acts as a validation layer for complex or ambiguous gestures

### 3. Machine Learning
- **Random Forest Classifier** for sensor data
- Suitable for structured numerical inputs
- Enables fast and interpretable predictions

### 4. Output
- Recognized gesture converted to:
  - **Text**
  - **Text-to-Speech audio**

---

## 🔄 Working Flow

1. User performs an ISL gesture  
2. Flex sensors capture finger movements  
3. ESP32 sends sensor data to the system  
4. Camera feed is processed using Gemini API  
5. Sensor prediction and vision output are fused  
6. Final gesture is displayed as text and spoken aloud  

---

## 🛠️ Technologies Used

- **Hardware**
  - ESP32
  - Flex Sensors
  - Camera Module
- **Software**
  - Python
  - Machine Learning (Random Forest)
  - Gemini Vision API
- **Others**
  - Serial / WiFi communication
  - Text-to-Speech engine

---

## ✅ Key Features

- Hybrid **sensor + vision-based** recognition
- Works under varying lighting conditions
- More reliable than single-mode systems
- Privacy-aware (no video storage)
- Real-time text and speech output
- Low-cost and scalable design

---

## ⚠️ Current Limitations

- Supports a limited set of ISL gestures
- Focused mainly on **static gestures**
- Camera module requires internet access for Gemini API
- Calibration needed for different users

---

## 🚀 Future Enhancements

- Support for **dynamic and continuous gestures**
- Full sentence-level ISL translation
- Improved offline capabilities
- Regional language speech output
- Ergonomic and production-ready glove design

---

## 👥 Team

**Team TrailBlazer**  
Hackathon Project – 2026
