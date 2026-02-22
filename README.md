# PS-HK01_TRAILBLAZER_Sign-Language-Capture-and-Illustration-System-with-Sensor-Integration



# MudraVaani 🤟  
### A Sensor-Based Indian Sign Language (ISL) Translation System

MudraVaani is an **assistive communication system** designed to translate **Indian Sign Language (ISL)** gestures into **real-time text and speech** using a **sensor-based smart glove**.

This project was developed as part of a hackathon to address accessibility and inclusion challenges faced by hearing- and speech-impaired individuals.

---

## 📌 Problem Statement

In India, millions of people rely on **Indian Sign Language** for daily communication.  
However, most of society does not understand ISL, leading to communication barriers in education, healthcare, workplaces, and public services.

Most existing solutions depend on **camera-based systems**, which are affected by:
- Lighting conditions  
- Background noise  
- Privacy concerns  
- High computational requirements  

---

## 💡 Our Solution

MudraVaani provides a **wearable, sensor-based solution** that translates ISL gestures into **text and speech** without using any camera.

By using **flex sensors and a microcontroller**, the system captures precise finger movements and processes them using **machine learning** to recognize gestures reliably.

---

## 🧠 System Architecture

### 1. Sensor-Based Smart Glove
- Five **flex sensors** (one per finger)
- Measures finger bend and hand posture
- Designed for real-time gesture capture

### 2. Microcontroller Unit
- **ESP32** microcontroller
- Reads sensor values
- Transmits data for processing

### 3. Machine Learning Module
- **Random Forest Classifier**
- Trained on numerical sensor data
- Suitable for fast and accurate prediction

### 4. Output Module
- Recognized gesture converted into:
  - **Text output**
  - **Text-to-Speech audio**

---

## 🔄 Working Flow

1. User performs an ISL gesture using the glove  
2. Flex sensors capture finger movement data  
3. ESP32 sends sensor values to the system  
4. Machine learning model classifies the gesture  
5. Recognized gesture is displayed as text  
6. Text is converted into speech output  

---

## 🛠️ Technologies Used

### Hardware
- ESP32
- Flex Sensors
- Smart Glove Setup

### Software
- Python
- Machine Learning (Random Forest)
- Serial / WiFi Communication
- Text-to-Speech Engine

---

## ✅ Key Features

- Fully **sensor-based** (no camera required)
- Works in any lighting condition
- Privacy-preserving design
- Real-time text and speech output
- Low-cost and wearable solution
- Suitable for classrooms, hospitals, and public spaces

---

## ⚠️ Current Limitations

- Supports a limited set of ISL gestures
- Focused mainly on **static gestures**
- Requires calibration for different users
- Continuous sentence-level translation is not yet supported

---

## 🚀 Future Enhancements

- Support for **dynamic gestures**
- Continuous sentence formation
- Improved accuracy with larger datasets
- Regional language speech output
- Ergonomic and production-ready glove design

---

## 👥 Team

**Team TrailBlazer**  
Hackathon Project – 2026
