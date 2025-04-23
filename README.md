# TrafficLightRecognisiton
# 🚦 Traffic Light Detection & Car Movement Simulation

This Python project uses **OpenCV** to detect real-world traffic lights (Red, Yellow, Green) from a live webcam feed and simulates a **car's movement** based on the signal detected. A transparent car image moves across the screen and responds in real time by stopping, slowing down, or going faster depending on the traffic light color.

---

## 🔍 Features

- 🎥 **Real-time traffic light detection** using HSV color thresholds and Hough Circles.
- 🚗 **Car overlay animation** that reacts to traffic light state:
  - **Red**: STOP
  - **Yellow**: SLOW DOWN
  - **Green**: GO
- ⚡ **Live FPS counter** to monitor performance.
- 📌 Region of interest (ROI) for more efficient detection.
- ✅ Console logs for signal detection and car state.
- 🖼️ Uses a **transparent PNG image** for the car (with alpha channel).

---

## 🛠️ Installation

1. **Clone the repository**:

```bash
git clone https://github.com/5hreyv/TrafficLightRecognisiton.git
cd traffic-light-detection
