# Real-Time sEMG Muscle Fatigue Monitoring System

An end-to-end, full-stack application for monitoring muscle fatigue in real time using surface Electromyography (sEMG). Built with an **ESP32**, **BioAmp EXG Pill**, a custom **RBF-SVM machine learning pipeline**, a **Flask-SocketIO backend**, and a **Tailwind CSS + Chart.js web dashboard**.

---

## 🚀 Features

* **Real-Time Data Acquisition:** Streams sEMG signals from an ESP32 microcontroller via Serial communication (115200 baud).
* **Advanced Signal Processing:** Applies a 50Hz IIR notch filter (mains hum removal) and a 20–90Hz Butterworth bandpass filter.
* **Feature Extraction:** Computes Root Mean Square (RMS), Mean Absolute Value (MAV), Waveform Length (WL), Median Frequency (MDF), and Mean Frequency (MNF).
* **Personalized Dynamic Calibration:** Automatically locks onto your peak fresh baseline muscle contraction upon squeezing, rather than using generic timers.
* **Smart Safety Gate:** Bypasses false fatigue alarms when the arm is resting or unloaded (`RMS Ratio < 0.7`).
* **Interactive Web Dashboard:** Live telemetry visualization featuring real-time charts, probability meters, and a remote calibration reset button.

---

## 🛠️ Tech Stack

* **Hardware:** ESP32 NodeMCU, BioAmp EXG Pill (Channel 4 / GPIO 34)
* **Backend:** Python, Flask, Flask-SocketIO, NumPy, SciPy, Scikit-Learn, PySerial, Joblib
* **Frontend:** HTML5, Tailwind CSS, Chart.js, Socket.IO

---

## 📂 Project Structure

```text
EMG-FATIGUE-PREDICTION/
│
├── templates/
│   └── index.html          # Web dashboard interface
├── app.py                  # Flask orchestrator & backend WebSocket server
├── train.py                # Model training script
├── scaler.pkl              # Fitted MinMaxScaler object
├── svm_fatigue_model.pkl   # Trained RBF-SVM Classifier model
├── requirements.txt        # Python dependencies
└── .gitignore              # Ignored files (venv, datasets, cache)