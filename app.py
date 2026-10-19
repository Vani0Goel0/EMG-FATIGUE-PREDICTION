import threading
import time
import joblib
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import numpy as np
from scipy.signal import butter, iirnotch, lfilter
import serial

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# --- CONFIGURATION ---
SERIAL_PORT = "COM3"
BAUD_RATE = 115200
FS = 200
WINDOW_SIZE = 200

# Load Model & Scaler
model = joblib.load("svm_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

b_notch, a_notch = iirnotch(50.0, 30.0, FS)
b_band, a_band = butter(4, [20.0 / (FS / 2), 90.0 / (FS / 2)], btype="band")

# Global calibration state so we can reset it from the web button
calibration_state = {"baseline": None, "windows": [], "reset_requested": False}


def filter_signal(raw_window):
  detrended = raw_window - np.mean(raw_window)
  notched = lfilter(b_notch, a_notch, detrended)
  return lfilter(b_band, a_band, notched)


def extract_features(sig, fs=200):
  sig = sig - np.mean(sig)
  rms = np.sqrt(np.mean(sig**2))
  mav = np.mean(np.abs(sig))
  wl = np.sum(np.abs(np.diff(sig)))

  fft_vals = np.fft.rfft(sig)
  psd = np.abs(fft_vals) ** 2
  freqs = np.fft.rfftfreq(len(sig), 1 / fs)
  total_power = np.sum(psd)

  if total_power == 0:
    mdf, mnf = 0.0, 0.0
  else:
    cumulative_psd = np.cumsum(psd)
    mdf_idx = np.where(cumulative_psd >= total_power / 2)[0][0]
    mdf = freqs[mdf_idx]
    mnf = np.sum(freqs * psd) / total_power

  return np.array([rms, mav, wl, mdf, mnf])


@socketio.on("request_reset")
def handle_reset():
  print(">>> Calibration reset requested from dashboard!")
  calibration_state["baseline"] = None
  calibration_state["windows"] = []
  calibration_state["reset_requested"] = True


def esp32_background_worker():
  print(f"Attempting connection to ESP32 on {SERIAL_PORT}...")
  try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
    time.sleep(3)
    ser.reset_input_buffer()
    socketio.emit(
        "connection_status", {"status": "Connected to ESP32", "connected": True}
    )
  except Exception as e:
    print(f"Serial connection failed: {e}. Running in simulation mode.")
    socketio.emit(
        "connection_status",
        {"status": "Simulation Mode (No Hardware)", "connected": False},
    )
    while True:
      time.sleep(1)
      sim_prob = float(np.random.randint(10, 85))
      sim_rms = float(np.round(np.random.uniform(0.5, 1.8), 2))
      sim_mdf = float(np.round(np.random.uniform(0.8, 1.2), 2))
      sim_status = "🔴 FATIGUED" if sim_prob > 50 else "🟢 NORMAL"
      socketio.emit(
          "fatigue_update",
          {
              "status": sim_status,
              "prob": sim_prob,
              "rms": sim_rms,
              "mdf": sim_mdf,
              "subtext": "Simulated Telemetry Stream",
          },
      )
    return

  buffer = []

  while True:
    try:
      # Check if user clicked reset on the web interface
      if calibration_state["reset_requested"]:
        buffer = []
        calibration_state["reset_requested"] = False
        print("Reset processed in worker thread.")

      if ser.in_waiting > 0:
        line = ser.readline().decode("utf-8", errors="ignore").strip()
        if line.isdigit():
          buffer.append(float(line))

          if len(buffer) == WINDOW_SIZE:
            raw_win = np.array(buffer)
            clean_win = filter_signal(raw_win)
            feats = extract_features(clean_win, fs=FS)
            buffer = buffer[WINDOW_SIZE // 2 :]

            current_rms = feats[0]

            if calibration_state["baseline"] is None:
              if current_rms < 100:
                socketio.emit(
                    "fatigue_update",
                    {
                        "status": "⏳ WAITING",
                        "prob": 0.0,
                        "rms": current_rms,
                        "mdf": 1.0,
                        "subtext": "Squeeze fist to capture baseline...",
                    },
                )
                continue

              calibration_state["windows"].append(feats)
              count = len(calibration_state["windows"])
              socketio.emit(
                  "fatigue_update",
                  {
                      "status": "⏳ CALIBRATING",
                      "prob": 0.0,
                      "rms": current_rms,
                      "mdf": 1.0,
                      "subtext": f"Capturing baseline window {count}/10...",
                  },
              )

              if len(calibration_state["windows"]) == 10:
                calibration_state["baseline"] = (
                    np.mean(calibration_state["windows"], axis=0) + 1e-8
                )
                print("Baseline locked successfully!")
              continue

            norm_feats = (feats / calibration_state["baseline"]).reshape(1, -1)
            rms_ratio = norm_feats[0][0]

            if rms_ratio < 0.7:
              socketio.emit(
                  "fatigue_update",
                  {
                      "status": "🟢 NORMAL",
                      "prob": 5.0,
                      "rms": rms_ratio,
                      "mdf": norm_feats[0][3],
                      "subtext": "Resting / Relaxed Arm",
                  },
              )
              continue

            scaled_feats = scaler.transform(norm_feats)
            pred = model.predict(scaled_feats)[0]
            prob = model.predict_proba(scaled_feats)[0][1] * 100

            status = "🔴 FATIGUED" if pred == 1 else "🟢 NORMAL"
            socketio.emit(
                "fatigue_update",
                {
                    "status": status,
                    "prob": float(prob),
                    "rms": float(rms_ratio),
                    "mdf": float(norm_feats[0][3]),
                    "subtext": "Active Muscle Load",
                },
            )
    except Exception as ex:
      print(f"Worker error: {ex}")
      break


@app.route("/")
def index():
  return render_template("index.html")


if __name__ == "__main__":
  t = threading.Thread(target=esp32_background_worker, daemon=True)
  t.start()
  print("Starting Flask Web Server on http://localhost:5000")
  socketio.run(app, host="0.0.0.0", port=5000, debug=False)