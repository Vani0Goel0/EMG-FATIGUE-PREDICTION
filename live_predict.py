import time
import joblib
import numpy as np
from scipy.signal import butter, iirnotch, lfilter
import serial

# --- CONFIGURATION ---
SERIAL_PORT = "COM3"  # Change this to your ESP32's COM port
BAUD_RATE = 115200
FS = 200  # 200 Hz sampling rate matching training data
WINDOW_SIZE = 200  # 1-second window (200 samples)

# Load your trained SVM model and scaler
model = joblib.load("svm_fatigue_model.pkl")
scaler = joblib.load("scaler.pkl")

# Setup digital filters for BioAmp EXG Pill (50Hz Notch + 20-90Hz Bandpass)
b_notch, a_notch = iirnotch(50.0, 30.0, FS)
b_band, a_band = butter(4, [20.0 / (FS / 2), 90.0 / (FS / 2)], btype="band")


def filter_signal(raw_window):
  detrended = raw_window - np.mean(raw_window)
  notched = lfilter(b_notch, a_notch, detrended)
  filtered = lfilter(b_band, a_band, notched)
  return filtered


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


# --- SAFE LIVE STREAMING CONNECTION ---
print(f"Connecting to ESP32 on {SERIAL_PORT}...")
try:
  ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=0.1)
  time.sleep(3)  # Give ESP32 time to reboot after connection opens
  ser.reset_input_buffer()
except Exception as e:
  print(
      f"Error opening {SERIAL_PORT}: {e}. Make sure Arduino IDE / Serial Monitor"
      " is closed!"
  )
  exit()

buffer = []
baseline_windows = []
baseline = None

print(
    "\n>>> PERSONALIZED CALIBRATION READY."
    "\n>>> Squeeze your fist or lift your weight firmly so we can capture your"
    " peak fresh baseline..."
)

try:
  while True:
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

          if baseline is None:
            if current_rms < 100:  # Adjust threshold if needed based on your resting noise
              print(
                  "Waiting for you to flex/squeeze... (Current RMS:"
                  f" {current_rms:.1f})"
              )
              continue

            baseline_windows.append(feats)
            print(
                f"Capturing fresh baseline window {len(baseline_windows)}/10..."
            )

            if len(baseline_windows) == 10:
              baseline = np.mean(baseline_windows, axis=0) + 1e-8
              print(
                  "\n>>> PERSONALIZED BASELINE LOCKED! Live Fatigue Monitoring"
                  " Active:\n"
              )
          else:
            norm_feats = (feats / baseline).reshape(1, -1)
            rms_ratio = norm_feats[0][0]

            if rms_ratio < 0.7:
              print(
                  f"Status: 🟢 NORMAL   | Resting / Relaxed (RMS Ratio:"
                  f" {rms_ratio:.2f})"
              )
              continue

            scaled_feats = scaler.transform(norm_feats)
            pred = model.predict(scaled_feats)[0]
            prob = model.predict_proba(scaled_feats)[0][1]

            status = "🔴 FATIGUED" if pred == 1 else "🟢 NORMAL  "
            print(
                f"Status: {status} | Fatigue Probability: {prob*100:5.1f}% |"
                f" RMS Ratio: {rms_ratio:.2f} | MDF Ratio:"
                f" {norm_feats[0][3]:.2f}"
            )

except KeyboardInterrupt:
  print("\nStopped live monitoring.")
  ser.close()