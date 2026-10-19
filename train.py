import glob
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC


def extract_features(signal_window, fs=200):
  """Extracts time and frequency domain features after removing DC offset."""
  # Remove DC offset so 0Hz doesn't skew the frequency spectrum
  sig = signal_window - np.mean(signal_window)

  # 1. Root Mean Square (RMS)
  rms = np.sqrt(np.mean(sig**2))

  # 2. Mean Absolute Value (MAV)
  mav = np.mean(np.abs(sig))

  # 3. Waveform Length (WL) - measures signal complexity
  wl = np.sum(np.abs(np.diff(sig)))

  # Frequency domain via FFT Power Spectral Density
  fft_vals = np.fft.rfft(sig)
  psd = np.abs(fft_vals) ** 2
  freqs = np.fft.rfftfreq(len(sig), 1 / fs)

  total_power = np.sum(psd)

  if total_power == 0:
    mdf, mnf = 0.0, 0.0
  else:
    # 4. Median Frequency (MDF)
    cumulative_psd = np.cumsum(psd)
    mdf_idx = np.where(cumulative_psd >= total_power / 2)[0][0]
    mdf = freqs[mdf_idx]

    # 5. Mean Frequency (MNF) - used in the original LS-SVM paper
    mnf = np.sum(freqs * psd) / total_power

  return [rms, mav, wl, mdf, mnf]


def load_zenodo_dataset(data_dir, window_size=200, fs=200):
  """Loads Zenodo files, selects Channel 4 (index 3), normalizes per subject,

  and separates clear Normal (first 35%) vs Fatigued (last 35%) states.
  """
  file_paths = glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True)
  if not file_paths:
    raise FileNotFoundError(
        f"No .txt files found in '{data_dir}'. Check folder name."
    )

  all_features = []
  all_labels = []

  print(f'Processing {len(file_paths)} files from {data_dir}...')

  for path in file_paths:
    # Skip the Read ME file if it's inside the same folder
    if 'read' in os.path.basename(path).lower():
      continue

    try:
      df = pd.read_csv(path, header=None, sep=r'\s+', on_bad_lines='skip')
    except Exception:
      continue

    # Ensure it's a valid 8-channel recording file
    if df.empty or df.shape[1] < 4:
      continue

    # Select Channel 4 (column index 3) - Pronator Teres muscle
    # Paper proved Ch4 is the most sensitive to forearm fatigue
    signal = df.iloc[:, 3].values

    subject_windows = []
    # Use 50% overlapping windows (step = 100 samples) to get more training data
    step_size = window_size // 2
    for i in range(0, len(signal) - window_size, step_size):
      window = signal[i : i + window_size]
      feats = extract_features(window, fs=fs)
      subject_windows.append(feats)

    subject_windows = np.array(subject_windows)

    # Per-subject normalization: divide by the mean of their first 10 fresh windows
    # This makes the model work across different people and sensor gains
    baseline = np.mean(subject_windows[:10], axis=0) + 1e-8
    subject_windows_norm = subject_windows / baseline

    num_win = len(subject_windows_norm)
    early_cutoff = int(num_win * 0.35)  # First 35% of time = Normal (0)
    late_start = int(num_win * 0.65)  # Last 35% of time = Fatigued (1)

    for idx in range(num_win):
      if idx < early_cutoff:
        all_features.append(subject_windows_norm[idx])
        all_labels.append(0)
      elif idx >= late_start:
        all_features.append(subject_windows_norm[idx])
        all_labels.append(1)
      # Middle 30% transition zone is intentionally skipped for cleaner training

  return np.array(all_features), np.array(all_labels)


# --- Main Execution ---
dataset_folder = 'Data as txt Files'

try:
  X, y = load_zenodo_dataset(dataset_folder, window_size=200, fs=200)
  print(f'Extracted {len(X)} clean, normalized feature windows.')

  # Train-Test Split
  X_train, X_test, y_train, y_test = train_test_split(
      X, y, test_size=0.2, random_state=42, stratify=y
  )

  # Scale features
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train)
  X_test_scaled = scaler.transform(X_test)

  # Automatically find the best C and gamma for the RBF SVM
  print('Tuning and training RBF-SVM classifier...')
  param_grid = {
      'C': [0.1, 1, 10, 50, 100],
      'gamma': ['scale', 'auto', 0.01, 0.1, 0.5, 1.0],
  }
  grid = GridSearchCV(
      SVC(kernel='rbf', probability=True), param_grid, cv=5, n_jobs=-1
  )
  grid.fit(X_train_scaled, y_train)

  model = grid.best_estimator_
  print(f'Best SVM Parameters found: {grid.best_params_}')

  # Evaluation
  y_pred = model.predict(X_test_scaled)
  y_prob = model.predict_proba(X_test_scaled)[:, 1]

  print('\n--- High-Accuracy Forearm Dataset Performance ---')
  print(f'Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%')
  print(f'ROC-AUC Score: {roc_auc_score(y_test, y_prob):.4f}\n')
  print(
      classification_report(y_test, y_pred, target_names=['Normal', 'Fatigued'])
  )

  # Save the model and scaler so we can use them with your ESP32 next
  joblib.dump(model, 'svm_fatigue_model.pkl')
  joblib.dump(scaler, 'scaler.pkl')
  print("Saved trained model to 'svm_fatigue_model.pkl' and 'scaler.pkl'!")

except Exception as e:
  print(f'Error: {e}')