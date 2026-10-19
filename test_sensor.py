import time
import serial

SERIAL_PORT = "COM3"  # Change if your port is different
BAUD_RATE = 115200

print(f"Attempting to read from {SERIAL_PORT}...")
try:
  ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
  time.sleep(2)
  print("Connected! Listening for raw sensor values (Press Ctrl+C to stop):")

  while True:
    if ser.in_waiting > 0:
      line = ser.readline().decode("utf-8", errors="ignore").strip()
      if line:
        print(f"Raw ADC Value: {line}")
    else:
        # If nothing is printing here, the ESP32 isn't sending data or port is wrong
        pass

except Exception as e:
  print(f"Connection failed: {e}")