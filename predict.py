#!/usr/bin/env python
import numpy as np
import csv
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import sessions_plotter  # your helper functions

# --- Parameters ---
MODEL_PATH = "overlap_multiclass_reg_non_bn_trained.h5"
INPUT_CSV = "oneline.csv"  # CSV file with a single session
height, width = 1500, 1500
class_names = ["voip", "video", "file", "chat", "browsing"]

# --- Load session from CSV ---
def load_session_from_csv(csv_file):
    with open(csv_file, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            row = row[1:]  # skip session name
            try:
                length = int(row[6])
            except:
                continue

            # Load timestamps
            ts_raw = row[7:7+length]
            ts = [float(t) for t in ts_raw if t.strip() != '']

            # Load sizes
            sizes_raw = row[7+length:7+length+length]
            sizes = [int(s) for s in sizes_raw if s.strip() != '']

            # Make sure lengths match
            min_len = min(len(ts), len(sizes))
            ts = ts[:min_len]
            sizes = sizes[:min_len]

            return ts, sizes
    return [], []


# --- Load model ---
model = load_model(MODEL_PATH, compile=False)

# --- Get session data ---
ts, sizes = load_session_from_csv(INPUT_CSV)

if len(ts) == 0 or len(sizes) == 0:
    raise ValueError("No valid timestamps or sizes found in CSV!")

# --- Show spectrogram ---
sessions_plotter.session_spectogram(ts, sizes, name="Session Spectrogram")

# --- Create FlowPic ---
flowpic = sessions_plotter.session_2d_histogram(ts, sizes)

# --- Show FlowPic ---
plt.figure(figsize=(8,8))
plt.title("FlowPic")
plt.imshow(flowpic, cmap='viridis', origin='lower')
plt.colorbar()
plt.show()

# --- Prepare for CNN ---
flowpic_input = flowpic.reshape(1, height, width, 1)  # batch x H x W x C

# --- Predict ---
pred = model.predict(flowpic_input)
pred_class = np.argmax(pred, axis=1)[0]

print("Predicted class:", class_names[pred_class])
print("Class probabilities:", pred[0])
