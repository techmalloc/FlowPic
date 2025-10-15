import numpy as np
import os
import glob

PATH_PREFIX = "datasets/"
class_names = ["voip", "video", "file", "chat", "browsing"]

# Collect all files matching the pattern *_vs_all_*.npy
x_list = []
y_list = []

for idx, class_name in enumerate(class_names):
    pattern = os.path.join(PATH_PREFIX, f"{class_name}_vs_all_*.npy")
    for file_path in glob.glob(pattern):
        print("Loading:", file_path)
        data = np.load(file_path)
        labels = np.full(len(data), idx)  # Assign the class index
        x_list.append(data)
        y_list.append(labels)

x_all = np.concatenate(x_list)
y_all = np.concatenate(y_list)

print("Combined dataset shapes:")
print("x_all:", x_all.shape)
print("y_all:", y_all.shape)
