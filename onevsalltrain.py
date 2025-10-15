import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import top_k_categorical_accuracy
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# --- Setup ---
K.set_image_data_format('channels_last')
print("Keras backend:", K.backend(), "Image data format:", K.image_data_format())

# --- Parameters ---
batch_size = 64
epochs = 10
height, width = 1500, 1500
input_shape = (height, width, 1)
PATH_PREFIX = "datasets/"

# 🔹 These are your known traffic classes
class_names = ["browsing", "chat", "file_transfer", "video", "voip"]
num_classes = len(class_names)

# --- Custom Metrics ---
def precision(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return tp / (pp + K.epsilon())

def recall(y_true, y_pred):
    tp = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    pp = K.sum(K.round(K.clip(y_true, 0, 1)))
    return tp / (pp + K.epsilon())

def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * ((p * r) / (p + r + K.epsilon()))

# --- Model Definition ---
def build_model():
    model = Sequential([
        Input(shape=input_shape, name='input_layer'),
        Conv2D(10, kernel_size=(10, 10), strides=5, padding="same", activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(20, kernel_size=(10, 10), strides=5, padding="same", activation='relu'),
        Dropout(0.25),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        loss='categorical_crossentropy',
        optimizer='adam',
        metrics=['accuracy', top_k_categorical_accuracy, f1_score, precision, recall]
    )
    return model

# --- Combine all datasets ---
npz_files = [f for f in os.listdir(PATH_PREFIX) if f.endswith('.npz')]
print(f"Found {len(npz_files)} datasets:", npz_files)

X_list = []
y_list = []

for idx, dataset_file in enumerate(npz_files):
    dataset_path = os.path.join(PATH_PREFIX, dataset_file)
    print(f"Loading {dataset_file} (label={idx})")

    dataset = np.load(dataset_path)
    x_train = dataset['x_train']
    y_train = np.full(x_train.shape[0], idx)  # assign numeric label per dataset
    X_list.append(x_train)
    y_list.append(y_train)

# --- Merge into single arrays ---
X_all = np.concatenate(X_list, axis=0)
y_all = np.concatenate(y_list, axis=0)

print("Combined dataset:", X_all.shape, y_all.shape)

# --- Split train/val ---
x_train, x_val, y_train_true, y_val_true = train_test_split(X_all, y_all, test_size=0.2, random_state=42)

# --- Ensure correct shape ---
if x_train.shape[1] == 1:  # channels_first
    x_train = x_train.transpose(0, 2, 3, 1)
    x_val = x_val.transpose(0, 2, 3, 1)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

# --- One-hot encode labels ---
y_train = to_categorical(y_train_true, num_classes)
y_val = to_categorical(y_val_true, num_classes)

# --- Build model ---
model = build_model()
model.summary()

# --- Train ---
print("\n🚀 Training on all combined datasets...\n")
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1
)

# --- Evaluate ---
val_metrics = model.evaluate(x_val, y_val, verbose=1)
print(f"\n✅ Final validation results:\n{dict(zip(model.metrics_names, val_metrics))}")

# --- Save model ---
model.save("multiclass_combined_model.h5")
print("\n💾 Saved as multiclass_combined_model.h5")

# --- Plot accuracy/loss ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.legend(); plt.grid(True); plt.title("Training Accuracy")
plt.savefig("combined_accuracy.png")

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.legend(); plt.grid(True); plt.title("Training Loss")
plt.savefig("combined_loss.png")
plt.show()
