import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import top_k_categorical_accuracy
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# --- Setup ---
K.set_image_data_format('channels_last')
print("Keras backend:", K.backend(), "Image data format:", K.image_data_format())

# --- Parameters ---
batch_size = 128       # smaller batch size to avoid memory issues
epochs = 40           # start small; you can increase later
num_classes = 5
class_names = ["voip", "video", "file", "chat", "browsing"]
height, width = 1500, 1500
input_shape = (height, width, 1)
MODEL_NAME = "overlap_multiclass_reg_non_bn"
PATH_PREFIX = "datasets/"

# --- Load Dataset ---
dataset_file = os.path.join(PATH_PREFIX, "file_vs_all_reg.npz")
dataset = np.load(dataset_file)
x_train = dataset['x_train']
y_train_true = dataset['y_train']
x_val = dataset['x_val']
y_val_true = dataset['y_val']

# --- Ensure correct format ---
if x_train.shape[1] == 1:  # channels_first
    x_train = x_train.transpose(0, 2, 3, 1)
    x_val = x_val.transpose(0, 2, 3, 1)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)

print("Loaded dataset shapes:")
print("x_train:", x_train.shape, "y_train_true:", y_train_true.shape)
print("x_val:", x_val.shape, "y_val_true:", y_val_true.shape)

# --- Shuffle data ---
def shuffle_data(x, y):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    return x[s], y[s]

x_train, y_train_true = shuffle_data(x_train, y_train_true)

# --- One-hot encode labels ---
y_train = to_categorical(y_train_true, num_classes)
y_val = to_categorical(y_val_true, num_classes)

# --- Custom Metrics ---
def precision(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    return true_positives / (predicted_positives + K.epsilon())

def recall(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    return true_positives / (possible_positives + K.epsilon())

def f1_score(y_true, y_pred):
    prec = precision(y_true, y_pred)
    rec = recall(y_true, y_pred)
    return 2 * ((prec * rec) / (prec + rec + K.epsilon()))

def top_2_categorical_accuracy(y_true, y_pred):
    return top_k_categorical_accuracy(y_true, y_pred, k=2)

# --- Model Definition ---
model = Sequential([
    Input(shape=input_shape, name='input_layer'),
    Conv2D(10, kernel_size=(10, 10), strides=5, padding="same", activation='relu', name="conv1"),
    MaxPooling2D(pool_size=(2, 2)),
    Conv2D(20, kernel_size=(10, 10), strides=5, padding="same", activation='relu', name="conv2"),
    Dropout(0.25),
    MaxPooling2D(pool_size=(2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy', top_2_categorical_accuracy, f1_score, precision, recall]
)
model.summary()

# --- TRAINING ---
print("\n🚀 Starting model training...\n")
history = model.fit(
    x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    validation_data=(x_val, y_val),
    verbose=1
)

# --- EVALUATION ---
print("\n📊 Evaluating model on validation set...\n")
val_loss, val_acc, val_top2, val_f1, val_prec, val_rec = model.evaluate(x_val, y_val, verbose=1)

print("\n✅ Validation Results:")
print(f"Loss: {val_loss:.4f}")
print(f"Accuracy: {val_acc:.4f}")
print(f"Top-2 Accuracy: {val_top2:.4f}")
print(f"F1 Score: {val_f1:.4f}")
print(f"Precision: {val_prec:.4f}")
print(f"Recall: {val_rec:.4f}")

# --- PREDICT ON VALIDATION SET ---
y_pred_probs = model.predict(x_val, verbose=1)
y_pred_classes = np.argmax(y_pred_probs, axis=1)
y_true_classes = np.argmax(y_val, axis=1)

from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
# --- CLASSIFICATION REPORT ---
print("Unique labels in y_val_true:", np.unique(y_val_true))
print("\n📈 Classification Report (per class):\n")
print(classification_report(y_true_classes, y_pred_classes, target_names=class_names, digits=4))

# --- CONFUSION MATRIX ---
cm = confusion_matrix(y_true_classes, y_pred_classes)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.savefig(f"{MODEL_NAME}_confusion_matrix.png", bbox_inches='tight')
plt.show()

# --- SAVE MODEL ---
model.save(f"{MODEL_NAME}_trained.h5")
print(f"\n💾 Model saved as {MODEL_NAME}_trained.h5")

# --- PLOT TRAINING RESULTS ---
plt.figure(figsize=(10, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.savefig(f"{MODEL_NAME}_training_accuracy.png", bbox_inches='tight')
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(f"{MODEL_NAME}_training_loss.png", bbox_inches='tight')
plt.show()
