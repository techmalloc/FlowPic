import os
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import top_k_categorical_accuracy
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy.ma as ma
import pylab as pl
from mpl_toolkits.axes_grid1 import make_axes_locatable

# Set data format to channels_last
K.set_image_data_format('channels_last')
print("Keras backend:", K.backend(), "Image data format:", K.image_data_format())

# Define Parameters
batch_size = 128
epochs = 40
num_classes = 5
class_names = ["voip", "video", "file transfer", "chat", "browsing"]
height, width = 1500, 1500
input_shape = (height, width, 1)  # Channels_last (NHWC)
MODEL_NAME = "overlap_multiclass_reg_non_bn"
PATH_PREFIX = "datasets/"

# Load Data
dataset_file = os.path.join(PATH_PREFIX, "file_vs_all_reg.npz")
dataset = np.load(dataset_file)
x_train = dataset['x_train']
y_train_true = dataset['y_train']
x_val = dataset['x_val']
y_val_true = dataset['y_val']

# Transpose to channels_last if needed
if x_train.shape[1] == 1:  # If channels_first
    x_train = x_train.transpose(0, 2, 3, 1)  # (N, 1, H, W) -> (N, H, W, 1)
    x_val = x_val.transpose(0, 2, 3, 1)
x_train = x_train.astype(np.float32)
x_val = x_val.astype(np.float32)
print("Loaded dataset shapes:")
print("x_train:", x_train.shape, "y_train_true:", y_train_true.shape)
print("x_val:", x_val.shape, "y_val_true:", y_val_true.shape)

# Dataset stats
print("x_train stats - min:", np.min(x_train), "max:", np.max(x_train), "mean:", np.mean(x_train), "std:", np.std(x_train))
print("x_train non-zero count:", np.sum(x_train > 0), "density:", np.sum(x_train > 0) / (x_train.shape[0] * height * width) * 100, "%")
for i in range(10):
    print(f"Sample {i} non-zero count:", np.sum(x_train[i] > 0), "density:", np.sum(x_train[i] > 0) / (height * width) * 100, "%")

# Shuffle Data
def shuffle_data(x, y):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print("Shuffled data shapes:", x.shape, y.shape)
    return x, y
x_train, y_train_true = shuffle_data(x_train, y_train_true)
print("First 10 labels after shuffle:", y_train_true[:10])

# One-Hot Encode
y_train = to_categorical(y_train_true, num_classes)
y_val = to_categorical(y_val_true, num_classes)
print("One-hot encoded shapes:", y_train.shape, y_val.shape)
print("First 10 y_train:\n", y_train[:10])

# Custom Metrics
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

# Model Definition
from tensorflow.keras.layers import Input

# Model Definition
from tensorflow.keras.layers import Input

# Model Definition
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
model.compile(loss='categorical_crossentropy', optimizer='adam',
              metrics=['accuracy', top_2_categorical_accuracy, f1_score, precision, recall])
model.summary()



# Visualization Helpers
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None, bar=True):
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap, origin='lower')
    if bar:
        plt.colorbar(im, ax=ax)

def mosaic_imshow(imgs, nrows, ncols, cmap=cm.binary, border=1, layer_name="layer"):
    h, w, nimgs = imgs.shape
    mosaic = np.ones((nrows*h + (nrows-1)*border,
                      ncols*w + (ncols-1)*border)) * np.nan
    for i in range(nimgs):
        row = i // ncols
        col = i % ncols
        mosaic[row*(h+border):row*(h+border)+h,
               col*(w+border):col*(w+border)+w] = imgs[:,:,i]
    plt.figure(figsize=(3*ncols, 3*nrows))
    plt.title(layer_name)
    plt.imshow(mosaic, cmap=cmap)
    plt.colorbar()
    plt.savefig(f"{MODEL_NAME}_mosaic_imshow_{layer_name}.png", bbox_inches='tight', pad_inches=1)
    plt.show()

def plotNNFilter(data, nrows, ncols, layer_name, cmap="gray", bar=True):
    plt.figure(figsize=(3*ncols, 3*nrows))
    for i in range(nrows*ncols):
        plt.subplot(nrows, ncols, i+1)
        plt.imshow(data[:, :, i], interpolation="nearest", cmap=cmap)
        plt.xticks([]); plt.yticks([])
        plt.gca().invert_yaxis()
    plt.subplots_adjust(wspace=0.025, hspace=0.05)
    plt.title(layer_name)
    plt.savefig(f"{MODEL_NAME}_plotNNFilter_{layer_name}.png", bbox_inches='tight', pad_inches=1)
    plt.show()

def plotNNFilter2(data, nrows, ncols, layer_name, cmap="gray", bar=True):
    fig, axes = plt.subplots(nrows, ncols, figsize=(5*ncols, 4*nrows))
    ims = []  # store imshow objects
    for i, ax in enumerate(axes.flat):
        im = ax.imshow(data[:, :, i], interpolation="nearest", cmap=cmap)
        ims.append(im)
        ax.set_xticks([]); ax.set_yticks([])
        ax.invert_yaxis()
    plt.subplots_adjust(wspace=0.025, hspace=0.05)
    plt.suptitle(layer_name)
    if bar and ims:
        fig.colorbar(ims[0], ax=axes.ravel().tolist())
    plt.savefig(f"{MODEL_NAME}_plotNNFilter2_{layer_name}.png", bbox_inches='tight', pad_inches=1)
    plt.show()

def get_layer_output(layer, input_img, layer_name="layer"):
    print("Model layers:", [l.name for l in model.layers])
    intermediate_model = tf.keras.Model(inputs=model.inputs, outputs=layer.output)
    C = intermediate_model.predict(input_img, verbose=0)
    C = np.squeeze(C)
    print(f"{layer_name} output shape: {C.shape}")
    return C


# Initialize model input
model.predict(np.zeros((1, 1500, 1500, 1), dtype=np.float32), verbose=0)

# Debug Sample
for i in list(range(35, 40)) + list(np.random.choice(x_train.shape[0], 5)):
    input_sample = x_train[i:i+1].astype(np.float32)
    label = y_train_true[i]
    squeezed_sample = np.squeeze(input_sample)
    print(f"\nInput sample {i} stats - min: {np.min(squeezed_sample)}, max: {np.max(squeezed_sample)}, mean: {np.mean(squeezed_sample)}, std: {np.std(squeezed_sample)}")
    print(f"Number of non-zero elements: {np.sum(squeezed_sample > 0)} (density: {np.sum(squeezed_sample > 0) / (height * width) * 100:.4f}%)")

    # Full image with normalized non-zero values
    norm_sample = squeezed_sample / (np.max(squeezed_sample) + 1e-10) if np.max(squeezed_sample) > 0 else squeezed_sample
    plt.figure(figsize=(12, 12))
    plt.title(f"Input Sample {i} (Full) - Label: {label}")
    plt.imshow(norm_sample, cmap='viridis')
    plt.colorbar()
    plt.savefig(f"{MODEL_NAME}_input_sample_full_{i}_{label}.png", bbox_inches='tight', pad_inches=1)
    plt.show()

    # Sparse matrix spy plot
    plt.figure(figsize=(12, 12))
    plt.title(f"Input Sample {i} (Spy) - Label: {label}")
    plt.spy(squeezed_sample, markersize=2.0, color='blue')
    plt.savefig(f"{MODEL_NAME}_input_sample_spy_{i}_{label}.png", bbox_inches='tight', pad_inches=1)
    plt.show()

    # Zoomed-in version
    zoom_size = 500
    plt.figure(figsize=(12, 12))
    plt.title(f"Input Sample {i} (Zoomed 0:{zoom_size}) - Label: {label}")
    plt.imshow(norm_sample[0:zoom_size, 0:zoom_size], cmap='viridis')
    plt.colorbar()
    plt.savefig(f"{MODEL_NAME}_input_sample_zoomed_{i}_{label}.png", bbox_inches='tight', pad_inches=1)
    plt.show()

    # Layer Outputs (only if sample is non-empty)
    if np.sum(squeezed_sample) > 0:
        conv1_layer = model.get_layer("conv1")
        conv2_layer = model.get_layer("conv2")
        C1 = get_layer_output(conv1_layer, input_sample, layer_name="conv1")
        C2 = get_layer_output(conv2_layer, input_sample, layer_name="conv2")
        mosaic_imshow(C1, 2, 5, cmap=cm.binary, border=2, layer_name=f"conv1_label_{label}")
        plotNNFilter(C1, 2, 5, layer_name=f"conv1_label_{label}")
        plotNNFilter2(C1, 2, 5, layer_name=f"conv1_label_{label}")
        mosaic_imshow(C2, 4, 5, cmap=cm.binary, border=2, layer_name=f"conv2_label_{label}")
        plotNNFilter(C2, 4, 5, layer_name=f"conv2_label_{label}")
        plotNNFilter2(C2, 4, 5, layer_name=f"conv2_label_{label}")