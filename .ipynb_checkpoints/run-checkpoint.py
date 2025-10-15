# Import Libraries
import numpy as np
import random
import os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy.ma as ma

# TensorFlow / Keras imports (modern)
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Activation
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras import backend as K
from tensorflow.keras.metrics import top_k_categorical_accuracy

# Set Keras backend image format
K.set_image_data_format('channels_first')
print(K.backend(), K.image_data_format())

# Define Parameters
batch_size = 128
samples_per_epoch = 10
num_classes = 5
epochs = 40
class_names = ["voip", "video", "file transfer", "chat", "browsing"]

# Input histogram dimensions
height, width = 1500, 1500
input_shape = (1, height, width)
MODEL_NAME = "overlap_multiclass_reg_non_bn"
PATH_PREFIX = "../datasets/"

# Load Train and Validation Data
x_train = np.load(PATH_PREFIX + "reg_x_train.npy")
y_train_true = np.load(PATH_PREFIX + "reg_y_train.npy")
x_val = np.load(PATH_PREFIX + "reg_x_val.npy")
y_val_true = np.load(PATH_PREFIX + "reg_y_val.npy")

print(x_train.shape, y_train_true.shape)
print(x_val.shape, y_val_true.shape)

# Shuffle Data
def shuffle_data(x, y):
    s = np.arange(x.shape[0])
    np.random.shuffle(s)
    x = x[s]
    y = y[s]
    print(x.shape, y.shape)
    return x, y

x_train, y_train_true = shuffle_data(x_train, y_train_true)

# Convert class vectors to categorical matrices
y_train = to_categorical(y_train_true, num_classes)
y_val = to_categorical(y_val_true, num_classes)
print(y_train.shape, y_val.shape)

# Define metrics
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

# Define and Compile model
model = Sequential()
model.add(Conv2D(10, kernel_size=(10, 10), strides=5, padding="same", input_shape=input_shape))
convout1 = Activation('relu')
model.add(convout1)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(20, (10, 10), strides=5, padding="same"))
convout2 = Activation('relu')
model.add(convout2)
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy', top_2_categorical_accuracy, f1_score, precision, recall])

print(model.summary())

# Visualization functions
def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None, bar=True):
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()
    divider = make_axes_locatable(ax)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap, origin='lower')
    if bar:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        plt.colorbar(im, cax=cax)

def make_mosaic(imgs, nrows, ncols, border=1):
    nimgs = imgs.shape[2]
    imshape = imgs.shape[0:2]
    mosaic = ma.masked_all((nrows*imshape[0] + (nrows-1)*border,
                            ncols*imshape[1] + (ncols-1)*border), dtype=np.float32)
    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i/ncols))
        col = i % ncols
        mosaic[row*paddedh:row*paddedh+imshape[0],
               col*paddedw:col*paddedw+imshape[1]] = imgs[:,:,i]
    return mosaic

def mosaic_imshow(imgs, nrows, ncols, cmap=None, border=1, layer_name="convout"):
    plt.figure(figsize=(3*ncols, 3*nrows))
    nice_imshow(plt.gca(), make_mosaic(imgs, nrows, ncols, border=border), cmap=cmap)
    plt.savefig(MODEL_NAME + "_mosaic_imshow_" + layer_name, bbox_inches='tight', pad_inches=1)
    plt.show()

# Example: visualize first layer output
i = 35
X = x_train[i][0]
plt.figure(figsize=(15, 15))
plt.title('input')
nice_imshow(plt.gca(), np.squeeze(X), vmin=0, vmax=1, cmap=cm.binary)
plt.savefig(MODEL_NAME + "_input", bbox_inches='tight', pad_inches=1)
plt.show()
