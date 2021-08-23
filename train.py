import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# import tensorflow as tf
# physical_devices = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_devices[0], True)
# from glob import glob
# from skimage import data, io, filters
# import skimage.transform as trans
# import random as r
import numpy as np
from unet_model import unet
import matplotlib.pyplot as plt

img_size = 120

# Load the dataset as npy format.
print("Loading dataset.")
train_X = np.load('x_{}.npy'.format(img_size))
train_Y = np.load('y_{}.npy'.format(img_size))
print("Dataset loaded")
print(train_X.shape)
print(train_Y.shape)

def seg_to_array(path, end, label):
    # get location
    files = glob(path + end, recursive=True)

    img_list = []

    r.seed(42)
    r.shuffle(files)

    for file in files:
        img = io.imread(file, plugin="simpleitk")

        # all tumor
        if label == 1:
            img[img != 0] = 1

        # Non-enhancing tumor
        if label == 2:
            img[img != 1] = 0

        # Without Edema
        if label == 3:
            img[img == 2] = 0
            img[img != 0] = 1

        # Enhancing tumor
        if label == 4:
            img[img != 4] = 0
            img[img == 4] = 1

        img.astype("float32")

        for slice in range(60, 130):
            img_s = img[slice, :, :]
            img_s = np.expand_dims(img_s, axis=0)
            img_list.append(img_s)
    return np.array(img_list)


def to_array(path, end):
    # get location
    files = glob(path + end, recursive=True)

    img_list = []

    r.seed(42)
    r.shuffle(files)

    for file in files:
        img = io.imread(file, plugin="simpleitk")
        # standardization
        img = ((img - img.mean()) / (img.std()))
        img.astype("float32")

        for slice in range(60, 130):
            img_s = img[slice, :, :]
            img_s = np.expand_dims(img_s, axis=0)
            img_list.append(img_s)
    return np.array(img_list)

path = r"C:\Users\MACHENIKE\Desktop\MDS\Research Project\Data\GzipData\\"
train_X = to_array(path=path, end="**/*flair.nii.gz")
train_Y = seg_to_array(path=path, end="**/*seg.nii.gz", label=1)

model = unet()

# history = model.fit(X_train, seg, validation_split=0.25, batch_size=5, epochs= 10, shuffle=True,  verbose=1,)
history = model.fit(train_X, train_Y, validation_split=0.25, batch_size=5, epochs= 1, shuffle=True,  verbose=1,)

# Plot training & validation accuracy values

plt.plot(history.history['dice_coef'])
plt.plot(history.history['val_dice_coef'])
plt.title('model dice_coef')
plt.ylabel('dice_coef')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('dice_score.png')

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'validation'], loc='upper left')
plt.savefig('dice_loss.png')


# # save the model
# model.save_weights('C:/Users/MACHENIKE/Desktop/MDS/Research Project/Data/test_dice_weights_flair&t2_{}_{}_{}images.h5'.format(img_size,10,10))

