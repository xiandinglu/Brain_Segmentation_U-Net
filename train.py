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
from model import unet_model
import matplotlib.pyplot as plt

# Load the dataset as npy format.
print("Loading dataset.")
train_X = np.load('x_{}.npy'.format(img_size))
train_Y = np.load('y_{}.npy'.format(img_size))
print("Dataset loaded")
print(train_X.shape)
print(train_Y.shape)

model = unet_model()

# history = model.fit(X_train, seg, validation_split=0.25, batch_size=5, epochs= 10, shuffle=True,  verbose=1,)
history = model.fit(train_X, train_Y, validation_split=0.25, batch_size=5, epochs= 2, shuffle=True,  verbose=1,)

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
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('dice_loss.png')


# # save the model
# model.save_weights('C:/Users/MACHENIKE/Desktop/MDS/Research Project/Data/test_dice_weights_flair&t2_{}_{}_{}images.h5'.format(img_size,10,10))

