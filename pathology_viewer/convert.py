import os
os.environ['CUDA_VISIBLE_DEVICES'] = ''
import cv2
import sys
import random
from skimage.io import imread,imshow
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from keras.layers import Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Reshape, Permute, Activation, Input,add, multiply
from keras.callbacks import CSVLogger, ReduceLROnPlateau, ModelCheckpoint,EarlyStopping
from keras.layers import concatenate, core, Dropout
from keras.models import Model

from keras.layers.merge import concatenate
from keras.optimizers import Adam
from keras.optimizers import SGD
from keras.layers.core import Lambda
import keras.backend as K

seed = 2019
random.seed = seed
np.random.seed = seed
tf.seed = seed

train_ratio = 0.70
validation_ratio = 0.10
test_ratio = 0.20

def up_and_concate(down_layer, layer):
    in_channel = down_layer.get_shape().as_list()[3]
    up = UpSampling2D(size=(2, 2))(down_layer)
    my_concat = Lambda(lambda x: K.concatenate([x[0], x[1]], axis=3))
    concate = my_concat([up, layer])
    return concate
  
def rec_res_block(input_layer, out_n_filters, batch_normalization=False, kernel_size=[3, 3], stride=[1, 1],padding='same'):
  input_n_filters = input_layer.get_shape().as_list()[3]

  if out_n_filters != input_n_filters:
      skip_layer = Conv2D(out_n_filters, [1, 1], strides=stride, padding=padding)(input_layer)
  else:
      skip_layer = input_layer

  layer = skip_layer
  for j in range(2):
      for i in range(2):
          if i == 0:
              layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(layer)
              if batch_normalization:
                  layer1 = BatchNormalization()(layer1)
              layer1 = Activation('relu')(layer1)
          layer1 = Conv2D(out_n_filters, kernel_size, strides=stride, padding=padding)(add([layer1, layer]))
          if batch_normalization:
              layer1 = BatchNormalization()(layer1)
          layer1 = Activation('relu')(layer1)
      layer = layer1

  out_layer = add([layer, skip_layer])
  return out_layer

def r2_unet(img_w, img_h, n_label):
    inputs = Input((img_w, img_h,3))
    x = inputs
    depth = 4
    features = 16
    skips = []
    for i in range(depth):
        x = rec_res_block(x, features)
        skips.append(x)
        x = MaxPooling2D((2, 2))(x)

        features = features * 2

    x = rec_res_block(x, features)
    for i in reversed(range(depth)):
        features = features // 2
        #print("features")
        x = up_and_concate(x, skips[i])
        x = rec_res_block(x, features)

    conv6 = Conv2D(n_label, (1, 1), padding='same')(x)
    conv7 = core.Activation('sigmoid')(conv6)
    model = Model(inputs=inputs, outputs=conv7)
    return model


opt=tf.keras.optimizers.Adam(learning_rate=2e-4)
model=r2_unet(256, 256, 1)
model.compile(optimizer=opt, loss="binary_crossentropy", metrics=[tf.keras.metrics.MeanIoU(num_classes=2)])


model.load_weights('./model.h5')


from glob import glob

for image in glob("./0/*.jpeg"):
	img = cv2.imread(image)
	pred = model.predict(img.reshape(1,256,256,3)).reshape(256,256)
	ret,pred_mask = cv2.threshold(pred,0.4,1,cv2.THRESH_BINARY)
	pred_mask = 255*pred_mask
	cv2.imwrite(image, pred_mask)

