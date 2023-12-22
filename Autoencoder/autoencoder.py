import os
import re
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress tensorflow alerts
import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D
from tensorflow.keras.models import Model
from tensorflow.keras import layers
from tensorflow.keras import backend as K
from tensorflow.keras.callbacks import TensorBoard
import random
import numpy as np

class Autoencoder(Model):
	def __init__(self, img_shape):
		# initialize the `Model` super class  
        super(Autoencoder, self).__init__()

        # shape of the images
        self.img_shape = img_shape
    def set_convolutional_network(self, ty=0):