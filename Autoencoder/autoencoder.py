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

	def set_convolutional_network(self):
		# Encoder
		self.encoder = tf.keras.Sequential([
			Input(shape=self.img_shape),
			Conv2D(32, (3, 3), padding='same'),
			layers.BatchNormalization(),  # Add batch normalization
			layers.LeakyReLU(alpha=0.01),
			MaxPooling2D((2, 2), padding='same'),

			Conv2D(64, (3, 3), padding='same'),
			layers.BatchNormalization(),  # Add batch normalization
			layers.LeakyReLU(alpha=0.01),
			MaxPooling2D((2, 2), padding='same'),

			Conv2D(128, (3, 3), padding='same'),
			layers.BatchNormalization(),  # Add batch normalization
			layers.LeakyReLU(alpha=0.01),
			MaxPooling2D((2, 2), padding='same'),
		])

		# Decoder
		self.decoder = tf.keras.Sequential([
			Conv2D(128, (3, 3), padding='same'),
        	layers.BatchNormalization(),  # Add batch normalization
			layers.LeakyReLU(alpha=0.01),
			UpSampling2D((2, 2)),

			Conv2D(64, (3, 3), padding='same'),
			layers.BatchNormalization(),  # Add batch normalization
			layers.LeakyReLU(alpha=0.01),
			UpSampling2D((2, 2)),

			Conv2D(32, (3, 3), padding='same'),
			layers.BatchNormalization(),  # Add batch normalization
			layers.LeakyReLU(alpha=0.01),
			UpSampling2D((2, 2)),

			Conv2D(1, (3, 3), activation='sigmoid', padding='same')
		])

		return

	def load_image(self, img_path, resize=False):
		"""Loads an image from `img_path` in grayscale and resize with padding if `resize=True`.

		Args:
			img_path (str): path to the image to be loaded
			resize (bool, optional): Set to true if have to resize the image. Defaults to False.

		Returns:
			(tf.dataset, tf.dataset, string): tensor flow datasets and the original path for the image to recover the category.
			Note: a tuple because the same dataset is used in as input (left) and output (right) in the network for compression
		"""
		# read a grayscale image 
		img = tf.image.decode_image(tf.io.read_file(img_path), channels=1, dtype=tf.float32) / 255.0

		# resize with padding 
		if resize:
			img = tf.image.resize_with_pad(img, self.img_shape[0], self.img_shape[1], 
											method=tf.image.ResizeMethod.BILINEAR, antialias=False)
		return img, img_path

	def create_dataset(self, files, resize = False, batch_size = 8, shuffle = True):
		""" Define a dataset to be used in tensorflow model training.
		Obs: Can train the model without setting datasets, but this allows TF use resources more smartly

		Args:
			train_folders (list): list with paths to the folders with training data
			test_folder (list): list with paths to the folders with test data
			resize (bool, optional): Set to True if have to resize images to a common size. Defaults to False.
				Obs: if False, make sure all images in each folder have the same size
			n (int, optional): Number of images in the folder to use in the model. If None, default to all images in the folder.
			batch_size (int, optional): Number of images used in each tensorflow pass. Defaults to 8.
		"""
		# Convert list of files to tensorflow datasets
		dataset = tf.data.Dataset.from_tensor_slices(files)

		# map the load_image function to create the train and test tensorflow datasets
		dataset = dataset.map(lambda x: self.load_image(x, resize=resize), 
                            num_parallel_calls=tf.data.experimental.AUTOTUNE)        

		# set batch size (how many images will be loaded per time; this controls RAM consumption)
		dataset = dataset.batch(batch_size)

		# prefetch
		dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
     
		# shuffle the dataset between epochs
		if shuffle:
			dataset = dataset.shuffle(buffer_size=len(files))
        
		return dataset

	def train(self, dataset_train, dataset_test, save_path = None, epochs = 50):
		"""Fit the model and save intermediary results into `save_path`

		Args:
			dataset_train (tf.dataset): the tensor flow dataset for training
			dataset_test (tf.dataset): the tensor flow dataset for test
			save_path (string, optional): path to save the model to disk. If None, model is not saved.
			epochs (int, optional): number of training iterations. Defaults to 50.
		"""
		# put tf datasets in the format (train, train) and (img, img) and no labels to use in the .fit method
		dataset_train = dataset_train.map(lambda img, label: (img, img))
		dataset_test = dataset_test.map(lambda img, label: (img, img))

		if save_path:
			# define the checkpoint for saving intermediary results
			checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
				filepath=save_path,
				save_best_only=True,      # save only the model that has achieved the best performance on the validation set.
				monitor='val_loss',       # monitor the validation loss to determine the 'best' model.
				verbose=1,                # print a message whenever a checkpoint is saved.
				save_weights_only=False   # False => whole model is saved.
			)
			# fit the model               
			self.fit(dataset_train,
				epochs=epochs,
				validation_data=dataset_test,
				callbacks=[checkpoint_callback])
					# callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
		else:
			# fit the model
			self.fit(dataset_train,
				epochs=epochs,
				validation_data=dataset_test)

	def compress_and_save(self, dataset, compressed_path):
		for img_batch, path_batch in dataset:
			compressed_batch = self.predict(img_batch)
			for img, orig_path in zip(compressed_batch, path_batch.numpy()):
				filename = os.path.basename(orig_path.decode('utf-8'))
				label = os.path.basename(os.path.dirname(orig_path.decode('utf-8')))
				path_to_save = f"{compressed_path}{label}/{filename}"
				save_image(img, path_to_save)

	def crop_encoder_output(encoder_output, decoder_output):
		encoder_shape = tf.shape(encoder_output)
		decoder_shape = tf.shape(decoder_output)
		cropped_encoder_output = tf.image.resize(encoder_output, (decoder_shape[1], decoder_shape[2]))
		return cropped_encoder_output

	def call(self, inputs):
		encoded = self.encoder(inputs)
		decoded = self.decoder(encoded)
		return decoded

	def load_model(model_path):
		# autoencoder_load = tf.keras.models.load_model(model_path)
		autoencoder_load = tf.keras.models.load_model(model_path, custom_objects={'Autoencoder': Autoencoder})

		# get input/output image shape
		input_shape = autoencoder_load.layers[0].input_shape[1:]  # [1:] to exclude the batch dimension

		# instantiate autoencoder class
		autoencoder = Autoencoder(input_shape)

		# dummy tensorflow pass just to initialize the weights
		autoencoder.set_convolutional_network()
		dummy_data = tf.random.normal([1, input_shape[0], input_shape[1], 1])
		_ = autoencoder(dummy_data)
		autoencoder.set_weights(autoencoder_load.get_weights())

		return autoencoder

	def save_image(image, save_path):
		"""
		Save the compressed image to the specified path.

		Args:
		- image: A numpy array or TensorFlow tensor representing the compressed image.
		- save_path: The path where the image will be saved.
		"""
		# Ensure the directory exists
		directory = os.path.dirname(save_path)
		if not os.path.exists(directory):
			os.makedirs(directory)

		# If the image is a TensorFlow tensor, convert it to numpy
		if isinstance(image, tf.Tensor):
			image = image.numpy()

		# re-scale to [0, 255] and save the image
		tf.keras.preprocessing.image.save_img(save_path, image, "channels_last", scale=True)





















