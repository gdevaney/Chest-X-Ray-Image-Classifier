import matplotlib.pyplot as plt
import imageio
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2' # suppress tensorflow warnings
import random
import numpy as np
import tensorflow as tf
from joblib import Parallel, delayed

#===============================================================================
# functions to work with the directories, get image paths, etc
#===============================================================================
def get_image_paths(folder_path:str, n:int=None, valid_extensions:list=['.jpg', '.jpeg', '.png', '.gif']) -> list:
    """Get `n` full paths to images in `folder_path`. If n=-1, get all images. If n=>0, get a sample without repetition, uniformly distributed.
    Args:
        folder_path (str): folder where images are located
        n (int): number of images to sample. If None, get all images in the folder.
        valid_extensions (list, optional): extensions of the image files in the folder. Defaults to ['.jpg', '.jpeg', '.png', '.gif'].
    Raises:
        ValueError: If no image is found, raise error.
    Returns:
        list: list of paths for images in `folder_path`
    """    
    # List all files in the directory
    files = [f for f in os.listdir(folder_path) if os.path.isfile(os.path.join(folder_path, f))]
    
    # Filter out files that aren't images (based on extension)
    images = [f for f in files if any(f.endswith(ext) for ext in valid_extensions)]

    # raise error if no images found
    if not images:
        raise ValueError("No valid images found in the specified directory.")

    # Select n random images
    if n:
        images = random.sample(images, n)
    
    # Return full paths to the images
    return [os.path.join(folder_path, img) for img in images]


def get_images(folder_paths:list, n:int=None, valid_extensions:list=['.jpg', '.jpeg', '.png', '.gif']) -> list:
    """ Get `n` images randomly from the folders in `folder_paths`

    Args:
        folder_paths (list): list of folders to get images from
        n (int, optional): number of images to retrive randomly. If None, gets all images.
        valid_extensions (list, optional): extensions for valid images. Defaults to ['.jpg', '.jpeg', '.png', '.gif'].

    Returns:
        list: _description_
    """
    img_paths = []
    for folder in folder_paths:
        img_paths += get_image_paths(folder_path=folder, n=n, valid_extensions=valid_extensions)

    load_img = lambda path: load_image(path, grayscale=True)
    imgs = Parallel(n_jobs=-1)(delayed(load_img)(img_path) for img_path in img_paths)

    return imgs

def get_subfolders(base_path:str):
    return [os.path.join(base_path, subfolder) for subfolder in os.listdir(base_path) 
                       if os.path.isdir(os.path.join(base_path, subfolder))]

#===============================================================================
# functions for image processing: load, isGrayscale, resize
#===============================================================================

def load_image(img_path:str, grayscale:bool=False) -> np.ndarray:
    """Load the image from `img_path` into a np.array

    Args:
        img_path (str): path to the image to be loaded
        grayscale (bool, optional): convert the image to grayscale. Defaults to False.

    Returns:
        np.ndarray:  numpy array with dimension (height, width, channels) containg image pixels
    """
    img = imageio.v2.imread(img_path, pilmode="RGB")

    if grayscale and not is_grayscale(img):
        # collapses the RGB axis to a grayscale
        img = np.dot(img[...,:3], [0.2989, 0.5870, 0.1140])  # img.shape [h,w,3] -> [w, h]
        img = img[..., np.newaxis] # img.shape [w,h] -> [w, h, 1]

    return img

def is_grayscale(img:np.ndarray):
    """ Returns true if the RGB image is grayscale.
    """
    
    # Check if image has only one channel
    if len(img.shape) == 2:
        return True
    
    # Check if Red, Green, and Blue channels have the same values across the image
    # TODO: add some tolerancy (might not be exactly equal but suff close to grayscale)
    return np.array_equal(img[:,:,0], img[:,:,1]) and np.array_equal(img[:,:,0], img[:,:,2])


def resize_images_parallel(img_paths:list, target_height:int, target_width:int, output_dir:str, batch_size:int=32, padding:bool=True):
    """ Uses tensorflow to resize images to a common size in parallel and without distortion
    Args:
        img_paths (list): paths to the images to be resized
        target_height (int): height of final image
        target_width (int): width of final image
        output_dir (str): base path of the directory to save the images
        batch_size (int, optional): how many images to load and resize per batch. Defaults to 32.
          # obs: do the process in batches to reduce RAM consumption, especially for Google Colab
    """
    # Convert the image paths to a TensorFlow dataset
    img_paths_dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    
    # if output dir does not exist, create it
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Batch the dataset
    img_paths_dataset = img_paths_dataset.batch(batch_size)

    # load, resize and write to disk images in parallel
    processed_paths_dataset = img_paths_dataset.map(
        lambda x: load_resize_save_image_batch(x, target_height, target_width, output_dir, padding),
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    )
    
    # trigger the process (tf.data.Dataset operations are lazy; this forces execution)
    [path for batch in processed_paths_dataset for path in batch]


def load_resize_save_image_batch(img_paths_batch, target_height, target_width, output_dir, padding=False):
    """ Auxiliary function to `resize_images_parallel`
    """
    def load_resize_save_single_image(img_path):
        # Load the Image
        image = tf.io.read_file(img_path)
        image = tf.image.decode_image(image, channels=3, expand_animations=False)
        image.set_shape([None, None, 3])
        image = tf.image.rgb_to_grayscale(image)

        # shrink/expand image with zero-padding (i.e., fill pixels with zeros where necessary)
        if padding:
          resized_image = tf.image.resize_with_pad(image, target_height, target_width, 
                                                  method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        else:
          resized_image = tf.image.resize(image, (target_height, target_width), 
                                    method=tf.image.ResizeMethod.BILINEAR, antialias=True)
        # Save the Resized Image
        output_path = tf.strings.join([output_dir, tf.strings.split(img_path, os.sep)[-1]])
        encoded_image = tf.image.encode_jpeg(tf.cast(resized_image, tf.uint8))
        tf.io.write_file(output_path, encoded_image)
        return output_path
    
    # apply the single-image processing function to each image in the batch
    return tf.map_fn(load_resize_save_single_image, img_paths_batch, fn_output_signature=tf.TensorSpec(shape=(), dtype=tf.string))



#===============================================================================
# functions for plotting
#===============================================================================
def plot_grid_images(image_paths:list, figsize=(10,10), grid_size=None) -> None:
    """
    Load images in `image_paths` as display in a grid.
    
    Args:
        image_paths (list): List of full paths to the images.
        grid_size (tuple, optional): Tuple indicating the grid size as (rows, cols). If None, the function tries to create a square grid.
        fig_size (tuple, optional): Tuple indicating the size of each subplot
    """
    if grid_size is None:
        # Try to create a roughly square grid
        rows = cols = int(len(image_paths) ** 0.5)

        # Adjust grid size if not perfectly square
        if rows * cols < len(image_paths):
            cols += 1
        if rows * cols < len(image_paths):
            rows += 1
    else:
        rows, cols = grid_size

    fig, axs = plt.subplots(rows, cols, figsize=figsize)

    for ax, img_path in zip(axs.ravel(), image_paths):
        # obs: use PILMODE so the images are displayed as gray by pyplot (else may appear as green)
        img = imageio.v2.imread(img_path, pilmode="RGB")
        ax.imshow(img)
        ax.axis('off')  # Hide axes

    # Turn off any remaining axes that didn't get an image
    for ax in axs.ravel()[len(image_paths):]:
        ax.axis('off')
    
    plt.show()

