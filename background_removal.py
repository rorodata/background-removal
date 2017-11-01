import os
import sys
from PIL import Image, ExifTags
from scipy.misc import imresize
import numpy as np
import keras
from keras.models import load_model
import tensorflow as tf
import io
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

MODEL_URL = 'https://gitlab.com/fast-science/background-removal-server/raw/master/webapp/model/main_model.hdf5'
MODEL_PATH = '/volumes/data/main_model.hdf5'

def download_model():
    """Downloads the model file.
    """
    if os.path.exists(MODEL_PATH):
        print("Model file is already downloaded.")
        return
    # Download to a tmp file and move it to final file to avoid inconsistent state
    # if download fails or cancelled.
    print("Model file is not available. downloading...")
    exit_status = os.system("wget {} -O {}.tmp".format(MODEL_URL, MODEL_PATH))
    if exit_status == 0:
        os.system("mv {}.tmp {}".format(MODEL_PATH, MODEL_PATH))
    else:
        print("Failed to download the model file", file=sys.stderr)
        sys.exit(1)

# Preload our model
download_model()
print("Loading model")
model = load_model('/volumes/data/main_model.hdf5', compile=False)
graph = tf.get_default_graph()

def ml_predict(image):
    with graph.as_default():
        # Add a dimension for the batch
        prediction = model.predict(image[None, :, :, :])
    prediction = prediction.reshape((224,224, -1))
    return prediction


THRESHOLD = 0.5
def predict1(image):
    height, width = image.shape[0], image.shape[1]
    resized_image = imresize(image, (224, 224)) / 255.0

    # Model input shape = (224,224,3)
    # [0:3] - Take only the first 3 RGB channels and drop ALPHA 4th channel in case this is a PNG
    prediction = ml_predict(resized_image[:, :, 0:3])
    print('PREDICTION COUNT', (prediction[:, :, 1]>0.5).sum())

    # Resize back to original image size
    # [:, :, 1] = Take predicted class 1 - currently in our model = Person class. Class 0 = Background
    prediction = imresize(prediction[:, :, 1], (height, width))
    
    prediction[prediction<THRESHOLD*255] = 0
    prediction[prediction>=THRESHOLD*255] = 1
    
    #return prediction

    res1=prediction*image[0:,:,0]
    res2=prediction*image[0:,:,1]
    res3=prediction*image[0:,:,2]
    img2=np.dstack([res1,res2,res3])
    return img2

def predict(image_file, format='jpg'):
    image=plt.imread(image_file, format=format)
    img2=predict1(image)
    f = io.BytesIO()
    plt.imsave(f, img2)
    f.seek(0)
    return f



