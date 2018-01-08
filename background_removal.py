import os
import sys
from PIL import Image
from scipy.misc import imresize
import numpy as np
from keras.models import load_model
import tensorflow as tf
import io
import requests
import base64

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
model = load_model(MODEL_PATH, compile=False)
graph = tf.get_default_graph()

def ml_predict(image):
    with graph.as_default():
        # Add a dimension for the batch
        prediction = model.predict(image[None, :, :, :])
    prediction = prediction.reshape((224,224, -1))
    return prediction


THRESHOLD = 0.5
def predict1(image):
    """Removed the background of given image.

    :param image: numpy array
    """
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

def read_image(image_spec):
    if not isinstance(image_spec, dict):
        return None
    if 'data' in image_spec:
        data = base64.b64decode(image_spec['data'])
    elif 'url' in image_spec:
        data = requests.get(image_spec['url']).content
    else:
        return None
    return Image.open(fp=io.BytesIO(data))

def write_image(image):
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)
    fp = io.BytesIO()
    image.save(fp, format='png')
    return {
        "data": base64.b64encode(fp.getvalue()).decode('ascii'),
        "content-type": "image/png"
    }

def predict(image):
    img=read_image(image)
    imgarray = predict1(np.array(img))
    return write_image(imgarray)
