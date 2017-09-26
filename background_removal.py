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


# Preload our model
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



