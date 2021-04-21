#ARCHIVO DE PRUEBAS DE DISTINTAS COSAS. NO ES IMPORTANTE

import tensorflow as tf
from tensorflow.python.client import device_lib
from feature_extractor_mobilenet import FeatureExtractor
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from tensorflow import keras


"""print("# GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

print('Tensorflow: ', tf.__version__)
fe = FeatureExtractor()

def get_available_gpus():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

xx= get_available_gpus()
print('The GPU device is: ', xx)
print('Tensorflow: ', tf.__version__)

fe = FeatureExtractor();"""

#Prueba cargar pesos de entrenamiento:

new_model = keras.models.load_model(r'C:\Users\Rocío\Downloads\mobilenet-fine-tuned.h5')
base_model = new_model.get_layer("mobilenet_1.00_224")
base_model.save(r'C:\Users\Rocío\Downloads\mobilenet_ft.h5')
base_model.summary()