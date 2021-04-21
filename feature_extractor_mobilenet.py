#MOBILENET
#Este archivo precarga la red MobileNet con los pesos de ImageNet
#No está especializado para ningun conjunto de imágenes en particular.

from tensorflow.keras.preprocessing import image
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from SpatialPyramidPooling import SpatialPyramidPooling
from tensorflow.keras.models import Model
import numpy as np

class FeatureExtractor:
    def __init__(self):
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=[None, None, 3])
        y = SpatialPyramidPooling([1, 2, 4])
        # input shape para aceptar entradas de tamaño variable
        #self.model = Model(inputs=base_model.input, outputs=SpatialPyramidPooling([1, 2, 4]).output)
        #self.model = Model(inputs=base_model.input, outputs=base_model.outputs)
        # intento de aplciar el modulo spp---> lo de abajo es sin el spp
        # self.model = Model(inputs=base_model.input, outputs=base_model.get_layer('global_average_pooling2d').output)

       #  x = Flatten(name='flatten')(base_model)
        x = SpatialPyramidPooling([1, 2, 4])(base_model.output)
        #x = Dense(1024, activation='relu', name='fc1')(x)

        self.model = Model(inputs=base_model.input, outputs=x)
        #Ahora voy a intentar agregar la capa de SPP
        """model_final = Sequential()
        model_final.add(SpatialPyramidPooling([1, 2, 4]))
        final_model = Model(self.model.input, model_final)
        self.model = final_model"""
        #base_model = MobileNet(weights='imagenet')
        #newModel = Model(base_model.input,  outputs=base_model.outputs)
        print("*********************** NEW MODEL *************************")
        print(self.model.summary())
        #newModel = Model(base_model.inputs,  [model.layers[5].output, model.layers[i].output])
        #base_model.get_layer('global_average_pooling2d')




    #cada vez que se haga una llamada a extract con una imagen como argumento retornará el conjunto de características
    #normalizadas
    def extract(self, img):
        """
        Args:
            img: de PIL.Image.open(path) o tensorflow.keras.preprocessing.image.load_img(path)

        Salida:
            feature (np.ndarray)
        """
        #img = img.resize((224, 224))  # Como tiene la capa SPP esto se comenta
        img = img.convert('RGB')  # La imagen tiene que ser a color (3 canales)
        x = image.img_to_array(img)  # Se pasa a np.array. Height x Width x Channel. dtype=float32
        x = np.expand_dims(x, axis=0)  # (H, W, C)->(1, H, W, C), en donde el primer elemento es el nro de imagenes
        #Keras trabaja con batches de imagenes, entonces la primer dimensión se utiliza para el número de muestras (o imágenes)
        # que tenemos. Cuando cargamos una única imágen se obtiene la shape de una imagen, la cual es (tam1,tam2,canales).
        #Para crear un barch de imágenes se necesita una dimensión adicional (muestras, tam1,tam2,canales)
        x = preprocess_input(x)
        # La función de preprocesameinto de la entrada tiene el propósito de adecuar la imagen al formato que el modelo requiera
        # Algunos modelos usan imágenes con valores en el rango de 0 a 1. Otros de -1 a +1. Otros el estilo 'caffe' que no
        #está normalizado, sí centrado
        feature = self.model.predict(x)[0]  #(1, 1024) -> (1024, ) mobilenet (la ultima capa extractora de características tendrá 1024 features)

        return feature / np.linalg.norm(feature)  # Se normaliza


