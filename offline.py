#Primer codigo a correr
#Extrae las deep features de todas las imagenes en el directorio asociado

from PIL import Image
#from feature_extractor import FeatureExtractor DESCOMENTAR CUANDO SE QUERA USAR EL QUE NO ESTA FINE TUNEADO Y COMENTAR LA SIGUIENTE LINEA
from feature_extractor_mobilenet_ft import FeatureExtractor
from pathlib import Path
import numpy as np

if __name__ == '__main__':
    fe = FeatureExtractor()

    for img_path in sorted(Path("./static/img").glob("**/*.jpg")): #CUIDADO con el path, puede confundir
        print(img_path)  # e.g., ./static/img/xxx.jpg
        feature = fe.extract(img=Image.open(img_path)) # feature es un arreglo numpy de 1024  elementos
        feature_path = Path("./static/feature") / (img_path.stem + ".npy")  # e.g., ./static/feature/xxx.npy
        np.save(feature_path, feature) #de cada imagen se almacenan esos 1024 valores en disco
        print(feature.shape)
