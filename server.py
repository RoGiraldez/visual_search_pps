#SERVIDOR PYTHON
#SE CORRE LUEGO DEL ARCHIVO OFFLINE.PY


import numpy as np
from PIL import Image
#from feature_extractor import FeatureExtractor  comento esta linea porque es de las primeras pruebas
from feature_extractor_mobilenet_ft import FeatureExtractor
from datetime import datetime
from flask import Flask, request, render_template
from pathlib import Path

app = Flask(__name__) #crea la instancia flask

# Lee las features de cada imagen que están en disco
fe = FeatureExtractor()
features = []
img_paths = []
#cuando inicia server.py se cargan en memoria los vectores de features que estaban almacenados en disco
for feature_path in Path("./static/feature").glob("*.npy"):
    features.append(np.load(feature_path))
    cadena = feature_path.stem.split('_')
    img_paths.append(Path("./static/img") / (cadena[0] + "/" + feature_path.stem + ".jpg"))
    #img_paths.append(Path("./static/img") / ( feature_path.stem + ".jpg"))
features = np.array(features)
#Obtengo features de tipo Numpy Array y img_paths de tipo lista

@app.route('/', methods=['GET', 'POST']) #RUTA DE LA APLICACION
def index():
    print(features.shape)
    if request.method == 'POST':
        file = request.files['query_img']

        # Se almacena la imagen a comparar
        img = Image.open(file.stream)  # PIL image
        uploaded_img_path = "static/uploaded/" + datetime.now().isoformat().replace(":", ".") + "_" + file.filename
        img.save(uploaded_img_path)

        # Se ejecuta la búsqueda
        query = fe.extract(img)

        dists = np.linalg.norm(features-query, axis=1)  # distancia L2 entre las features. Cuanto menor valor sea, los puntos serán más similares
        print(dists)
        ids = np.argsort(dists)[:30]  # Top 30 resultados
        scores = [(dists[id], img_paths[id]) for id in ids]

        return render_template('index.html',
                               query_path=uploaded_img_path,
                               scores=scores)
    else:
        return render_template('index.html')


if __name__=="__main__":
    app.run("0.0.0.0")
