### VISUAL SEARCH UTILIZANDO KERAS Y POOLING PIRAMIDAL ESPACIAL

Tutorial para la ejecución de archivos con la WHU-RS:

1) Colocar un conjunto de imágenes en la carpeta static/img yo utilicé la Whu-RS y el archivo feature_extractor_mobilenet_ft.py está preparado para esa base de datos
2) Correr offline.py
3) Se va a generar la carpeta static/feature en donde se almacenarán los descriptores de cada imagen
4) Ejecutar server.py y acceder a la dirección localhost:5000/  desde el navegador
5) Hacer la consulta deseada

ACLARACIÓN yo me separé un conjunto con el cual no fue entrenada la red para hacer mis consultas


Tutorial para la ejecución de archivos con cualquier base de datos:

1) Colocar un conjunto de imágenes en la carpeta static/img y tanto en el archivo server.py como en offline.py comentar el import feature_extractor_mobilenet_ft.py y descomentar el import feature_extractor_mobilenet.py
2) Correr offline.py
3) Se va a generar la carpeta static/feature en donde se almacenarán los descriptores de cada imagen
4) Ejecutar server.py y acceder a la dirección localhost:5000/  desde el navegador
5) Hacer la consulta deseada


Para entrenar utilicé Google Colab y hay que cargar las imagenes en el drive y configurar la ruta en el notebook antes de ejecutar el entrenamiento. Nombre del archivo: entrenamiento_mobilenet.ipynb
