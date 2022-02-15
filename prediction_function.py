import numpy as np
from skimage import io
from skimage import img_as_ubyte
from skimage import transform
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json

import os
import pandas as pd
from tensorflow.python.framework import ops
ops.reset_default_graph()

MODEL_ARCHITECTURE = './deploy/modelo_Inception.json'
MODEL_WEIGHTS = './deploy/pesos_Inception.h5'


json_file = open(MODEL_ARCHITECTURE)
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(MODEL_WEIGHTS)
imgsize = 256


# ::: MODEL FUNCTIONS ::: all file predictions
def model_predict(path):

    label = {'arroz': 0,'caramelo': 1,'mermelada': 2,'cafe': 3,'vinagre': 4,'chocolate': 5,'azucar': 6,
    'agua': 7, 'jugo': 8,'leche': 9,'gaseosa': 10,'nueces': 11,'chips': 12,'especias': 13,'cereal': 14,
    'frijoles': 15,'torta': 16,'miel': 17,'harina': 18,'pasta': 19,'salsatomate': 20,'te': 21,
    'maiz': 22,'aceite': 23,'pescado': 24}

    imagen_id = [str(i)[:-4] for i in np.sort(os.listdir(path))]
    rev_label = {v:k for k,v in label.items()}
    test_img = []
    for img_path in np.sort(os.listdir(path)):
        img = io.imread(path+r'/'+img_path)
        img = img_as_ubyte(transform.resize(img, (256, 256)))
        test_img.append(img)

    x_test = np.array(test_img, np.float32) / 255.
    predictions = model.predict(x_test)
    predictions = np.argmax(predictions, axis=1)

    pred_labels = [rev_label[k] for k in predictions]

    sub = pd.DataFrame({'image_id':imagen_id, 'label':pred_labels})

    return sub

#model_predict(path)
