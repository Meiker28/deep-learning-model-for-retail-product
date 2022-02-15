
#::: Import modules and packages :::
# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import numpy as np
from skimage import io
from skimage import img_as_ubyte
from skimage import transform
from tensorflow.keras.models import load_model
from tensorflow.keras.models import model_from_json
import keras
from tensorflow.python.framework import ops
ops.reset_default_graph()

#MODEL_ARCHITECTURE = './deploy/modelo_Inception.json'
#MODEL_WEIGHTS = './deploy/pesos_Inception.h5'
MODELO = './my_model_inception_uniq.h5'

#json_file = open(MODEL_ARCHITECTURE)
#loaded_model_json = json_file.read()
#json_file.close()
#model = model_from_json(loaded_model_json)
#model.load_weights(MODEL_WEIGHTS)
imgsize = 256

#model.save('my_model_inception_uniq.h5')
import tensorflow as tf
global graph
graph = tf.get_default_graph()
model = tf.keras.models.load_model(MODELO)

from tensorflow.python.framework import ops
ops.reset_default_graph()
# ::: MODEL FUNCTIONS ::: individual prediction
def model_predict(img_path):

    label = {'arroz': 0,'caramelo': 1,'mermelada': 2,'cafe': 3,'vinagre': 4,'chocolate': 5,'azucar': 6,
    'agua': 7, 'jugo': 8,'leche': 9,'gaseosa': 10,'nueces': 11,'chips': 12,'especias': 13,'cereal': 14,
    'frijoles': 15,'torta': 16,'miel': 17,'harina': 18,'pasta': 19,'salsatomate': 20,'te': 21,
    'maiz': 22,'aceite': 23,'pescado': 24}
    rev_label = {v:k for k,v in label.items()}
    imagen = io.imread(img_path)
    imagen = img_as_ubyte(transform.resize(imagen, (imgsize, imgsize)))
    imagen = np.array(imagen, np.float32)
    imagen /= 255.
    imagen = imagen.reshape(1, 256, 256, 3)
    with graph.as_default():
        prediction = model.predict(imagen)
    prediction = np.argmax(prediction, axis= 1)
    prediction = rev_label[prediction[0]]

    return prediction

def model_predict2(path):

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
    with graph.as_default():
        predictions = model.predict(x_test)

    predictions = np.argmax(predictions, axis=1)

    pred_labels = [rev_label[k] for k in predictions]

    sub = pd.DataFrame({'image_id':imagen_id, 'label':pred_labels})

    return sub
#
#keras.backend.clear_session()

#::: Flask App Engine :::
# Define a Flask app
app = Flask(__name__)

# ::: FLASK ROUTES
@app.route('/', methods=['GET'])
def index():
	# Main Page
	return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():


    if request.method == 'POST':

        # Get the file from post request
        f = request.files['file']
        #print(request.files['file'].file.name)
        #pp = os.path.dirname(f)
        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
        	basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make a prediction

        prediction = model_predict(file_path)
        predicted_class = prediction
        #print(basepath)
        #submission = model_predict2(basepath)
        #submission.to_csv(os.path.join(basepath, 'uploads'),index =False)
        #predicted_class = classes['TRAIN'][prediction[0]]
        #print('We think that is {}.'.format(predicted_class.lower()))

        return str(predicted_class).lower()


if __name__ == '__main__':
	app.run(debug = True)
