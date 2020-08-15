import sys
import os
import glob
import re
import numpy as np
import cv2

# Keras

from keras.models import load_model
from keras.preprocessing import image

# Flask utils
from flask import Flask, redirect, url_for, request, render_template
from werkzeug.utils import secure_filename

# Define a flask app
app = Flask(__name__)

# Model saved with Keras model.save()
MODEL_PATH = 'models/model.h5'

# Load your trained model
model = load_model(MODEL_PATH)
model._make_predict_function()          
print('Model loaded. Start serving...')




def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128, 128))

    # Preprocessing the image
    img = image.img_to_array(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = img.astype('float32')
    img/=255
    img = np.expand_dims(img, axis=0)
    img = np.expand_dims(img, axis=3)

    # Be careful how your trained model deals with the input
    # otherwise, it won't make correct prediction!
    preds = img
    return preds


@app.route('/', methods=['GET'])
def index():
    # Main page
    return render_template('index.html')


def int_to_string(argument):
    switcher = {

        0: "one",
        1: "two",
        2: "three",
        3: "four",
        4: "five",
        5: "six",
        6: "seven",
        7: "eight",
        8: "nine",
        9: "Letter A",
        10: "Letter B",
        11: "Letter C",
        12: "Letter D",
        13: "Letter E",
        14: "Letter F",
        15: "Letter G",
        16: "Letter H",
        17: "Letter I",
        18: "Letter K",
        19: "Letter L",
        20: "Letter M",
        21: "Letter N",
        22: "Letter O",
        23: "Letter P",
        24: "Letter Q",
        25: "Letter R",
        26: "Letter S",
        27: "Letter T",
        28: "Letter U",
        29: "Letter V",
        30: "Letter W",
        31: "Letter X",
        32: "Letter Y",

    }

    # Get the function from switcher dictionary
    func = switcher.get(argument, lambda: "Invalid")
    # Execute the function
    return func




@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Get the file from post request
        f = request.files['file']

        # Save the file to ./uploads
        basepath = os.path.dirname(__file__)
        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)

        # Make prediction
        preds = model_predict(file_path, model)


        # Convert to string

        result = model.predict(preds, batch_size = 32)
        f_result = np.argmax(result[0])
  

       	#return result
       	new_result = int_to_string(f_result)
       	result = new_result
        
        return result



    return None


if __name__ == '__main__':
    app.run(port=5002, debug=True)


