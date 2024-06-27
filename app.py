import os
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import cv2
from keras.models import load_model
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('trainedModel.h5')
print('Model loaded. Check http://127.0.0.1:5000/')

def get_className(classNo):
    if classNo == 0:
        return """   Normal
A healthy kidney typically exhibits no side effects. However, maintaining overall kidney health is crucial for preventing potential complications, emphasizing a balanced lifestyle, hydration, and regular medical check-ups"""
    elif classNo == 1:
        return """ Tumor:
The side effects of kidney tumors may include pain, blood in the urine, fatigue, and unexplained weight loss. Malignant tumors can lead to systemic symptoms, such as fever and night sweats """
    elif classNo == 2:
        return """ Cyst:
Most kidney cysts are asymptomatic. However, large or complex cysts may cause pain, fever, or obstruction, impacting kidney function. In rare cases, cysts can rupture, leading to bleeding or infection. """
    elif classNo == 3:
        return """ Stone:
Kidney stones can cause intense pain, hematuria (blood in urine), frequent urination, and nausea. Larger stones may obstruct the urinary tract, potentially causing infection, kidney damage, or complications requiring medical intervention."""
    else:
        return "Invalid Class Number"

def getResult(img):
    image = cv2.imread(img)
    image = Image.fromarray(image, 'RGB')
    image = image.resize((64, 64))
    image = np.array(image)
    input_img = np.expand_dims(image, axis=0)
    result = model.predict(input_img)
    # Get the index of the class with the highest probability
    class_index = np.argmax(result, axis=1)
    return class_index[0]

@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        f = request.files['file']

        basepath = os.path.dirname(__file__)

        # Create 'uploads' directory if it doesn't exist
        if not os.path.exists(os.path.join(basepath, 'uploads')):
            os.makedirs(os.path.join(basepath, 'uploads'))

        file_path = os.path.join(
            basepath, 'uploads', secure_filename(f.filename))
        f.save(file_path)
        value = getResult(file_path)
        result = get_className(value)
        return result
    return None

if __name__ == '__main__':
    app.run(debug=True)
