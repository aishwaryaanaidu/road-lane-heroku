import json

from PIL import ImageTk, Image
import numpy as np
import base64
import io
from flask import Flask, request
from keras.models import load_model


model = load_model('traffic_classifier.h5')
#dictionary to label all traffic signs class.
classes_dict = { 1:'Speed limit (20km/h)',
           2:'Speed limit (30km/h)',
           3:'Speed limit (50km/h)',
           4:'Speed limit (60km/h)',
           5:'Speed limit (70km/h)',
           6:'Speed limit (80km/h)',
           7:'End of speed limit (80km/h)',
           8:'Speed limit (100km/h)',
           9:'Speed limit (120km/h)',
           10:'No passing',
           11:'No passing veh over 3.5 tons',
           12:'Right-of-way at intersection',
           13:'Priority road',
           14:'Yield',
           15:'Stop',
           16:'No vehicles',
           17:'Veh > 3.5 tons prohibited',
           18:'No entry',
           19:'General caution',
           20:'Dangerous curve left',
           21:'Dangerous curve right',
           22:'Double curve',
           23:'Bumpy road',
           24:'Slippery road',
           25:'Road narrows on the right',
           26:'Road work',
           27:'Traffic signals',
           28:'Pedestrians',
           29:'Children crossing',
           30:'Bicycles crossing',
           31:'Beware of ice/snow',
           32:'Wild animals crossing',
           33:'End speed + passing limits',
           34:'Turn right ahead',
           35:'Turn left ahead',
           36:'Ahead only',
           37:'Go straight or right',
           38:'Go straight or left',
           39:'Keep right',
           40:'Keep left',
           41:'Roundabout mandatory',
           42:'End of no passing',
           43:'End no passing vehicle with a weight greater than 3.5 tons' }


def classify(file_path):
    global label_packed
    #  image = Image.open(file_path)
    image = Image.open(io.BytesIO(file_path))
    image = image.resize((30, 30))
    image = np.expand_dims(image, axis=0)
    image = np.array(image)
    pred = model.predict([image])[0]
    classes_x = np.argmax(pred, axis=0)

    print(classes_x)
    sign = classes_dict[classes_x + 1]
    print(sign)
    return sign


def stringToImage(base64_string):
    imgdata = base64.b64decode(base64_string)
    # return Image.open(io.BytesIO(imgdata))
    return imgdata




app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict():
    t = request.get_data()
    new_str = t.decode('utf-8')
    # # print(new_str)
    data = json.loads(new_str)
    # data = request.get_json()
    print(data)
    test_string = data['input']
    print("Hereeee")
    print(test_string)
    resp = classify(stringToImage(test_string.split(",")[1]))
    # prediction = classify(test_string)
    # print(prediction)
    response = {"prediction": str(resp)}
    # response.headers.add('Access-Control-Allow-Origin', '*')
    return response


if __name__ == "__main__":
    app.run(debug=True)

