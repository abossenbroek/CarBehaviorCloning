import argparse
import base64
import json

import numpy as np
import socketio
import eventlet
import eventlet.wsgi
import time
from PIL import Image
from PIL import ImageOps
from flask import Flask, render_template
from io import BytesIO
import cv2

from keras.models import model_from_json

sio = socketio.Server()
app = Flask(__name__)
model = None
prev_image_array = None

def preprocess_img(img, fromColorSpace="BGR"):
    if fromColorSpace == "BGR":
       convert = cv2.COLOR_BGR2YUV
    else:
        convert = cv2.COLOR_RGB2YUV
    img = cv2.cvtColor(img, convert)
    img = img[50:140,:,:]
    img = cv2.resize(img, (200,66), interpolation=cv2.INTER_AREA)
    img = np.asfarray(img)
    return img

@sio.on('telemetry')
def telemetry(sid, data):
    print("telemetry")
    # The current steering angle of the car
    steering_angle = data["steering_angle"]
    # The current throttle of the car
    throttle = data["throttle"]
    # The current speed of the car
    speed = data["speed"]

    # The current image from the center camera of the car
    imgString = data["image"]
    print("about image open")
    image = Image.open(BytesIO(base64.b64decode(imgString)))
    print("done with image open")
    X = np.asarray(image)
    print("about to preprocess")
    X = preprocess_img(X, fromColorSpace="RGB")
    print("done with preprocess")
    print("going to reshape")
    X = X.reshape(1, X.shape[2], X.shape[0], X.shape[1])
    print("done to reshape")
    X = X.astype('float32')

    # This model currently assumes that the features of the model are just the images. Feel free to change this.

    print("about to predict")
    prediction = model.predict(X, batch_size=1, verbose=1)
    print("predicted: " + str(prediction))
    steering_angle = prediction[0][0]#[0][0]
    throttle = max(0.1, -0.15/0.05 * abs(steering_angle) + 0.35)
    speed = 0.2 #prediction[2][0][0]

    # The driving model currently just outputs a constant throttle. Feel free to edit this.
    print(steering_angle, throttle, speed)
    send_control(steering_angle, throttle, speed)


@sio.on('connect')
def connect(sid, environ):
    print("connect ", sid)
    send_control(0, 0, 0)


def send_control(steering_angle, throttle, speed):
    sio.emit("steer", data={
        'steering_angle': steering_angle.__str__(),
        'throttle': throttle.__str__(),
        'speed': speed.__str__()
    }, skip_sid=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Remote Driving')
    parser.add_argument('model', type=str,
    help='Path to model definition json. Model weights should be on the same path.')
    args = parser.parse_args()
    with open(args.model, 'r') as jfile:
        # NOTE: if you saved the file by calling json.dump(model.to_json(), ...)
        # then you will have to call:
        #
        # model = model_from_json(json.loads(jfile.read()))
        #
        # instead.
        model = model_from_json(jfile.read())


    model.compile("adam", "mse")
    weights_file = args.model.replace('json', 'h5')
    model.load_weights(weights_file)

    # wrap Flask application with engineio's middleware
    app = socketio.Middleware(sio, app)

    # deploy as an eventlet WSGI server
    eventlet.wsgi.server(eventlet.listen(('', 4567)), app)
