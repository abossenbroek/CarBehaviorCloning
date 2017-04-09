import argparse
import numpy as np
import pandas as pd
from skimage import io
import cv2


from keras.models import Model
from keras.layers import Conv2D
from keras import regularizers
from keras.layers import Dropout, Flatten, Dense, Input, Lambda
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint

MISSING = object()

def nvidia_model(input):
    x = Conv2D(3, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001))(input)
    x = ELU()(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = ELU()(x)
    x = Conv2D(36, (5, 5), strides=(2, 2), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = ELU()(x)
    x = Conv2D(48, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = ELU()(x)
    x = Conv2D(64, (3, 3), padding='valid', kernel_regularizer=regularizers.l2(0.001))(x)
    x = ELU()(x)

    x = Flatten()(x)
    x = Dense(1164, activation="elu", kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(100, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(50, kernel_regularizer=regularizers.l2(0.001))(x)
    x = Dropout(0.2)(x)
    x = Dense(1)(x)
    return x


def preprocess_img(file, fromColorSpace="BGR"):
    img = cv2.imread(file)
    if fromColorSpace == "BGR":
       convert = cv2.COLOR_BGR2YUV
    else:
        convert = cv2.COLOR_RGB2YUV
    img = cv2.cvtColor(img, convert)
    img = img[50:140,:,:]
    img = cv2.resize(img, (200,66), interpolation=cv2.INTER_AREA)
    img = np.asfarray(img)
    return img


def process_line(line, log_path):
    center_image_nm, left_image_nm, right_image_nm, angle, throttle, brake, speed = line.split(",")
    # Generate the angles
    center_angle = float(angle)
    # Add random angle adjustment.
    left_angle = center_angle + np.random.uniform(low=0.05, high=0.8)
    right_angle = center_angle - np.random.uniform(low=0.05, high=0.8)
    # Load the images
    center_img = preprocess_img(log_path + center_image_nm.strip())
    left_img = preprocess_img(log_path + left_image_nm.strip())
    right_img = preprocess_img(log_path + right_image_nm.strip())

    rv = np.random.uniform(0,1)
    # In half of the cased do a random rotation of -15 to 15 degrees.
    if rv < 0.5:
        rows, cols, colors = center_img.shape

        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-15,15), 1)
        center_img = cv2.warpAffine(center_img, rot_mat, (cols, rows))

        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-15,15), 1)
        left_img = cv2.warpAffine(left_img, rot_mat, (cols, rows))

        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-15,15), 1)
        right_img = cv2.warpAffine(right_img, rot_mat, (cols, rows))

    rv = np.random.uniform(0, 1)
    # Flip one on two center pictures.
    if rv < 0.5:
        center_angle = -center_angle
        center_img = np.fliplr(center_img)

    x = np.concatenate((center_img[np.newaxis, ...],
                        right_img[np.newaxis, ...],
                        left_img[np.newaxis, ...]), axis=0)
    y = np.vstack((center_angle, right_angle, left_angle))

    return [x, y]


def generate_input_from_file(log_file, log_path):
    while 1:
        f = open(log_file)
        for line in f:
            x, y = process_line(line, log_path)
            yield (x, y)
        f.close()


def load_original_file(log_file, log_path):
    X = []
    Y = []

    f = open(log_file)
    for i, line in enumerate(f):
        if i < 1:
            continue
        else:
            x, y = process_line(line, log_path)
            X.append(x)
            Y.append(y)
    f.close()

    return [np.concatenate(X, axis=0), np.concatenate(Y, axis=0)]


def build_model(model_path, data_path, learning_file, epochs, load=MISSING):
    # Load the log file.
    drive_log = "%s/driving_log.csv" % data_path

    X, y = load_original_file(drive_log, data_path)

    X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])


    img_input = Input(shape=X.shape[1:], dtype='float32', name="images")

    input = Lambda(lambda x: x/127.5 - 1.0)(img_input)

    x = nvidia_model(input)

    steering_output = Dense(1, activation='linear', name='steering_output')(x)

    model = Model(input=img_input,
                  output=steering_output)
    model.compile(optimizer='adamax',
                  loss='mse')
                  #loss='mean_absolute_percentage_error')
    print(model.summary())

    if load is not MISSING:
        model_file = "%s/model.json" % (load)
        weights_file = "%s/model.h5" % (load)
        print("Loading model from %s" % (load))
        model = model_from_json(model_file)
        model.load_weights(weights_file)

    early_stopping = EarlyStopping(monitor='val_loss', patience=20)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1,
            save_best_only=True)
    csv_logger = CSVLogger('training.log')

    model.fit(x=X,
              y=y,
              epochs=epochs, batch_size=200, validation_split=0.25,
              shuffle=True,
              callbacks=[early_stopping,
                  checkpointer,
                  csv_logger])

    if learning_file is not MISSING:
        model.fit_generator(generate_input_from_file(learning_file, data_path),
                            samples_per_epoch=200,
                            epochs=10)

    model_json_file = "%s/model.json" % (model_path)
    model_weights_file = "%s/model.h5" % (model_path)
    print("About to save model to '%s'" % (model_json_file))
    print("About to save model weights to '%s'" % (model_weights_file))

    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)

    print("Successfully saved JSON file")
    model.save_weights(model_weights_file)
    print("Successfully saved weights")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavior cloning training')
    parser.add_argument('--model', dest='model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--data', dest='data', type=str,
                        help='Path to data that should be used to train model')
    parser.add_argument('--epochs', dest='epochs', type=int,
                        help='Path to data that should be used to train model')
    parser.add_argument('--load', dest='load', type=str,
                        help='Name of the model to load to perform transfer learning.')

    #TODO: adjust to new paramaters in the function call.
    args = parser.parse_args()
    build_model(args.model, args.data, MISSING, args.epochs)
