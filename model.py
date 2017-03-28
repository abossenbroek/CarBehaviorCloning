import argparse
import numpy as np
import pandas as pd
from skimage import io
import cv2


from keras.models import Model
from keras.layers import Convolution2D
from keras import regularizers
from keras.layers import Dropout, Flatten, Dense, Input, Lambda
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, CSVLogger

MISSING = object()

def load_images(files, data_path):
    def load_func(files):
        for fl in files:
            img = io.imread('%s/%s' % (data_path, fl.strip()))
            img = img[50:140, 0:320]
            yield img

    images = np.stack(load_func(files), axis=-1)
    images = images.reshape([images.shape[3], images.shape[2],
                             images.shape[0], images.shape[1]])
    images = images.astype('float32')
    images = images/256.0
    return images

def nvidia_model(input):
    x = Convolution2D(3, 5, 5, border_mode='valid', subsample=(2, 2),
                      W_regularizer=regularizers.l2(0.01))(input)
    x = ELU()(x)
    x = Convolution2D(24, 5, 5, border_mode='valid',subsample=(2, 2),
                      W_regularizer=regularizers.l2(0.01))(x)
    x = ELU()(x)
    x = Convolution2D(36, 5, 5, border_mode='valid', subsample=(2, 2),
                      W_regularizer=regularizers.l2(0.01))(x)
    x = ELU()(x)
    x = Convolution2D(48, 3, 3, border_mode='valid', subsample=(2, 2),
                      W_regularizer=regularizers.l2(0.01))(x)
    x = ELU()(x)
    x = Convolution2D(64, 3, 3, border_mode='valid', subsample=(2, 2),
                      W_regularizer=regularizers.l2(0.01))(x)
    x = ELU()(x)

    x = Flatten()(x)
    x = Dense(1164, activation="elu", W_regularizer=regularizers.l2(0.01))(x)
    x = Dense(512, W_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(100, W_regularizer=regularizers.l2(0.01))(x)
    x = Dropout(0.2)(x)
    x = Dense(50, W_regularizer=regularizers.l2(0.01))(x)
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

def process_line(line):
    center_image_nm, left_image_nm, right_image_nm, angle, steering, throttle, brake, speed = line.split(",")
    # Generate the angles
    center_angle = float(angle)
    # Add random angle adjustment.
    left_angle = center_angle + np.random.uniform(low=0.05, high=0.8)
    right_angle = center_angle - np.random.uniform(low=0.05, high=0.8)
    # Load the images
    center_img = preprocess_img(center_image_nm)
    left_img = preprocess_img(left_image_nm)
    right_img = preprocess_img(right_image_nm)

    rv = np.random.uniform(0,1)
    # In half of the cased do a random rotation of -15 to 15 degrees.
    if rv < 0.5
        rows, cols = center_img.shape

        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), rv.uniform(-15,15), 1)
        center_img = cv2.warpAffine(center_img, rot_mat, (cols, rows))

        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), rv.uniform(-15,15), 1)
        left_img = cv2.warpAffine(left_img, rot_mat, (cols, rows))

        rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), rv.uniform(-15,15), 1)
        right_img = cv2.warpAffine(right_img, rot_mat, (cols, rows))

    rv = np.random.uniform(0,1)
    # Flip one on two center pictures.
    if rv < 0.5:
        center_angle = -center_angle
        center_img = np.fliplr(center_img)

    x = np.vstack((center_img, right_img, left_img))
    y = np.vstack((center_angle, right_angle, left_angle))

    return [x, y]


def generate_input_from_file(path):
    while 1:
        f = open(path)
        for line in f:
            x, y = process_line(line)
            yield (x, y)
        f.close()


def load_original_file(path):
    X = []
    Y = []

    f = open(path)
    for i, line in enumerate(f):
        if i < 1:
            continue
        else:
            x, y = process_line(line)
            X = np.vstack((X, x))
            Y = np.vstack((Y, y))
    f.close()

    return [X, Y]


def build_model(model_path, data_path, learning_file, epochs, threshold, load=MISSING):
    # Load the log file.
    drive_log = pd.read_csv("%s/driving_log.csv" % (data_path))

    X, y = load_original_file(drive_log)


    img_input = Input(shape=(66, 200, 3), dtype='float32', name="images")

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
    csv_logger = CSVLogger('training.log')

    model.fit(x=X,
              y=y,
              nb_epoch=epochs, batch_size=200, validation_split=0.25,
              shuffle=True,
              callbacks=[early_stopping,
                         csv_logger])

    model.fit_generator(generate_input_from_file(learning_file),
                        samples_per_epoch=200,
                        epochs=10)

    model_json_file = "%s/model.json" % (model_path)
    model_weights_file = "%s/model.h5" % (model_path)
    print("About to save model to '%s'" % (model_json_file))
    print("About to save model weights to '%s'" % (model_weights_file))

    model_json = model.to_json()
    with open(model_json_file, "w") as json_file:
        json_file.write(model_json)

    print("Succesfully saved JSON file")
    model.save_weights(model_weights_file)
    print("Succesfully saved weights")

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
    parser.add_argument('--threshold', dest='threshold', type=float,
                        help='Driving angle threshold that should be met before using an input for calibration.')

    #TODO: adjust to new paramaters in the function call.
    args = parser.parse_args()
    build_model(args.model, args.data, args.epochs, args.threshold)
