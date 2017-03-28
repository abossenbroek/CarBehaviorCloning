import argparse
import numpy as np
import pandas as pd
from skimage import io
import cv2

from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras import regularizers
from keras.layers import Dropout, Flatten, Dense, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, CSVLogger

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
    # Load the images
    center_img = preprocess_img(center_image_nm)
    left_img = preprocess_img(left_image_nm)
    right_img = preprocess_img(right_image_nm)

    # Generate the angles
    center_angle = float(angle)
    left_angle = center_angle + np.random.uniform(low=0, high=0.15)
    right_angle = center_angle - np.random.uniform(low=0, high=0.15)

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


def build_model(model_path, data_path, epochs, threshold, load=MISSING):
    # Load the log file.
    drive_log = pd.read_csv("%s/driving_log.csv" % (data_path))

    # Load the left, center and right images that come from the simulator,
    # these are the training features.
    print("Loading training images:")
    center_images = load_images(drive_log['center'][abs(steering) > threshold],
                                data_path)
    print("center [%s, %s, %s, %s]" % (center_images.shape))
    left_images = load_images(drive_log['left'][abs(steering) > threshold],
                                data_path)
    print("left [%s, %s, %s, %s]" % (left_images.shape))
    right_images = load_images(drive_log['right'][abs(steering) > threshold],
                                data_path)
    print("right [%s, %s, %s, %s]" % (right_images.shape))

    # Copy all the images in one big array.
    images = np.concatenate((left_images, center_images, right_images))

    flipped_images = np.fliplr(images.reshape(90, 320, 3, images.shape[0]))
    flipped_images = flipped_images.reshape(images.shape)

    images = np.concatenate((images, flipped_images))

    print("training on total of [%s, %s, %s, %s]" % (images.shape))

    # Keep only steering angles higher than the threshold.
    steering = steering[abs(steering) > threshold]
    # Copy the series three times, once for each camera viewpoint.
    steering = pd.concat(
        (steering + ),
         steering,
         steering - np.random.uniform(low=0, high=0.15, size = len(steering)),
         -(steering + np.random.uniform(low=0, high=0.15, size = len(steering))),
         -steering,
         -(steering - np.random.uniform(low=0, high=0.15, size = len(steering)))))

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

    model.fit(x=images,
              y=steering,
              nb_epoch=epochs, batch_size=200, validation_split=0.25,
              shuffle=True,
              callbacks=[early_stopping,
                         csv_logger])

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

    args = parser.parse_args()
    build_model(args.model, args.data, args.epochs, args.threshold)
