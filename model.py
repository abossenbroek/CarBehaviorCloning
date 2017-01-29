import argparse
import numpy as np
import pandas as pd
from skimage import io
from skimage import transform

from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import Dropout, Flatten, Dense, Input, merge
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import PReLU


def load_images(files, data_path):
    def load_func(files):
        for fl in files:
            img = io.imread('%s/%s' % (data_path, fl.strip()))
            img = transform.downscale_local_mean(img, factors=(2, 2, 1))
            yield img

    images = np.stack(load_func(files), axis=-1)
    images = images.reshape([images.shape[3], images.shape[2],
                             images.shape[0], images.shape[1]])
    images = images.astype('float32')
    images = images/256.0
    return images


def build_model(model_path, data_path, epochs):
    # Load the log file.
    drive_log = pd.read_csv("%s/driving_log.csv" % (data_path))

    # Load the left, center and right images that come from the simulator,
    # these are the training features.
    print("Loading training images:")
    left_images = load_images(drive_log['left'], data_path)
    print("left [%s, %s, %s, %s]" % (left_images.shape))
    center_images = load_images(drive_log['center'], data_path)
    print("center [%s, %s, %s, %s]" % (center_images.shape))
    right_images = load_images(drive_log['right'], data_path)
    print("right [%s, %s, %s, %s]" % (right_images.shape))

    # Load the training labels.
    steering = drive_log['steering']
    throttle = drive_log['throttle']
    brake = drive_log['brake']
    speed = drive_log['speed']

    left_img = Input(shape=(3, 80, 160), dtype='float32', name="left_img")
    left_input = BatchNormalization()(left_img)
    left_input = Convolution2D(32, 5, 5, border_mode='same')(left_input)
    left_input = PReLU()(left_input)
    left_input = Dropout(0.25)(left_input)
    right_img = Input(shape=(3, 80, 160), dtype='float32', name="right_img")
    right_input = BatchNormalization()(right_img)
    right_input = Convolution2D(32, 5, 5, border_mode='same')(right_input)
    right_input = PReLU()(right_input)
    right_input = Dropout(0.25)(right_input)
    center_img = Input(shape=(3, 80, 160), dtype='float32', name="center_img")
    center_input = BatchNormalization()(center_img)
    center_input = Convolution2D(32, 5, 5, border_mode='same')(center_input)
    center_input = PReLU()(center_input)
    center_input = Dropout(0.25)(center_input)

    x = merge([left_input, center_input, right_input], mode='concat')
    x = Convolution2D(3, 5, 5, border_mode='same')(x)
    x = PReLU()(x)
    x = Convolution2D(24, 5, 5, border_mode='same')(x)
    x = PReLU()(x)
    x = Convolution2D(36, 5, 5, border_mode='same')(x)
    x = PReLU()(x)
    x = Convolution2D(48, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = PReLU()(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(10)(x)
    steering_output = Dense(1, activation='sigmoid', name='steering_output')(x)
    throttle_output = Dense(1, activation='linear', name='throttle_output')(x)
    brake_output = Dense(1, activation='linear', name='brake_output')(x)
    speed_output = Dense(1, activation='linear', name='speed_output')(x)

    model = Model(input=[left_img, center_img, right_img],
                  output=[steering_output, throttle_output,
                          brake_output, speed_output])
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(x=[left_images, center_images, right_images],
              y=[steering, throttle, brake, speed],
              nb_epoch=epochs, batch_size=100)

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

    args = parser.parse_args()
    build_model(args.model, args.data, args.epochs)
