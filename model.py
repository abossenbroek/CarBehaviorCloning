import argparse
import numpy as np
import pandas as pd
from skimage import io

from keras.models import Sequential
from keras.layers import Convolution2D, Model
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Merge, MaxPooling2D


def load_images(files, data_path):
    def load_func(files):
        for fl in files:
            img = io.imread('%s/%s' % (data_path, fl.strip()))
            yield img

    return np.stack(load_func(files), axis=-1)


def build_model(model_path, data_path):
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

    model = Sequential()
    left_input = Convolution2D(32, 3, 3, border_mode='same',
                               activation='relu',
                               input_shape=left_images.shape[0:3])
    right_input = Convolution2D(32, 3, 3, border_mode='same',
                                activation='relu',
                                input_shape=right_images.shape[0:3])
    center_input = Convolution2D(32, 3, 3, border_mode='same',
                                 activation='relu',
                                 input_shape=center_images.shape[0:3])

    x = Merge([left_input, center_input, right_input], mode='concat')
    x = Convolution2D(32, 3, 3, activation='relu')(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = model.add(Dropout(0.5))(x)

    steering_output = Dense(1, activation='linear', name='steering_output')(x)
    throttle_output = Dense(1, activation='linear', name='throttle_output')(x)
    brake_output = Dense(1, activation='linear', name='brake_output')(x)
    speed_output = Dense(1, activation='linear', name='speed_output')(x)

    model = Model(input=[left_input, center_input, right_input],
                  output=[steering_output, throttle_output,
                          brake_output, speed_output])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.fit([left_images, center_images, right_images],
               [steering, throttle, brake, speed],
              nb_epochs=50, batch_size=100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavior cloning training')
    parser.add_argument('--model', dest='model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--data', dest='data', type=str,
                        help='Path to data that should be used to train model')
    args = parser.parse_args()
    build_model(args.model, args.data)

