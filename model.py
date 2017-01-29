import argparse
import numpy as np
import pandas as pd
from skimage import io
from skimage import transform

from keras.models import Sequential, Model
from keras.layers import Convolution2D
from keras.layers import Activation, Dropout, Flatten, Dense, Input, Merge
from keras.layers import MaxPooling2D, merge
from keras.layers.normalization import BatchNormalization


def load_images(files, data_path):
    def load_func(files):
        for fl in files:
            img = transform.downscale_local_mean(io.imread('%s/%s' %
                                                 (data_path, fl.strip())),
                                       factors=(4,4,1))
            yield img

    images = np.stack(load_func(files[0:100]), axis=-1)
    images = images.reshape([images.shape[3], images.shape[2],
                           images.shape[0], images.shape[1]])
    images = images.astype('float32')
    images = images/256.0
    return images


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
    steering = drive_log['steering'][0:100]
    throttle = drive_log['throttle'][0:100]
    brake = drive_log['brake'][0:100]
    speed = drive_log['speed'][0:100]

    left_img = Input(shape=(3, 40, 80), dtype='float32', name="left_img")
    left_input = BatchNormalization()(left_img)
    left_input = Convolution2D(32, 5, 5, border_mode='same')(left_input)
    left_input = Activation('relu')(left_input)
    left_input = Dropout(0.5)(left_input)
    right_img = Input(shape=(3, 40, 80), dtype='float32', name="right_img")
    right_input = BatchNormalization()(right_img)
    right_input = Convolution2D(32, 5, 5, border_mode='same')(right_input)
    right_input = Activation('relu')(right_input)
    right_input = Dropout(0.5)(right_input)
    center_img = Input(shape=(3, 40, 80), dtype='float32', name="center_img")
    center_input = BatchNormalization()(center_img)
    center_input = Convolution2D(32, 5, 5, border_mode='same')(center_input)
    center_input = Activation('relu')(center_input)
    center_input = Dropout(0.5)(center_input)

    x = merge([left_input, center_input, right_input], mode='concat')
    x = Convolution2D(3, 5, 5, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(1,1), border_mode='same')(x)
    x = Dropout(0.5)(x)
    x = Convolution2D(64, 3, 3, border_mode='same', activation='relu')(x)
    x = Convolution2D(64, 5, 5, border_mode='same', activation='relu')(x)
    x = MaxPooling2D((3, 3), strides=(1,1), border_mode='same')(x)
    x = Dropout(0.5)(x)

    x = Flatten()(x)
    steering_output = Dense(1, activation='sigmoid', name='steering_output')(x)
    throttle_output = Dense(1, activation='linear', name='throttle_output')(x)
    brake_output = Dense(1, activation='linear', name='brake_output')(x)
    speed_output = Dense(1, activation='linear', name='speed_output')(x)

    model = Model(input=[left_img, center_img, right_img],
                  output=[steering_output, throttle_output,
                          brake_output, speed_output])
    model.compile(optimizer='rmsprop', loss='mean_squared_error')
    model.fit(x=[left_images, center_images, right_images],
               y=[steering, throttle, brake, speed],
              nb_epoch=50, batch_size=100)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Behavior cloning training')
    parser.add_argument('--model', dest='model', type=str,
                        help='Path to model definition json. Model weights should be on the same path.')
    parser.add_argument('--data', dest='data', type=str,
                        help='Path to data that should be used to train model')
    args = parser.parse_args()
    build_model(args.model, args.data)

