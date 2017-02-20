import argparse
import numpy as np
import pandas as pd
from skimage import io

from keras.models import Model
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import Dropout, Flatten, Dense, Input, merge
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
    print("Using nvidia architecture")
    x = Convolution2D(3, 5, 5, border_mode='same')(input)
    x = ELU()(x)
    x = Convolution2D(24, 5, 5, border_mode='same')(x)
    x = ELU()(x)
    x = Convolution2D(36, 5, 5, border_mode='same')(x)
    x = ELU()(x)
    x = Dropout(0.25)(x)
    x = Convolution2D(48, 3, 3, border_mode='same')(x)
    x = ELU()(x)
    x = Dropout(0.25)(x)
    x = Convolution2D(64, 3, 3, border_mode='same')(x)
    x = ELU()(x)
    x = Dropout(0.25)(x)

    x = Flatten()(x)
    x = Dense(512)(x)
    x = Dense(100)(x)
    x = Dense(50)(x)
    x = Dense(1)(x)
    return x

def squeezenet(input):
    """ Specify the squeeze net architecture on an input.

    Code is taken from: https://github.com/DT42/squeezenet_demo
    The squeeze net network is described in https://arxiv.org/pdf/1602.07360.pdf

    Keyword arguments:
    input -- the keras input that should be used as input to squeezenet
    """
    print("Using squeezenet architecture")
    conv1 = Convolution2D(
        96, 7, 7, activation='relu', init='glorot_uniform',
        subsample=(2, 2), border_mode='same', name='conv1')(input)
    maxpool1 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), border_mode='same', name='maxpool1')(conv1)

    fire2_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_squeeze')(maxpool1)
    fire2_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand1')(fire2_squeeze)
    fire2_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire2_expand2')(fire2_squeeze)
    merge2 = merge(
        [fire2_expand1, fire2_expand2], mode='concat', concat_axis=1)

    fire3_squeeze = Convolution2D(
        16, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_squeeze')(merge2)
    fire3_expand1 = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand1')(fire3_squeeze)
    fire3_expand2 = Convolution2D(
        64, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire3_expand2')(fire3_squeeze)
    merge3 = merge(
        [fire3_expand1, fire3_expand2], mode='concat', concat_axis=1)

    fire4_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_squeeze')(merge3)
    fire4_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand1')(fire4_squeeze)
    fire4_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire4_expand2')(fire4_squeeze)
    merge4 = merge(
        [fire4_expand1, fire4_expand2], mode='concat', concat_axis=1)
    maxpool4 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool4')(merge4)

    fire5_squeeze = Convolution2D(
        32, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_squeeze')(maxpool4)
    fire5_expand1 = Convolution2D(
        128, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand1')(fire5_squeeze)
    fire5_expand2 = Convolution2D(
        128, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire5_expand2')(fire5_squeeze)
    merge5 = merge(
        [fire5_expand1, fire5_expand2], mode='concat', concat_axis=1)

    fire6_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_squeeze')(merge5)
    fire6_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand1')(fire6_squeeze)
    fire6_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire6_expand2')(fire6_squeeze)
    merge6 = merge(
        [fire6_expand1, fire6_expand2], mode='concat', concat_axis=1)

    fire7_squeeze = Convolution2D(
        48, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_squeeze')(merge6)
    fire7_expand1 = Convolution2D(
        192, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand1')(fire7_squeeze)
    fire7_expand2 = Convolution2D(
        192, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire7_expand2')(fire7_squeeze)
    merge7 = merge(
        [fire7_expand1, fire7_expand2], mode='concat', concat_axis=1)

    fire8_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_squeeze')(merge7)
    fire8_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand1')(fire8_squeeze)
    fire8_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire8_expand2')(fire8_squeeze)
    merge8 = merge(
        [fire8_expand1, fire8_expand2], mode='concat', concat_axis=1)

    maxpool8 = MaxPooling2D(
        pool_size=(3, 3), strides=(2, 2), name='maxpool8')(merge8)

    fire9_squeeze = Convolution2D(
        64, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_squeeze')(maxpool8)
    fire9_expand1 = Convolution2D(
        256, 1, 1, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand1')(fire9_squeeze)
    fire9_expand2 = Convolution2D(
        256, 3, 3, activation='relu', init='glorot_uniform',
        border_mode='same', name='fire9_expand2')(fire9_squeeze)
    merge9 = merge(
        [fire9_expand1, fire9_expand2], mode='concat', concat_axis=1)

    fire9_dropout = Dropout(0.5, name='fire9_dropout')(merge9)
    conv10 = Convolution2D(
        10, 1, 1, init='glorot_uniform',
        border_mode='valid', name='conv10')(fire9_dropout)
    # The size should match the output of conv10
    avgpool10 = AveragePooling2D((13, 13), border_mode='same', name='avgpool10')(conv10)

    flatten = Flatten(name='flatten')(avgpool10)
    return flatten

def build_model(model_path, data_path, epochs, threshold, arch, load=MISSING):
    # Load the log file.
    drive_log = pd.read_csv("%s/driving_log.csv" % (data_path))
    # Load the training labels.
    steering = drive_log['steering']

    print("min absolute driving angle %s max driving angle %s" % (
        min(abs(steering)), max(abs(steering))))

    print("original data size %s, with threshold we keep %s" % (
        len(steering), sum(abs(steering) > threshold)))

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
        (steering + np.random.uniform(low=0, high=0.15, size = len(steering)),
         steering,
         steering - np.random.uniform(low=0, high=0.15, size = len(steering)),
         -(steering + np.random.uniform(low=0, high=0.15, size = len(steering))),
         -steering,
         -(steering - np.random.uniform(low=0, high=0.15, size = len(steering)))))

    img_input = Input(shape=(3, 90, 320), dtype='float32', name="images")
    input = BatchNormalization()(img_input)

    if arch == 'nvidia':
        x = nvidia_model(input)
    else:
        x = squeezenet(input)

    steering_output = Dense(1, activation='linear', name='steering_output')(x)

    model = Model(input=img_input,
                  output=steering_output)
    model.compile(optimizer='adamax',
                  loss='mean_absolute_percentage_error')
    print(model.summary())

    if load is not MISSING:
        model_file = "%s/model.json" % (load)
        weights_file = "%s/model.h5" % (load)
        print("Loading model from %s" % (load))
        model = model_from_json(model_file)
        model.load_weights(weights_file)

    early_stopping = EarlyStopping(monitor='val_loss', patience=30)
    csv_logger = CSVLogger('training.log')

    model.fit(x=images,
              y=steering,
              nb_epoch=epochs, batch_size=200, validation_split=0.75,
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

def archSpecification(string):
    if not(string == "nvidia" or string == "squeeze"):
        raise argparse.ArgumentError('Value should be either nvidia or squeeze')
    return string

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
    parser.add_argument('--arch', dest='arch', type=archSpecification,
                        help='The model architecture that should be used (nvidia or squeeze).')

    args = parser.parse_args()
    build_model(args.model, args.data, args.epochs, args.threshold, args.arch)
