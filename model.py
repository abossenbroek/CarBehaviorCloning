import argparse
import numpy as np
import cv2
import os.path


from keras.models import Model
from keras.layers import Conv2D
from keras.layers.merge import add
from keras import regularizers
from keras.layers import Dropout, Flatten, Dense, Input, Lambda
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import model_from_json
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint
from keras.layers.pooling import AveragePooling2D

MISSING = object()

image_counter = 0

def nvidia_model(input):
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same',
            kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(input)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(24, (5, 5), strides=(2, 2), padding='same', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(36, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(24, (5, 5), padding='same', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = add([x, resnet_in])
    x = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D(pool_size=(4,4), strides=(1,1), padding='valid')(x)

    x = Flatten()(x)
   # x = Dense(1164, activation="elu", kernel_regularizer=regularizers.l2(0.001),
   #         kernel_initializer='glorot_normal')(x)
    x = Dense(512, kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(x)
    x = Dense(100, kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(x)
    x = Dense(50, kernel_regularizer=regularizers.l2(0.001),
            kernel_initializer='glorot_normal')(x)
    x = Dense(1, activation='tanh')(x)
    return x


def preprocess_img(file, fromColorSpace="BGR"):
    if not os.path.isfile(file):
        print("%s could not be found" % file)

    img = cv2.imread(file)
    if fromColorSpace == "BGR":
       convert = cv2.COLOR_BGR2YUV
    else:
        convert = cv2.COLOR_RGB2YUV
    img = cv2.cvtColor(img, convert)
    img = img[50:140,:,:]
    img = cv2.resize(img, (100,66), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, np.float32)
    return img

# From https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brightness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_bright = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_bright
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(cv2.cvtColor(image1, cv2.COLOR_HSV2RGB),
                          cv2.COLOR_RGB2YUV)
    return image1

def process_line(line, log_path):
    center_image_nm, left_image_nm, right_image_nm, angle, throttle, brake, speed = line.split(",")
    # Generate the angles
    center_angle = float(angle)
    # Add random angle adjustment.
    left_angle = center_angle + np.random.uniform(low=0.10, high=0.25)
    right_angle = center_angle - np.random.uniform(low=0.10, high=0.25)
    # Load the images
    left_img = preprocess_img(log_path + "/" + left_image_nm.strip())
    right_img = preprocess_img(log_path + "/" + right_image_nm.strip())
    center_img = preprocess_img(log_path + "/" + center_image_nm.strip())

    # Always add a flipped image
    center_flipped_angle = -center_angle
    center_flipped_img = np.fliplr(center_img)
    right_flipped_angle = -(center_angle - np.random.uniform(low=0.10, high=0.25))
    right_flipped_img = np.fliplr(right_img)
    left_flipped_angle = -(center_angle + np.random.uniform(low=0.10, high=0.25))
    left_flipped_img = np.fliplr(left_img)

    center_br_img = augment_brightness_camera_images(center_img)
    left_br_img = augment_brightness_camera_images(left_img)
    right_br_img = augment_brightness_camera_images(right_img)

    rows, cols, colors = center_img.shape

    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-5,5), 1)
    center_img = cv2.warpAffine(center_img, rot_mat, (cols, rows))

    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-5,5), 1)
    left_img = cv2.warpAffine(left_img, rot_mat, (cols, rows))

    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2), np.random.uniform(-5,5), 1)
    right_img = cv2.warpAffine(right_img, rot_mat, (cols, rows))

    x = np.concatenate((center_img[np.newaxis, ...],
                        right_img[np.newaxis, ...],
                        left_img[np.newaxis, ...],
                        center_flipped_img[np.newaxis, ...],
                        right_flipped_img[np.newaxis, ...],
                        left_flipped_img[np.newaxis, ...],
                        center_br_img[np.newaxis, ...],
                        right_br_img[np.newaxis, ...],
                        left_br_img[np.newaxis, ...],
                        ), axis=0)
    y = np.vstack((center_angle, right_angle, left_angle, center_flipped_angle,
                   right_flipped_angle, left_flipped_angle,
                   center_angle, right_angle, left_angle,))

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


def build_model(model_path, data_path, epochs, new_data=MISSING, model_file=MISSING):
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


    if model_file is not MISSING:
        print("Loading model from %s" % model_file)
        with open(model_file, 'r') as jfile:
            model = model_from_json(jfile.read())

        weights_file = model_file.replace('json', 'h5')
        model.load_weights(weights_file)

    model.compile(optimizer='adamax',
                  loss='mse')
    print(model.summary())

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1,
            save_best_only=True)
    csv_logger = CSVLogger('training.log')

    model.fit(x=X,
              y=y,
              epochs=epochs, batch_size=128, validation_split=0.10,
              shuffle=True,
              callbacks=[early_stopping,
                  checkpointer,
                  csv_logger])

    if new_data is not MISSING:
        model.fit_generator(generate_input_from_file(new_data + "/driving_log.csv", new_data),
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
    parser.add_argument('--new_data', dest='new_data', type=str,
                        help='Location of new data')

    args = parser.parse_args()
    if args.load and args.new_data:
        build_model(args.model, args.data, args.epochs, args.new_data, args.load)
    elif args.load:
        build_model(args.model, args.data, args.epochs, MISSING, args.load)
    else:
        build_model(args.model, args.data, args.epochs, MISSING, MISSING)

