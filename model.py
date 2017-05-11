import argparse
import numpy as np
import cv2
import os.path
from tqdm import tqdm

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
MIN_SIDE_ANGLE = 0.03
MAX_SIDE_ANGLE = 0.07


def nvidia_model(input):
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(input)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(32, (5, 5), strides=(2, 2), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=(2, 2), padding='same')(x)
    resnet_in = x
    x = Conv2D(32, (5, 5), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (5, 5), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    resnet_in = x
    x = Conv2D(32, (5, 5), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(32, (5, 5), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(64, (5, 5), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    resnet_in = ELU()(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(resnet_in)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Dropout(0.5)(x)
    x = Conv2D(64, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = add([x, resnet_in])
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = Conv2D(256, (3, 3), padding='same',
               kernel_regularizer=regularizers.l2(0.001),
               kernel_initializer='glorot_normal')(x)
    x = BatchNormalization()(x)
    x = ELU()(x)
    x = AveragePooling2D(pool_size=(4, 4), strides=(2, 2), padding='valid')(x)

    x = Flatten()(x)
    x = Dense(1164, activation="elu", kernel_regularizer=regularizers.l2(0.001),
              kernel_initializer='glorot_normal')(x)
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
    img = img[50:140, :, :]
    img = cv2.resize(img, (80, 80), interpolation=cv2.INTER_AREA)
    img = np.asarray(img, np.float32)
    return img


# From https://chatbotslife.com/using-augmentation-to-mimic-human-driving-496b569760a9
def augment_brness_camera_images(image):
    image1 = cv2.cvtColor(image, cv2.COLOR_YUV2RGB)
    image1 = cv2.cvtColor(image1, cv2.COLOR_RGB2HSV)
    image1 = np.array(image1, dtype=np.float64)
    random_br = .5 + np.random.uniform()
    image1[:, :, 2] = image1[:, :, 2] * random_br
    image1[:, :, 2][image1[:, :, 2] > 255] = 255
    image1 = np.array(image1, dtype=np.uint8)
    image1 = cv2.cvtColor(cv2.cvtColor(image1, cv2.COLOR_HSV2RGB),
                          cv2.COLOR_RGB2YUV)
    return image1


def process_line(line, log_path):
    c_image_nm, l_image_nm, r_image_nm, ang, throttle, brake, speed = line.split(",")

    # Load the images
    l_img = preprocess_img(log_path + "/" + l_image_nm.strip())
    r_img = preprocess_img(log_path + "/" + r_image_nm.strip())
    c_img = preprocess_img(log_path + "/" + c_image_nm.strip())

    # Generate the angs
    c_ang = float(ang)
    # Add random ang adjustment.
    l_ang = c_ang + np.random.uniform(low=MIN_SIDE_ANGLE,
                                               high=MAX_SIDE_ANGLE)
    r_ang = c_ang - np.random.uniform(low=MIN_SIDE_ANGLE,
                                               high=MAX_SIDE_ANGLE)
    c_flp_ang = -c_ang
    r_flp_ang = -(c_ang - np.random.uniform(low=MIN_SIDE_ANGLE,
                                                 high=MAX_SIDE_ANGLE))
    l_flp_ang = -(c_ang + np.random.uniform(low=MIN_SIDE_ANGLE,
                                                 high=MAX_SIDE_ANGLE))
    # Always add a flipped image
    c_flp_img = np.fliplr(c_img)
    r_flp_img = np.fliplr(r_img)
    l_flp_img = np.fliplr(l_img)

    c_br_img = augment_brness_camera_images(c_img)
    l_br_img = augment_brness_camera_images(l_img)
    r_br_img = augment_brness_camera_images(r_img)

    rows, cols, colors = c_img.shape

    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2),
                                      np.random.uniform(-5, 5), 1)
    c_img = cv2.warpAffine(c_img, rot_mat, (cols, rows))

    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2),
                                      np.random.uniform(-5, 5), 1)
    l_img = cv2.warpAffine(l_img, rot_mat, (cols, rows))

    rot_mat = cv2.getRotationMatrix2D((cols/2, rows/2),
                                      np.random.uniform(-5, 5), 1)
    r_img = cv2.warpAffine(r_img, rot_mat, (cols, rows))

    x = np.concatenate((c_img[np.newaxis, ...], r_img[np.newaxis, ...],
                        l_img[np.newaxis, ...],
                        c_flp_img[np.newaxis, ...], r_flp_img[np.newaxis, ...],
                        l_flp_img[np.newaxis, ...],
                        c_br_img[np.newaxis, ...], r_br_img[np.newaxis, ...],
                        l_br_img[np.newaxis, ...],), axis=0)
    y = np.vstack((c_ang, r_ang, l_ang, c_flp_ang,
                   r_flp_ang, l_flp_ang,
                   c_ang, r_ang, l_ang,))

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
    for i, line in enumerate(tqdm(f)):
        if i < 1:
            continue
        else:
            x, y = process_line(line, log_path)
            X.append(x)
            Y.append(y)
    f.close()

    return [np.concatenate(X, axis=0), np.concatenate(Y, axis=0)]


def build_model(model_path, data_path, epochs, new_data=MISSING,
                model_file=MISSING):
    # Load the log file.
    drive_log = "%s/driving_log.csv" % data_path

    X, y = load_original_file(drive_log, data_path)

    #X = X.reshape(X.shape[0], X.shape[3], X.shape[1], X.shape[2])

    img_input = Input(shape=X.shape[1:], dtype='float32', name="images")

    input = Lambda(lambda x: x/127.5 - 1.0)(img_input)

    x = nvidia_model(input)

    steering_output = Dense(1, activation='linear', name='steering_output')(x)

    model = Model(inputs=img_input,
                  outputs=steering_output)

    if model_file is not MISSING:
        print("Loading model from %s" % model_file)
        with open(model_file, 'r') as jfile:
            model = model_from_json(jfile.read())

        weights_file = model_file.replace('json', 'h5')
        model.load_weights(weights_file)

    print(model.summary())

    model.compile(optimizer='adamax', loss='mse')

    early_stopping = EarlyStopping(monitor='val_loss', patience=10)
    checkpointer = ModelCheckpoint(filepath="./weights.hdf5", verbose=1,
                                   save_best_only=True)
    csv_logger = CSVLogger('training.log')

    model.fit(x=X,
              y=y,
              epochs=epochs, batch_size=1024, validation_split=0.10,
              shuffle=True,
              callbacks=[early_stopping,
                         checkpointer,
                         csv_logger])

    if new_data is not MISSING:
        file_path = new_data + "/driving_log.csv"
        model.fit_generator(generate_input_from_file(file_path, new_data),
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
        build_model(args.model, args.data, args.epochs,
                    args.new_data, args.load)
    elif args.load:
        build_model(args.model, args.data, args.epochs,
                    MISSING, args.load)
    else:
        build_model(args.model, args.data, args.epochs,
                    MISSING, MISSING)

